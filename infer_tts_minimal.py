import os
import sys
import math
import platform
import soundfile as sf
import onnxruntime as ort
import numpy as np
import cProfile
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

profiler = cProfile.Profile()
profiler.enable()  # 여기서부터 기록 시작
# GPT-SoVITS 모듈 내부 경로 강제 주입
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

from TTS_infer_pack.TTS import TTS, TTS_Config


def get_container_cpu_threads() -> int:
    """Docker / Google Cloud Run Gen2(cgroups v1/v2)의 할당된 vCPU 코어 수를 감지합니다."""
    env_threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get("ORT_INTRA_OP_NUM_THREADS")
    if env_threads and env_threads.isdigit():
        return max(1, int(env_threads))

    # Linux cgroups v2 (Cloud Run Gen2 기본 환경)
    try:
        if os.path.exists("/sys/fs/cgroup/cpu.max"):
            with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
                if len(parts) == 2 and parts[0] != "max":
                    quota, period = int(parts[0]), int(parts[1])
                    if quota > 0 and period > 0:
                        return max(1, math.ceil(quota / period))
    except Exception:
        pass

    # Linux cgroups v1 폴백
    try:
        if os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") and os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="utf-8") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8") as f:
                period = int(f.read().strip())
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except Exception:
        pass

    # 로컬 PC 환경 (Windows 노트북 등): 가용 논리 CPU/스레드 수 전체 활용
    logical_cpus = os.cpu_count() or 4
    return max(1, logical_cpus)


def detect_cpu_vendor() -> str:
    """Windows 및 Linux(Docker/Cloud Run) 환경에서 CPU 제조사(intel / amd / generic)를 정확히 판별합니다."""
    proc = platform.processor().lower()
    if "intel" in proc or "genuineintel" in proc:
        return "intel"
    if "amd" in proc or "authenticamd" in proc:
        return "amd"

    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                cpuinfo = f.read().lower()
                if "genuineintel" in cpuinfo or "intel" in cpuinfo:
                    return "intel"
                if "authenticamd" in cpuinfo or "amd" in cpuinfo:
                    return "amd"
        except Exception:
            pass
    return "generic"


def create_ort_session_options(num_threads: int) -> ort.SessionOptions:
    """컨테이너 vCPU 코어 수에 맞춘 최적화된 ONNX Runtime SessionOptions를 생성합니다."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return sess_options


def get_onnx_providers(num_threads: int) -> list:
    """Google Cloud Run Gen2 (Intel Xeon / AMD EPYC) 환경에 맞춰 최적의 프로바이더를 구성합니다."""
    vendor = detect_cpu_vendor()
    available = ort.get_available_providers()

    if vendor == "intel" and "OpenVINOExecutionProvider" in available:
        if sys.platform == "win32":
            try:
                import openvino
                ov_libs_dir = os.path.join(os.path.dirname(openvino.__file__), "libs")
                if os.path.isdir(ov_libs_dir):
                    os.environ["PATH"] = ov_libs_dir + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(ov_libs_dir)
            except Exception:
                pass
        print(f"🚀 [Intel CPU 감지 ({num_threads} Threads)] OpenVINOExecutionProvider 가속을 활성화합니다!")
        return [
            (
                "OpenVINOExecutionProvider",
                {
                    "device_type": "CPU",
                    "num_of_threads": num_threads,
                    "enable_opencl_throttling": False,
                },
            ),
            "CPUExecutionProvider",
        ]

    print(f"⚡ [{vendor.upper()} CPU 감지 ({num_threads} Threads)] 기본 CPUExecutionProvider로 구동합니다.")
    return ["CPUExecutionProvider"]


class ONNXT2SAdapter:
    """GPT(Text2Semantic) 3단계 분할 ONNX 모델(encoder, fsdec, sdec) 순수 NumPy 어댑터"""

    def __init__(self, enc_path: str, fsdec_path: str, sdec_path: str, providers: list, num_threads: int):
        print("⚡ C++ 백엔드 기반 Full ONNX GPT(T2S) 3단계 어댑터 세팅 중...")
        sess_options = create_ort_session_options(num_threads)

        self.encoder_session = ort.InferenceSession(enc_path, sess_options=sess_options, providers=providers)
        self.fsdec_session = ort.InferenceSession(fsdec_path, sess_options=sess_options, providers=providers)
        self.sdec_session = ort.InferenceSession(sdec_path, sess_options=sess_options, providers=providers)
        print(f"✅ 활성화된 GPT ONNX 프로바이더: {self.sdec_session.get_providers()}")

        self.EOS = 1024
        self.dummy_ssl = np.zeros((1, 768, 10), dtype=np.float32)
        # TTS.py의 self.t2s_model.model.infer_panel 호출 호환용
        self.model = self
        self.infer_panel_batch_infer = self.infer_panel
        self.infer_panel_naive_batched = self.infer_panel

    def infer_panel(
        self,
        all_phoneme_ids,
        all_phoneme_lens,
        prompt,
        all_bert_features,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        early_stop_num: int = 2700,
        max_len: int = 0,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        prompts_np = np.ascontiguousarray(prompt[:1], dtype=np.int64)
        prefix_len = prompts_np.shape[1]

        pred_semantic_list = []
        idx_list = []

        for ph_ids, bert_feat in zip(all_phoneme_ids, all_bert_features):
            ph_np = np.asarray(ph_ids, dtype=np.int64)
            bert_np = np.asarray(bert_feat, dtype=np.float32).T  # [L, 1024]

            ref_seq = np.ascontiguousarray(ph_np[:1][None, :])
            text_seq = np.ascontiguousarray(ph_np[1:][None, :])
            ref_bert = np.ascontiguousarray(bert_np[:1, :])
            text_bert = np.ascontiguousarray(bert_np[1:, :])

            # 1. T2S Encoder
            x = self.encoder_session.run(
                ["x"],
                {
                    "ref_seq": ref_seq,
                    "text_seq": text_seq,
                    "ref_bert": ref_bert,
                    "text_bert": text_bert,
                    "ssl_content": self.dummy_ssl,
                },
            )[0]

            # 2. First Stage Decoder (Prefill & KV-Cache 초기화)
            fsdec_io = self.fsdec_session.io_binding()
            fsdec_io.bind_cpu_input("x", x)
            fsdec_io.bind_cpu_input("prompts", prompts_np)
            for out_name in ["y", "k", "v", "y_emb", "x_example"]:
                fsdec_io.bind_output(out_name)

            self.fsdec_session.run_with_iobinding(fsdec_io)
            outputs = fsdec_io.get_outputs()

            # Use raw OrtValues in C++ without fetching to Python
            y, k, v, y_emb, x_example = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

            # 3. Stage Decoder (Autoregressive Step Loop)
            idx = 1
            sdec_io_1 = self.sdec_session.io_binding()
            sdec_io_2 = self.sdec_session.io_binding()
            io_bindings = [sdec_io_1, sdec_io_2]

            # Initial setup for io_1
            sdec_io_1.bind_ortvalue_input("iy", y)
            sdec_io_1.bind_ortvalue_input("ik", k)
            sdec_io_1.bind_ortvalue_input("iv", v)
            sdec_io_1.bind_ortvalue_input("iy_emb", y_emb)
            sdec_io_1.bind_ortvalue_input("ix_example", x_example)
            for out_name in ["y", "k", "v", "y_emb", "logits", "samples"]:
                sdec_io_1.bind_output(out_name)

            for idx in tqdm(range(1, 1500)):
                io_curr = io_bindings[(idx - 1) % 2]
                io_next = io_bindings[idx % 2]

                self.sdec_session.run_with_iobinding(io_curr)
                outputs = io_curr.get_outputs()

                y, k, v, y_emb, logits, samples = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

                # Bind the newly produced OrtValues to the NEXT IOBinding object (Ping-Pong)
                io_next.clear_binding_inputs()
                io_next.clear_binding_outputs()

                io_next.bind_ortvalue_input("iy", y)
                io_next.bind_ortvalue_input("ik", k)
                io_next.bind_ortvalue_input("iv", v)
                io_next.bind_ortvalue_input("iy_emb", y_emb)
                io_next.bind_ortvalue_input("ix_example", x_example)

                for out_name in ["y", "k", "v", "y_emb", "logits", "samples"]:
                    io_next.bind_output(out_name)

                y_shape = y.shape()

                stop = False
                if early_stop_num != -1 and (y_shape[1] - prefix_len) > early_stop_num:
                    stop = True

                # Fetch only small tensors for early stopping check
                logits_np = logits.numpy()
                samples_np = samples.numpy()

                if int(np.argmax(logits_np, axis=-1)[0]) == self.EOS or int(samples_np[0, 0]) == self.EOS:
                    stop = True
                if stop:
                    break

            y_np = y.numpy()
            print(f"T2S ONNX Decoding EOS [{prefix_len} -> {y_np.shape[1]}]")
            y_np[0, -1] = 0
            pred_tokens = y_np[0, -idx:].copy().astype(np.int64)
            pred_semantic_list.append(pred_tokens)
            idx_list.append(idx)

        return pred_semantic_list, idx_list


class ONNXVitsAdapter:
    def __init__(self, vits_onnx_path: str, providers: list, num_threads: int):
        print("⚡ C++ 백엔드 기반 Full ONNX VITS 디코더 어댑터 세팅 중...")
        sess_options = create_ort_session_options(num_threads)
        self.vits_session = ort.InferenceSession(vits_onnx_path, sess_options=sess_options, providers=providers)
        print(f"✅ 활성화된 VITS ONNX 프로바이더: {self.vits_session.get_providers()}")

        self.upsample_rates = [10, 8, 2, 2, 2]
        self.ref_audio_sr = None

    def decode(self, pred_semantic: np.ndarray, text_seq: np.ndarray, refer_audio_spec, speed: float = 1.0) -> np.ndarray:
        onnx_inputs = {
            "text_seq": np.ascontiguousarray(text_seq, dtype=np.int64),
            "pred_semantic": np.ascontiguousarray(pred_semantic, dtype=np.int64),
            "ref_audio": self.ref_audio_sr,
        }
        audio_onnx = self.vits_session.run(["audio"], onnx_inputs)[0]
        return audio_onnx


class HybridTTS(TTS):
    def __init__(
        self,
        configs: TTS_Config,
        gpt_enc_path: str,
        gpt_fsdec_path: str,
        gpt_sdec_path: str,
        vits_onnx_path: str,
        ref_cache_path: str,
        num_threads: int,
    ):
        self.gpt_enc_path = gpt_enc_path
        self.gpt_fsdec_path = gpt_fsdec_path
        self.gpt_sdec_path = gpt_sdec_path
        self.vits_onnx_path = vits_onnx_path
        self.num_threads = num_threads
        self.onnx_providers = get_onnx_providers(num_threads)

        print(f"📦 사전 베이킹된 NumPy 레퍼런스 캐시 로드 중: {os.path.basename(ref_cache_path)}")
        npz_data = np.load(ref_cache_path, allow_pickle=True)
        self.ref_cache = {
            "prompt_text": str(npz_data["prompt_text"].item() if npz_data["prompt_text"].ndim == 0 else npz_data["prompt_text"]),
            "prompt_lang": str(npz_data["prompt_lang"].item() if npz_data["prompt_lang"].ndim == 0 else npz_data["prompt_lang"]),
            "phones": npz_data["phones"].tolist(),
            "bert_features": np.asarray(npz_data["bert_features"], dtype=np.float32),
            "norm_text": str(npz_data["norm_text"].item() if npz_data["norm_text"].ndim == 0 else npz_data["norm_text"]),
            "audios": npz_data["audios"].item(),
        }

        super().__init__(configs)

    def init_bert_weights(self, base_path: str):
        print("🔪 중국어 전용 BERT(1.24GB) 로드를 건너뜁니다!")
        self.bert_tokenizer = None
        self.bert_model = None

    def init_cnhuhbert_weights(self, base_path: str):
        print("🔪 사전 베이킹 캐시(ref_cache.npz)를 사용하므로 CNHuBERT(360MB) 로드를 건너뜁니다!")
        self.cnhuhbert_model = None

    def init_t2s_weights(self, weights_path: str):
        print("🔪 PyTorch GPT(.ckpt/.safetensors) 로드를 건너뛰고 Full ONNX GPT(Encoder + FSDec + SDec) 래퍼를 주입합니다!")
        self.configs.hz = 50
        self.configs.max_sec = 54
        self.t2s_model = ONNXT2SAdapter(
            self.gpt_enc_path,
            self.gpt_fsdec_path,
            self.gpt_sdec_path,
            self.onnx_providers,
            self.num_threads,
        )

    def init_vits_weights(self, weights_path: str):
        print("🔪 PyTorch VITS(.pth) 및 Latent ONNX 로드를 건너뛰고 ONNX VITS 디코더 래퍼를 주입합니다!")
        self.configs.update_version("v2")
        self.vits_model = ONNXVitsAdapter(
            self.vits_onnx_path,
            self.onnx_providers,
            self.num_threads,
        )

    def set_ref_audio(self, ref_audio_path: str):
        """오디오 파일 파싱 및 HuBERT 추론 대신 ref_cache.npz에서 즉시 NumPy 배열을 주입합니다."""
        audios_dict = self.ref_cache["audios"]
        basename = os.path.basename(ref_audio_path)
        stem = os.path.splitext(basename)[0]

        if ref_audio_path in audios_dict:
            entry = audios_dict[ref_audio_path]
        elif basename in audios_dict:
            entry = audios_dict[basename]
        elif stem in audios_dict:
            entry = audios_dict[stem]
        else:
            available_keys = [k for k in audios_dict.keys() if k.endswith(".wav")]
            raise KeyError(
                f"캐시(ref_cache.npz)에서 '{ref_audio_path}'를 찾을 수 없습니다. "
                f"사용 가능한 오디오: {available_keys}"
            )

        self.prompt_cache["ref_audio_path"] = ref_audio_path
        self.prompt_cache["prompt_semantic"] = np.asarray(entry["prompt_semantic"], dtype=np.int64)
        self.prompt_cache["refer_spec"] = [np.empty(0, dtype=np.float32)]
        self.prompt_cache["prompt_text"] = self.ref_cache["prompt_text"]
        self.prompt_cache["prompt_lang"] = self.ref_cache["prompt_lang"]
        self.prompt_cache["phones"] = self.ref_cache["phones"]
        self.prompt_cache["bert_features"] = self.ref_cache["bert_features"]
        self.prompt_cache["norm_text"] = self.ref_cache["norm_text"]
        self.vits_model.ref_audio_sr = np.asarray(entry["ref_audio_sr"], dtype=np.float32)
        print(f"⚡ [캐시 히트] '{basename}' 특징 배열(NumPy) 즉시 로드 완료!")


def main():
    print("🚀 100% Pure NumPy + Full ONNX (PyTorch 제로 의존성) 파이프라인 초기화 중...")

    num_threads = get_container_cpu_threads()
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    config_path = os.path.join(now_dir, "GPT_SoVITS", "configs", "tts_infer.yaml")
    config = TTS_Config(config_path)
    config.device = "cpu"
    config.is_half = False

    gpt_enc_path = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_encoder.onnx")
    gpt_fsdec_path = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_fsdec.onnx")
    gpt_sdec_int8_path = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_sdec_int8.onnx")
    gpt_sdec_fp32_path = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_sdec.onnx")

    if os.path.exists(gpt_sdec_int8_path):
        gpt_sdec_path = gpt_sdec_int8_path
        print(f"📦 [INT8 양자화 모델 감지] 8비트 경량화 디코더를 로드합니다: {os.path.basename(gpt_sdec_int8_path)}")
    else:
        gpt_sdec_path = gpt_sdec_fp32_path
        print(f"📦 [FP32 기본 모델 감지] 원본 디코더를 로드합니다: {os.path.basename(gpt_sdec_fp32_path)}")

    onnx_vits_path = os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_vits.onnx")
    ref_cache_path = os.path.join(now_dir, "reference_audios", "ref_cache.npz")

    if not os.path.exists(ref_cache_path):
        raise FileNotFoundError(
            f"사전 베이킹 캐시 파일이 없습니다: {ref_cache_path}\n"
            f"먼저 'python preload.py'를 실행하여 캐시를 생성해주세요."
        )

    config.t2s_weights_path = gpt_sdec_path
    config.vits_weights_path = onnx_vits_path

    ref_audio_path = os.path.join(now_dir, "reference_audios", "01_ref_emoNormal026.wav")

    print("📦 프론트엔드 및 Full ONNX (GPT + VITS) 엔진 로딩 중...")
    tts_pipeline = HybridTTS(
        config,
        gpt_enc_path,
        gpt_fsdec_path,
        gpt_sdec_path,
        onnx_vits_path,
        ref_cache_path,
        num_threads,
    )

    request_data = {
        "text": "야호! 나는 즌다몬인 거시다! 오늘은 내 한국어 목소리를 테스트해서, 여러 모델을 비교해 본다고 들은 거시다. 후훗, 어떤 내 목소리가 제일 귀엽고, 즌다모찌에 대한 사랑을 잘 표현하는지 똑똑히 들어보란 거시다!",
        "text_lang": "ko",
        "ref_audio_path": ref_audio_path,
        "prompt_text": tts_pipeline.ref_cache["prompt_text"],
        "prompt_lang": tts_pipeline.ref_cache["prompt_lang"],
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut2",
    }

    print(f"🔍 [런타임 무결성 검증] PyTorch(torch) 모듈 로드 여부: {'torch' in sys.modules} (False여야 순수 NumPy)")
    print("🧠 실시간 음성 합성 진행 중...")

    output_audio = None
    sample_rate = config.sampling_rate

    try:
        for sr, audio in tts_pipeline.run(request_data):
            sample_rate = sr
            output_audio = audio if output_audio is None else np.concatenate([output_audio, audio])
    except Exception as e:
        print(f"🚨 추론 중 에러 발생: {e}")
        return

    if output_audio is not None:
        output_path = "zundamon_hybrid_final.wav"
        sf.write(output_path, output_audio, sample_rate)
        print(f"🎉 합성 대성공! PyTorch 없이 100% 순수 NumPy + ONNX로 '{output_path}'가 생성되었습니다.")
        print(f"🔍 [최종 무결성 검증] 추론 완료 후 PyTorch(torch) 로드 여부: {'torch' in sys.modules}")
        profiler.disable()  # 기록 종료
        profiler.dump_stats("tts_inference.prof")  # 결과 파일로 저장
        print("✅ 프로파일링 완료! tts_inference.prof 저장됨.")
    else:
        print("⚠️ 오디오 데이터가 생성되지 않았습니다.")


if __name__ == "__main__":
    main()
