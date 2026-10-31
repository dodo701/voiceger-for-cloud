import os
import sys
import math
import platform
import soundfile as sf
import onnxruntime as ort
import numpy as np
import cProfile
from typing import Generator, List, Tuple
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 프로젝트 루트 및 GPT-SoVITS 모듈 경로 등록
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, now_dir)
sys.path.insert(0, os.path.join(now_dir, "GPT_SoVITS"))

from TTS_infer_pack.TextPreprocessor import TextPreprocessor


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
    """Windows 및 Linux(Docker/Cloud Run) 환경에서 CPU 제조사(intel / amd / generic)를 판별합니다."""
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
    """GPT(Text2Semantic) 3단계 분할 ONNX 모델(encoder, fsdec, sdec) 제로카피 어댑터"""

    def __init__(self, enc_path: str, fsdec_path: str, sdec_path: str, providers: list, num_threads: int):
        sess_options = create_ort_session_options(num_threads)
        self.encoder_session = ort.InferenceSession(enc_path, sess_options=sess_options, providers=providers)
        self.fsdec_session = ort.InferenceSession(fsdec_path, sess_options=sess_options, providers=providers)
        self.sdec_session = ort.InferenceSession(sdec_path, sess_options=sess_options, providers=providers)

        self.EOS = 1024
        self.dummy_ssl = np.zeros((1, 768, 10), dtype=np.float32)

    def infer(
        self,
        phoneme_ids: np.ndarray,
        prompt_semantic: np.ndarray,
        bert_features: np.ndarray,
        top_k: int = 15,
        early_stop_num: int = 2700,
    ) -> Tuple[np.ndarray, int]:
        """단일 문장에 대해 GPT 억양/운율 토큰을 생성합니다."""
        prompts_np = np.ascontiguousarray(prompt_semantic[None, :], dtype=np.int64)
        prefix_len = prompts_np.shape[1]

        ph_np = np.asarray(phoneme_ids, dtype=np.int64)
        bert_np = np.asarray(bert_features, dtype=np.float32).T  # [L, 1024]

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
        y, k, v, y_emb, x_example = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

        # 3. Stage Decoder (Autoregressive Step Loop with Zero-Copy Ping-Pong)
        sdec_io_1 = self.sdec_session.io_binding()
        sdec_io_2 = self.sdec_session.io_binding()
        io_bindings = [sdec_io_1, sdec_io_2]

        sdec_io_1.bind_ortvalue_input("iy", y)
        sdec_io_1.bind_ortvalue_input("ik", k)
        sdec_io_1.bind_ortvalue_input("iv", v)
        sdec_io_1.bind_ortvalue_input("iy_emb", y_emb)
        sdec_io_1.bind_ortvalue_input("ix_example", x_example)
        for out_name in ["y", "k", "v", "y_emb", "logits", "samples"]:
            sdec_io_1.bind_output(out_name)

        idx = 1
        for idx in tqdm(range(1, 1500), leave=False):
            io_curr = io_bindings[(idx - 1) % 2]
            io_next = io_bindings[idx % 2]

            self.sdec_session.run_with_iobinding(io_curr)
            outputs = io_curr.get_outputs()

            y, k, v, y_emb, logits, samples = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

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

            logits_np = logits.numpy()
            samples_np = samples.numpy()
            if int(np.argmax(logits_np, axis=-1)[0]) == self.EOS or int(samples_np[0, 0]) == self.EOS:
                stop = True
            if stop:
                break

        y_np = y.numpy()
        y_np[0, -1] = 0
        pred_tokens = y_np[0, -idx:].copy().astype(np.int64)
        return pred_tokens, idx


class ONNXVitsAdapter:
    """VITS 오디오 신경망 디코더 ONNX 어댑터"""

    def __init__(self, vits_onnx_path: str, providers: list, num_threads: int):
        sess_options = create_ort_session_options(num_threads)
        self.vits_session = ort.InferenceSession(vits_onnx_path, sess_options=sess_options, providers=providers)
        self.ref_audio_sr = None

    def decode(self, pred_semantic: np.ndarray, text_seq: np.ndarray, speed: float = 1.0) -> np.ndarray:
        onnx_inputs = {
            "text_seq": np.ascontiguousarray(text_seq, dtype=np.int64),
            "pred_semantic": np.ascontiguousarray(pred_semantic, dtype=np.int64),
            "ref_audio": self.ref_audio_sr,
        }
        audio_onnx = self.vits_session.run(["audio"], onnx_inputs)[0]
        if audio_onnx.ndim == 3:
            audio_onnx = audio_onnx[0, 0, :]
        return audio_onnx


class ZundamonTTSEngine:
    """레거시 TTS.py 의존성을 100% 제거한 단일 자립형 Full ONNX + 순수 NumPy 음성 합성 엔진"""

    def __init__(
        self,
        gpt_enc_path: str = None,
        gpt_fsdec_path: str = None,
        gpt_sdec_path: str = None,
        vits_onnx_path: str = None,
        ref_cache_path: str = None,
        num_threads: int = None,
    ):
        self.num_threads = num_threads or get_container_cpu_threads()
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        self.onnx_providers = get_onnx_providers(self.num_threads)

        # 모델 기본 경로 세팅
        gpt_enc_path = gpt_enc_path or os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_encoder.onnx")
        gpt_fsdec_path = gpt_fsdec_path or os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_fsdec.onnx")
        vits_onnx_path = vits_onnx_path or os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_vits.onnx")
        ref_cache_path = ref_cache_path or os.path.join(now_dir, "reference_audios", "ref_cache.npz")

        if gpt_sdec_path is None:
            gpt_sdec_int8_path = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_sdec_int8.onnx")
            gpt_sdec_fp32_path = os.path.join(now_dir, "GPT_weights_v2", "zundamon_t2s_sdec.onnx")
            gpt_sdec_path = gpt_sdec_int8_path if os.path.exists(gpt_sdec_int8_path) else gpt_sdec_fp32_path

        # 1. 사전 베이킹 캐시 로드 (NPZ)
        print(f"📦 [캐시 로드] 사전 베이킹 레퍼런스 캐시: {os.path.basename(ref_cache_path)}")
        npz_data = np.load(ref_cache_path, allow_pickle=True)
        self.ref_cache = {
            "prompt_text": str(npz_data["prompt_text"].item() if npz_data["prompt_text"].ndim == 0 else npz_data["prompt_text"]),
            "prompt_lang": str(npz_data["prompt_lang"].item() if npz_data["prompt_lang"].ndim == 0 else npz_data["prompt_lang"]),
            "phones": npz_data["phones"].tolist(),
            "bert_features": np.asarray(npz_data["bert_features"], dtype=np.float32),
            "norm_text": str(npz_data["norm_text"].item() if npz_data["norm_text"].ndim == 0 else npz_data["norm_text"]),
            "audios": npz_data["audios"].item(),
        }

        # 2. 텍스트 전처리기 (순수 NumPy G2P)
        self.text_preprocessor = TextPreprocessor()

        # 3. ONNX 딥러닝 엔진 로드
        print(f"📦 [엔진 로드] Full ONNX GPT 3종 + VITS 디코더 로딩 중...")
        self.t2s = ONNXT2SAdapter(gpt_enc_path, gpt_fsdec_path, gpt_sdec_path, self.onnx_providers, self.num_threads)
        self.vits = ONNXVitsAdapter(vits_onnx_path, self.onnx_providers, self.num_threads)

        self.sampling_rate = 32000
        self.hz = 50
        self.max_sec = 54
        self.current_ref_name = None

        # 기본 감정 음원(01_ref_emoNormal026) 활성화
        self.set_ref_audio("01_ref_emoNormal026.wav")

    def warmup(self, warmup_text: str = "안녕"):
        """서버 시작 시 1회 호출하여 ONNX Runtime / OpenVINO 그래프 컴파일 및 메모리 할당을 사전에 완료합니다."""
        _ = self.generate_sentence(warmup_text, lang="ko")

    def set_ref_audio(self, ref_name: str):
        """캐시에서 특정 감정의 레퍼런스 오디오 특징 텐서를 즉시 전환합니다."""
        audios_dict = self.ref_cache["audios"]
        basename = os.path.basename(ref_name)
        stem = os.path.splitext(basename)[0]

        if ref_name in audios_dict:
            entry = audios_dict[ref_name]
        elif basename in audios_dict:
            entry = audios_dict[basename]
        elif stem in audios_dict:
            entry = audios_dict[stem]
        else:
            available_keys = [k for k in audios_dict.keys() if k.endswith(".wav")]
            raise KeyError(f"캐시에서 '{ref_name}'를 찾을 수 없습니다. 사용 가능: {available_keys}")

        self.current_ref_name = basename
        self.prompt_semantic = np.asarray(entry["prompt_semantic"], dtype=np.int64)
        self.vits.ref_audio_sr = np.asarray(entry["ref_audio_sr"], dtype=np.float32)

    def generate_sentence(
        self,
        text: str,
        lang: str = "ko",
        ref_name: str = None,
        top_k: int = 15,
        speed: float = 1.0,
    ) -> np.ndarray:
        """단일 문장 청크를 입력받아 float32 오디오 배열(shape: (N,))을 즉시 합성합니다."""
        if ref_name and ref_name != self.current_ref_name:
            self.set_ref_audio(ref_name)

        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32)

        # 1. 텍스트 G2P 음소 분석
        phones, bert_features, _ = self.text_preprocessor.segment_and_extract_feature_for_text(clean_text, lang, "v2")
        if not phones:
            return np.zeros(0, dtype=np.float32)

        # 2. 레퍼런스 프롬프트 텐서와 결합
        all_phones = np.asarray(self.ref_cache["phones"] + phones, dtype=np.int64)
        all_bert = np.concatenate([self.ref_cache["bert_features"], bert_features], axis=1).astype(np.float32, copy=False)

        # 3. GPT T2S 운율/억양 토큰 생성
        pred_tokens, idx = self.t2s.infer(
            all_phones,
            self.prompt_semantic,
            all_bert,
            top_k=top_k,
            early_stop_num=self.hz * self.max_sec,
        )

        # 4. VITS 오디오 파형 디코딩
        pred_semantic = pred_tokens[-idx:][None, None, :]
        text_seq = np.asarray(phones, dtype=np.int64)[None, :]
        audio = self.vits.decode(pred_semantic, text_seq, speed=speed)

        # 5. 피크 방지 정규화
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio

    def generate_sentence_pcm(self, text: str, **kwargs) -> bytes:
        """단일 문장을 입력받아 32kHz 16-bit Mono PCM 바이트 스트림 반환 (gRPC 직결용)"""
        audio = self.generate_sentence(text, **kwargs)
        pcm = (audio * 32767).astype(np.int16)
        return pcm.tobytes()

    def generate_stream(
        self,
        text: str,
        lang: str = "ko",
        text_split_method: str = "cut2",
        **kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """긴 텍스트를 문장 부호로 분할하여 문장 단위로 float32 오디오 배열을 yield (실시간 스트리밍용)"""
        sentences = self.text_preprocessor.pre_seg_text(text, lang, text_split_method)
        for s in sentences:
            if not s.strip():
                continue
            audio = self.generate_sentence(s, lang=lang, **kwargs)
            if len(audio) > 0:
                yield audio

    def generate_all(self, text: str, **kwargs) -> np.ndarray:
        """긴 텍스트 전체를 합성하여 하나의 완성된 float32 오디오 배열로 반환 (파일 저장 / 일괄용)"""
        chunks = list(self.generate_stream(text, **kwargs))
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        pause = np.zeros(int(self.sampling_rate * 0.3), dtype=np.float32)
        full = []
        for i, c in enumerate(chunks):
            full.append(c)
            if i < len(chunks) - 1:
                full.append(pause)
        return np.concatenate(full, axis=0)


def main():
    print("🚀 [자립형 단일 엔진 검증] TTS.py와 YAML 의존성 없이 단독 구동 시작...")

    engine = ZundamonTTSEngine()

    test_text = (
        "야호! 나는 즌다몬인 거시다! 오늘은 내 한국어 목소리를 테스트해서, 여러 모델을 비교해 본다고 들은 거시다. "
        "후훗, 어떤 내 목소리가 제일 귀엽고, 즌다모찌에 대한 사랑을 잘 표현하는지 똑똑히 들어보란 거시다!"
    )

    print(f"🔍 [런타임 무결성 검증] PyTorch(torch) 로드 여부: {'torch' in sys.modules} (False여야 함)")
    print("🧠 실시간 음성 합성 진행 중 (generate_all)...")

    output_audio = engine.generate_all(test_text, lang="ko")

    if len(output_audio) > 0:
        output_path = "zundamon_hybrid_final.wav"
        sf.write(output_path, output_audio, engine.sampling_rate)
        print(f"🎉 합성 대성공! TTS.py 없이 자립형 엔진으로 '{output_path}'가 완벽히 생성되었습니다.")
        print(f"🔍 [최종 무결성 검증] 추론 완료 후 PyTorch(torch) 로드 여부: {'torch' in sys.modules}")
    else:
        print("⚠️ 오디오 데이터가 생성되지 않았습니다.")


if __name__ == "__main__":
    main()
