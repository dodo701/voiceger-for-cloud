import os
import sys
import time
import math
import json
import random
import threading
import platform
import soundfile as sf
import onnxruntime as ort
import numpy as np
import torch
from typing import Generator, Tuple, Dict, Optional, Union
from safetensors import safe_open
from safetensors.torch import load_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 프로젝트 루트 및 GPT_SoVITS 모듈 경로 등록
now_dir = os.path.dirname(os.path.abspath(__file__))
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
gpt_sovits_dir = os.path.join(now_dir, "GPT_SoVITS")
if gpt_sovits_dir not in sys.path:
    sys.path.insert(0, gpt_sovits_dir)

from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from AR.models.t2s_model import Text2SemanticDecoder


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


class PyTorchT2SAdapter:
    """GPT(Text2Semantic) PyTorch CPU Safetensors 고속 추론 어댑터"""

    def __init__(self, safetensors_path: str, device: str = "cpu"):
        print(f"⚡ PyTorch CPU 기반 Safetensors GPT(T2S) 모델 로드 중: {os.path.basename(safetensors_path)}")
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            self.config = json.loads(f.metadata()["config"])

        self.device = device
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.top_k = self.config["inference"]["top_k"]

        weights = load_file(safetensors_path, device=device)
        state_dict = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in weights.items()
        }

        self.model = Text2SemanticDecoder(config=self.config, top_k=self.top_k)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    def infer(
        self,
        all_phoneme_ids: np.ndarray,
        prompt_semantic: np.ndarray,
        all_bert_features: np.ndarray,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        early_stop_num: int = -1,
        repetition_penalty: float = 1.35,
    ) -> Tuple[np.ndarray, int]:
        """단일 문장에 대해 GPT 억양/운율 토큰을 생성합니다."""
        if prompt_semantic.ndim == 1:
            prompt_semantic = prompt_semantic[None, :]

        x = torch.from_numpy(all_phoneme_ids).unsqueeze(0).long().to(self.device)
        x_lens = torch.LongTensor([all_phoneme_ids.shape[0]]).to(self.device)
        prompts = torch.from_numpy(prompt_semantic).long().to(self.device)
        bert_feat = torch.from_numpy(all_bert_features).unsqueeze(0).float().to(self.device)

        prefix_len = prompts.shape[1]

        with torch.no_grad():
            y, idx = self.model.infer_panel_naive(
                x,
                x_lens,
                prompts,
                bert_feat,
                top_k=top_k,
                top_p=top_p,
                early_stop_num=early_stop_num if early_stop_num != -1 else (self.hz * self.max_sec),
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

        pred_tokens = y[0, prefix_len:].cpu().numpy().astype(np.int64)
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


def configure_random_seed(seed: int = None) -> int:
    """글로벌 난수 시드를 설정하고 콘솔/로그에 출력합니다."""
    if seed is None or seed == -1 or seed == "":
        seed = random.randrange(1 << 32)
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ort.set_seed(seed)
    print(f"🎲 [시드 설정] 엔진 난수 시드가 설정되었습니다: {seed}")
    return seed


class ZundamonTTSEngine:
    """하이브리드(PyTorch CPU Safetensors T2S + ONNX VITS) 고속 음성 합성 엔진"""

    def __init__(
        self,
        gpt_safetensors_path: str = None,
        vits_onnx_path: str = None,
        ref_cache_path: str = None,
        num_threads: int = None,
        seed: int = None,
    ):
        self.lock = threading.RLock()
        self.num_threads = num_threads or get_container_cpu_threads()
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        torch.set_num_threads(self.num_threads)
        self.onnx_providers = get_onnx_providers(self.num_threads)

        # 모델 기본 경로 세팅
        self.gpt_safetensors_path = gpt_safetensors_path or os.path.join(
            now_dir, "GPT_weights_v2", "zudamon_style_1-e15.safetensors"
        )
        self.vits_onnx_path = vits_onnx_path or os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_vits.onnx")
        self.ref_cache_path = ref_cache_path or os.path.join(now_dir, "reference_audios", "ref_cache.npz")

        # 1. 난수 시드 설정
        self.seed = configure_random_seed(seed)

        # 2. 사전 베이킹 캐시 로드 (NPZ)
        print(f"📦 [캐시 로드] 사전 베이킹 레퍼런스 캐시: {os.path.basename(self.ref_cache_path)}")
        npz_data = np.load(self.ref_cache_path, allow_pickle=True)
        self.ref_cache = {
            "prompt_text": str(npz_data["prompt_text"].item() if npz_data["prompt_text"].ndim == 0 else npz_data["prompt_text"]),
            "prompt_lang": str(npz_data["prompt_lang"].item() if npz_data["prompt_lang"].ndim == 0 else npz_data["prompt_lang"]),
            "phones": npz_data["phones"].tolist(),
            "bert_features": np.asarray(npz_data["bert_features"], dtype=np.float32),
            "norm_text": str(npz_data["norm_text"].item() if npz_data["norm_text"].ndim == 0 else npz_data["norm_text"]),
            "audios": npz_data["audios"].item(),
        }

        # 3. 텍스트 전처리기 (순수 NumPy G2P)
        self.text_preprocessor = TextPreprocessor()

        # 4. 하이브리드 엔진 로드: PyTorch T2S + ONNX VITS
        print(f"📦 [엔진 로드] PyTorch T2S (Safetensors) + ONNX VITS 디코더 로딩 중...")
        self.t2s = PyTorchT2SAdapter(self.gpt_safetensors_path, device="cpu")
        self.vits = ONNXVitsAdapter(self.vits_onnx_path, self.onnx_providers, self.num_threads)

        self.sampling_rate = 32000
        self.hz = 50
        self.max_sec = 54
        self.current_ref_name = None

        # 기본 감정 음원(01_ref_emoNormal026) 활성화
        self.set_ref_audio("01_ref_emoNormal026.wav")

    def set_seed(self, seed: Optional[Union[int, str]] = None, warmup: bool = False) -> int:
        """
        새로운 시드로 난수 생성기를 즉각 재설정합니다 (< 1ms).
        PyTorch 및 ONNX/NumPy 시드가 즉시 반영되어 모델 재로드가 불필요합니다.
        """
        with self.lock:
            self.seed = configure_random_seed(seed)
            if warmup:
                self.warmup("안녕")
            return self.seed

    def warmup(self, warmup_text: str = "안녕"):
        """서버 시작 시 1회 호출하여 모델 그래프 및 메모리 할당을 사전에 완료합니다."""
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

    def generate_sentence_with_latency(
        self,
        text: str,
        lang: str = "ko",
        ref_name: str = None,
        top_k: int = 15,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        단일 문장 청크를 합성하며 각 파이프라인 단계별 소요 시간(ms) 및 RTF를 측정하여 반환합니다.
        Returns:
            (audio, latency_dict)
        """
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32), {
                "text_frontend_ms": 0.0,
                "gpt_latent_ms": 0.0,
                "vits_decode_ms": 0.0,
                "total_elapsed_ms": 0.0,
                "audio_duration_sec": 0.0,
                "rtf": 0.0,
            }

        with self.lock:
            t_start = time.perf_counter()

            if ref_name and ref_name != self.current_ref_name:
                self.set_ref_audio(ref_name)

            # 1. 텍스트 G2P 음소 분석
            t_g2p_start = time.perf_counter()
            phones, bert_features, _ = self.text_preprocessor.segment_and_extract_feature_for_text(clean_text, lang, "v2")
            t_g2p = time.perf_counter() - t_g2p_start

            if not phones:
                return np.zeros(0, dtype=np.float32), {
                    "text_frontend_ms": round(t_g2p * 1000, 2),
                    "gpt_latent_ms": 0.0,
                    "vits_decode_ms": 0.0,
                    "total_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 2),
                    "audio_duration_sec": 0.0,
                    "rtf": 0.0,
                }

            # 2. 레퍼런스 프롬프트 텐서와 결합
            all_phones = np.asarray(self.ref_cache["phones"] + phones, dtype=np.int64)
            all_bert = np.concatenate([self.ref_cache["bert_features"], bert_features], axis=1).astype(np.float32, copy=False)

            # 3. GPT T2S 운율/억양 토큰 생성 (PyTorch Safetensors)
            t_gpt_start = time.perf_counter()
            pred_tokens, idx = self.t2s.infer(
                all_phones,
                self.prompt_semantic,
                all_bert,
                top_k=top_k,
                early_stop_num=self.hz * self.max_sec,
            )
            t_gpt = time.perf_counter() - t_gpt_start

            # 4. VITS 오디오 파형 디코딩 (ONNX)
            t_vits_start = time.perf_counter()
            pred_semantic = pred_tokens[None, None, :]
            text_seq = np.asarray(phones, dtype=np.int64)[None, :]
            audio = self.vits.decode(pred_semantic, text_seq, speed=speed)
            t_vits = time.perf_counter() - t_vits_start

            # 5. 피크 방지 정규화
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val

            t_total = time.perf_counter() - t_start
            audio_duration_sec = len(audio) / self.sampling_rate
            rtf = (t_total / audio_duration_sec) if audio_duration_sec > 0 else 0.0

            latency = {
                "text_frontend_ms": round(t_g2p * 1000, 2),
                "gpt_latent_ms": round(t_gpt * 1000, 2),
                "vits_decode_ms": round(t_vits * 1000, 2),
                "total_elapsed_ms": round(t_total * 1000, 2),
                "audio_duration_sec": round(audio_duration_sec, 3),
                "rtf": round(rtf, 3),
            }

            return audio, latency

    def generate_sentence(
        self,
        text: str,
        lang: str = "ko",
        ref_name: str = None,
        top_k: int = 15,
        speed: float = 1.0,
    ) -> np.ndarray:
        """단일 문장 청크를 입력받아 float32 오디오 배열(shape: (N,))을 즉시 합성합니다."""
        audio, _ = self.generate_sentence_with_latency(
            text=text, lang=lang, ref_name=ref_name, top_k=top_k, speed=speed
        )
        return audio

    def generate_sentence_pcm_with_latency(self, text: str, **kwargs) -> Tuple[bytes, Dict[str, float]]:
        """단일 문장을 입력받아 32kHz 16-bit Mono PCM 바이트 스트림 및 지연시간 메트릭 반환 (gRPC 직결용)"""
        audio, latency = self.generate_sentence_with_latency(text, **kwargs)
        pcm = (audio * 32767).astype(np.int16)
        return pcm.tobytes(), latency

    def generate_sentence_pcm(self, text: str, **kwargs) -> bytes:
        """단일 문장을 입력받아 32kHz 16-bit Mono PCM 바이트 스트림 반환 (gRPC 직결용)"""
        pcm, _ = self.generate_sentence_pcm_with_latency(text, **kwargs)
        return pcm

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
    print("🚀 [하이브리드 엔진 검증] PyTorch CPU Safetensors T2S + ONNX VITS 구동 시작...")

    engine = ZundamonTTSEngine()

    test_text = (
        "나는 즌다몬이다! 앞으로 잘 지내보자는 거시다!"
    )

    print("🧠 실시간 음성 합성 진행 중 (generate_all)...")
    t0 = time.perf_counter()
    output_audio = engine.generate_all(test_text, lang="ko")
    t_total = time.perf_counter() - t0

    if len(output_audio) > 0:
        output_path = "zundamon_hybrid_final.wav"
        sf.write(output_path, output_audio, engine.sampling_rate)
        duration = len(output_audio) / engine.sampling_rate
        rtf = t_total / duration
        print(f"🎉 합성 대성공! '{output_path}'가 완벽히 생성되었습니다.")
        print(f"⏱️ 합성 시간: {t_total:.2f}초 | 오디오 길이: {duration:.2f}초 | RTF: {rtf:.3f}")
    else:
        print("⚠️ 오디오 데이터가 생성되지 않았습니다.")


if __name__ == "__main__":
    main()
