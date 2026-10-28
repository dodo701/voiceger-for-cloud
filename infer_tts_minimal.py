import os
import sys
import platform
import torch
import soundfile as sf
import onnxruntime as ort
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# GPT-SoVITS 모듈 내부 경로 강제 주입
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

from TTS_infer_pack.TTS import TTS, TTS_Config
from AR.modules.activation import MhaPatched
from tools.my_utils import load_audio


def is_intel_cpu() -> bool:
    """현재 시스템의 CPU가 Intel 프로세서인지 감지합니다."""
    if "intel" in platform.processor().lower():
        return True
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "intel" in f.read().lower()
    return False


def get_onnx_providers() -> list:
    """Intel CPU 감지 시 OpenVINOExecutionProvider를 우선 활성화합니다."""
    if is_intel_cpu() and "OpenVINOExecutionProvider" in ort.get_available_providers():
        if sys.platform == "win32":
            try:
                import onnxruntime.tools.add_openvino_win_libs as ov_utils
                ov_utils.add_openvino_libs_to_path()
            except (Exception, SystemExit):
                pass
        print("🚀 [Intel CPU 감지] OpenVINOExecutionProvider 가속을 활성화합니다!")
        return [
            ("OpenVINOExecutionProvider", {"device_type": "CPU"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


class ONNXVitsAdapter:
    def __init__(self, vits_onnx_path: str, latent_onnx_path: str, ref_audio_sr: np.ndarray):
        print("⚡ C++ 백엔드 기반 Full ONNX VITS 어댑터(Latent + Decoder) 세팅 중...")
        providers = get_onnx_providers()
        self.vits_session = ort.InferenceSession(vits_onnx_path, providers=providers)
        self.latent_session = ort.InferenceSession(latent_onnx_path, providers=providers)
        print(f"✅ 활성화된 ONNX 프로바이더: {self.vits_session.get_providers()}")

        self.upsample_rates = [10, 8, 2, 2, 2]
        self.ref_audio_sr = ref_audio_sr

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        ssl_content = x.detach().cpu().float().numpy()
        codes = self.latent_session.run(["codes"], {"ssl_content": ssl_content})[0]
        return torch.from_numpy(codes).to(device=x.device, dtype=torch.long)

    def decode(self, pred_semantic: torch.Tensor, text_seq: torch.Tensor, refer_audio_spec, speed: float = 1.0) -> torch.Tensor:
        onnx_inputs = {
            "text_seq": text_seq.detach().cpu().numpy(),
            "pred_semantic": pred_semantic.detach().cpu().numpy(),
            "ref_audio": self.ref_audio_sr,
        }
        audio_onnx = self.vits_session.run(["audio"], onnx_inputs)[0]
        return torch.from_numpy(audio_onnx).unsqueeze(0).unsqueeze(0).to(text_seq.device)


class HybridTTS(TTS):
    def __init__(self, configs: TTS_Config, vits_onnx_path: str, latent_onnx_path: str, ref_audio_sr: np.ndarray):
        self.vits_onnx_path = vits_onnx_path
        self.latent_onnx_path = latent_onnx_path
        self.ref_audio_sr = ref_audio_sr
        super().__init__(configs)

    def init_vits_weights(self, weights_path: str):
        print("🔪 PyTorch VITS(.pth) 로드를 건너뛰고 Full ONNX(Latent + VITS) 래퍼를 주입합니다!")
        self.configs.update_version("v2")
        self.vits_model = ONNXVitsAdapter(self.vits_onnx_path, self.latent_onnx_path, self.ref_audio_sr)

    def _set_ref_spec(self, ref_audio_path: str):
        # ONNX VITS는 ref_audio_sr 배열을 직접 사용하므로 불필요한 PyTorch STFT 연산 생략
        self.prompt_cache["refer_spec"] = [torch.empty(0)]


def main():
    print("🚀 하이브리드(PyTorch GPT + Full ONNX VITS + Safetensors) 파이프라인 초기화 중...")

    config_path = os.path.join(now_dir, "GPT_SoVITS", "configs", "tts_infer.yaml")
    config = TTS_Config(config_path)
    config.device = "cpu"
    config.is_half = False

    gpt_path = os.path.join(now_dir, "GPT_weights_v2", "zudamon_style_1-e15.safetensors")
    onnx_vits_path = os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_vits.onnx")
    onnx_latent_path = os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_latent.onnx")

    config.t2s_weights_path = gpt_path
    config.vits_weights_path = onnx_vits_path

    torch.set_num_threads(4)

    ref_audio_path = os.path.join(now_dir, "reference_audios", "01_ref_emoNormal026.wav")
    ref_audio_sr = np.expand_dims(load_audio(ref_audio_path, config.sampling_rate), axis=0)

    print("📦 프론트엔드 및 Full ONNX VITS 하이브리드 엔진 로딩 중...")
    tts_pipeline = HybridTTS(config, onnx_vits_path, onnx_latent_path, ref_audio_sr)

    request_data = {
        "text": "야호! 나는 즌다몬인 거시다! 오늘은 내 한국어 목소리를 테스트해서, 여러 모델을 비교해 본다고 들은 거시다. 후훗, 어떤 내 목소리가 제일 귀엽고, 즌다모찌에 대한 사랑을 잘 표현하는지 똑똑히 들어보란 거시다!",
        "text_lang": "ko",
        "ref_audio_path": ref_audio_path,
        "prompt_text": "私はいつもミネラルウォーターを持ち歩いています。",
        "prompt_lang": "ja",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut2",
    }

    print("🧠 실시간 음성 합성 진행 중...")

    output_audio = None
    sample_rate = config.sampling_rate

    try:
        with MhaPatched():
            for sr, audio in tts_pipeline.run(request_data):
                sample_rate = sr
                output_audio = audio if output_audio is None else np.concatenate([output_audio, audio])
    except Exception as e:
        print(f"🚨 추론 중 에러 발생: {e}")
        return

    if output_audio is not None:
        output_path = "zundamon_hybrid_final.wav"
        sf.write(output_path, output_audio, sample_rate)
        print(f"🎉 합성 대성공! 콜드 스타트와 렌더링 병목이 모두 해결된 '{output_path}'가 생성되었습니다.")
    else:
        print("⚠️ 오디오 데이터가 생성되지 않았습니다.")


if __name__ == "__main__":
    main()
