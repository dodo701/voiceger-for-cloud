import os
import sys
import torch
import soundfile as sf
import onnxruntime as ort
import numpy as np
import torchaudio

# GPT-SoVITS 모듈 내부 경로 강제 주입
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

from TTS_infer_pack.TTS import TTS, TTS_Config
from AR.modules.activation import MhaPatched
from tools.my_utils import load_audio

def main():
    print("🚀 하이브리드(PyTorch + ONNX + Safetensors) 파이프라인 초기화 중...")

    # 1. 설정 로드 및 CPU 모드 강제 적용
    config_path = os.path.join(now_dir, "GPT_SoVITS", "configs", "tts_infer.yaml")
    config = TTS_Config(config_path)
    config.device = "cpu"
    config.is_half = False 

    # GPT 가중치(파이토치)와 더미 VITS 가중치 경로 주입
    gpt_path = os.path.join(now_dir, "GPT_weights_v2", "zudamon_style_1-e15.ckpt")
    sovits_path = os.path.join(now_dir, "SoVITS_weights_v2", "zudamon_style_1_e8_s96.pth")
    config.t2s_weights_path = gpt_path
    config.vits_weights_path = sovits_path

    torch.set_num_threads(3)

    # 2. 파이프라인 객체 생성 (이 과정에서 Safetensors 포맷의 BERT/HuBERT가 Zero-copy로 초고속 로드됨)
    print(f"📦 프론트엔드 및 GPT 모델 로딩 중...")
    tts_pipeline = TTS(config)
    
    # 3. 🔥 ONNX VITS 엔진 장착 (Monkey Patching)
    print("⚡ C++ 백엔드 기반 ONNX VITS 어댑터 장착 중...")
    onnx_vits_path = os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_vits.onnx")
    ort_session = ort.InferenceSession(onnx_vits_path, providers=["CPUExecutionProvider"])

    ref_audio_path = os.path.join(now_dir, "reference_audios", "01_ref_emoNormal026.wav")
    ref_audio_tensor = torch.tensor([load_audio(ref_audio_path, 48000)]).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio_tensor, 48000, config.sampling_rate).numpy()

    # 기존 파이토치 VITS를 대체할 커스텀 어댑터 클래스
    class ONNXVitsAdapter:
        def __call__(self, pred_semantic, text_seq, refer):
            onnx_inputs = {
                "text_seq": text_seq.detach().cpu().numpy(),
                "pred_semantic": pred_semantic.detach().cpu().numpy(),
                "ref_audio": ref_audio_sr
            }
            audio_onnx = ort_session.run(["audio"], onnx_inputs)[0]
            # 파이토치 텐서 규격(1, 1, Length)으로 위장하여 반환
            return torch.from_numpy(audio_onnx).unsqueeze(0).unsqueeze(0).to(text_seq.device)
            
        def decode(self, pred_semantic, text_seq, refer):
            return self.__call__(pred_semantic, text_seq, refer)

    # 핵심 렌더링 엔진 갈아끼우기
    tts_pipeline.vq_model = ONNXVitsAdapter()

    # 4. 추론 요청 데이터 세팅
    request_data = {
        "text": "야호! 나는 즌다몬인 거시다! 오늘은 내 한국어 목소리를 테스트해서, 여러 모델을 비교해 본다고 들은 거시다. 후훗, 어떤 내 목소리가 제일 귀엽고, 즌다모찌에 대한 사랑을 잘 표현하는지 똑똑히 들어보란 거시다! ……어라? 만약 이상한 목소리로 들리면 어떡하냐고? 그, 그건 내 잘못이 아니라 설정이나 모델 탓인 거시다! 절대 오해하지 말아 달란 거시다!",
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
    sample_rate = 32000

    try:
        with MhaPatched():  
            for sr, audio in tts_pipeline.run(request_data):
                sample_rate = sr
                if output_audio is None:
                    output_audio = audio
                else:
                    output_audio = np.concatenate([output_audio, audio])
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