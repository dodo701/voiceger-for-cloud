import os
import sys
import glob
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

import numpy as np
import torch
import librosa
import onnxruntime as ort
from feature_extractor.cnhubert import CNHubert
from TTS_infer_pack.TextPreprocessor import TextPreprocessor, splits
from tools.my_utils import load_audio

DEFAULT_PROMPT_TEXT = "私はいつもミネラルウォーターを持ち歩いています。"
DEFAULT_PROMPT_LANG = "ja"
SAMPLING_RATE = 32000


def build_reference_cache(
    ref_dir: str = os.path.join(now_dir, "reference_audios"),
    cnhubert_path: str = os.path.join(now_dir, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base"),
    latent_onnx_path: str = os.path.join(now_dir, "SoVITS_weights_v2", "zundamon_latent.onnx"),
    output_cache_path: str = os.path.join(now_dir, "reference_audios", "ref_cache.npz"),
    prompt_text: str = DEFAULT_PROMPT_TEXT,
    prompt_lang: str = DEFAULT_PROMPT_LANG,
):
    print("=" * 65)
    print("🚀 레퍼런스 오디오(8종) 및 고정 프롬프트 텍스트 사전 캐싱(Preload -> NPZ) 시작")
    print("=" * 65)

    t0 = time.time()
    device = torch.device("cpu")

    # 1. 고정 프롬프트 텍스트 전처리 (음소 ID 및 Zero-BERT 배열 생성)
    print(f"📝 공통 레퍼런스 대사 전처리 중: '{prompt_text}' ({prompt_lang})")
    preprocessor = TextPreprocessor()
    clean_prompt = prompt_text.strip("\n")
    if clean_prompt[-1] not in splits:
        clean_prompt += "。" if prompt_lang != "en" else "."

    phones, bert_features, norm_text = preprocessor.segment_and_extract_feature_for_text(
        clean_prompt, prompt_lang, "v2"
    )
    bert_features_np = np.asarray(bert_features, dtype=np.float32)
    print(f"✅ 텍스트 전처리 완료: 음소 {len(phones)}개, BERT 배열 {tuple(bert_features_np.shape)}")

    # 2. CNHuBERT 및 Latent ONNX 세션 로드 (캐시 빌드 시에만 1회 사용)
    print(f"📦 CNHuBERT 로드 중: {cnhubert_path}")
    cnhubert_model = CNHubert(cnhubert_path).eval().to(device)

    print(f"📦 Latent ONNX 로드 중: {latent_onnx_path}")
    latent_session = ort.InferenceSession(latent_onnx_path, providers=["CPUExecutionProvider"])

    # 3. reference_audios 내 모든 .wav 파일 순회 추출
    wav_files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"레퍼런스 오디오(.wav)를 찾을 수 없습니다: {ref_dir}")

    zero_wav = np.zeros(int(SAMPLING_RATE * 0.3), dtype=np.float32)
    zero_wav_torch = torch.from_numpy(zero_wav).to(device)

    audios_cache = {}
    with torch.no_grad():
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            stem = os.path.splitext(fname)[0]
            print(f"🎧 오디오 특징 추출 중: {fname} ...", end=" ")

            # (A) 16kHz 로드 -> HuBERT -> Latent ONNX -> prompt_semantic (np.int64) 추출
            wav16k, _ = librosa.load(wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError(f"참고 오디오({fname})가 3~10초 범위를 벗어났습니다!")

            wav16k_torch = torch.from_numpy(wav16k).to(device)
            wav16k_cat = torch.cat([wav16k_torch, zero_wav_torch])
            hubert_feature = cnhubert_model.model(wav16k_cat.unsqueeze(0))["last_hidden_state"].transpose(1, 2)

            ssl_content = hubert_feature.detach().cpu().float().numpy()
            codes = latent_session.run(["codes"], {"ssl_content": ssl_content})[0]
            prompt_semantic_np = np.asarray(codes[0, 0], dtype=np.int64)

            # (B) 32kHz 리샘플링 파형 (VITS ONNX 입력용)
            ref_audio_sr = np.expand_dims(load_audio(wav_path, SAMPLING_RATE), axis=0).astype(np.float32)

            entry = {
                "prompt_semantic": prompt_semantic_np,
                "ref_audio_sr": ref_audio_sr,
            }
            # 파일명("01_ref_emoNormal026.wav")과 확장자 제외 이름("01_ref_emoNormal026") 모두 조회 가능하도록 등록
            audios_cache[fname] = entry
            audios_cache[stem] = entry
            print(f"완료! (semantic: {tuple(prompt_semantic_np.shape)}, wave: {ref_audio_sr.shape})")

    np.savez(
        output_cache_path,
        prompt_text=clean_prompt,
        prompt_lang=prompt_lang,
        phones=np.asarray(phones, dtype=np.int64),
        bert_features=bert_features_np,
        norm_text=norm_text,
        audios=audios_cache,
    )
    cache_size_mb = os.path.getsize(output_cache_path) / (1024 * 1024)
    elapsed = time.time() - t0

    print("=" * 65)
    print(f"🎉 사전 캐시(NPZ) 빌드 완료! (소요 시간: {elapsed:.2f}초)")
    print(f"💾 저장된 캐시 경로: {output_cache_path} ({cache_size_mb:.2f} MB)")
    print(f"🔑 캐시된 오디오 개수: {len(wav_files)}종")
    print("=" * 65)


if __name__ == "__main__":
    build_reference_cache()
