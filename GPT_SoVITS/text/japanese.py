# Forwarding module for backward compatibility
from TTS_infer_pack.TextPreprocessor import (
    japanese_g2p as g2p,
    preprocess_jap,
    pyopenjtalk_g2p_prosody,
    replace_consecutive_punctuation as text_normalize,
)

if __name__ == "__main__":
    phones = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
