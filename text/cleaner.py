# Forwarding cleaner module for backward compatibility
from TTS_infer_pack.TextPreprocessor import clean_text_direct as clean_text

if __name__ == '__main__':
    phones, word2ph, norm = clean_text("안녕하세요, 반갑습니다!", "ko")
    print(phones, norm)
