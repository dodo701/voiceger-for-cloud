# Forwarding module for backward compatibility
from TTS_infer_pack.TextPreprocessor import (
    splits,
    punctuation,
    METHODS,
    get_method,
    get_method_names,
    register_method,
    split_big_text,
    split,
    cut0,
    cut1,
    cut2,
    cut3,
    cut4,
    cut5,
)

if __name__ == '__main__':
    method = get_method("cut5")
    print(method("테스트 문장입니다. 잘 작동하나요?"))
