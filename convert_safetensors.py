from transformers import AutoModel

# 1. BERT 변환
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
bert_model = AutoModel.from_pretrained(bert_path)
# safe_serialization=True 옵션이 safetensors 포맷으로 저장하게 만듭니다.
bert_model.save_pretrained(bert_path, safe_serialization=True)
print("✅ BERT safetensors 변환 완료!")

# 2. HuBERT 변환
hubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
hubert_model = AutoModel.from_pretrained(hubert_path)
hubert_model.save_pretrained(hubert_path, safe_serialization=True)
print("✅ HuBERT safetensors 변환 완료!")