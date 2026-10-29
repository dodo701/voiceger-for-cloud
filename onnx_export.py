import os
import sys
import shutil
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 🔥 안쪽 폴더의 모듈들을 인식할 수 있도록 경로 강제 주입
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
import torch
import torchaudio
from torch import nn
from feature_extractor import cnhubert

cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model()
from text import cleaned_text_to_sequence
import soundfile
from tools.my_utils import load_audio


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(
        dtype=y.dtype, device=y.device
    )
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    # OpenVINO 호환: .squeeze(1) 대신 슬라이싱을 사용하여 동적 If 분기 노드 생성을 방지
    y = y[:, 0, :]
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.vits = vits

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)
        prompt = prompt_semantic.unsqueeze(0)
        return self.encoder(all_phoneme_ids, bert), prompt


class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        if t2s_path.endswith(".safetensors"):
            from safetensors import safe_open
            from safetensors.torch import load_file
            with safe_open(t2s_path, framework="pt", device="cpu") as f:
                self.config = json.loads(f.metadata()["config"])
            weights = load_file(t2s_path, device="cpu")
        else:
            dict_s1 = torch.load(t2s_path, map_location="cpu", weights_only=False)
            self.config = dict_s1["config"]
            weights = dict_s1["weight"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
        self.t2s_model.load_state_dict(weights)
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model.model.top_k = torch.LongTensor([self.config["inference"]["top_k"]])
        self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.onnx_encoder = T2SEncoder(self.t2s_model, self.vits_model)
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.t2s_model.early_stop_num

        # [1,N] [1,N] [N, 1024] [N, 1024] [1, 768, N]
        x, prompts = self.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

        prefix_len = prompts.shape[1]

        # [1,N,512] [1,N]
        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        stop = False
        for idx in range(1, 1500):
            # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            enco = self.stage_decoder(y, k, v, y_emb, x_example)
            y, k, v, y_emb, logits, samples = enco
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS or samples[0, 0] == self.t2s_model.EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, dynamo=False):
        import onnx
        os.makedirs("GPT_weights_v2", exist_ok=True)
        enc_path = f"GPT_weights_v2/{project_name}_t2s_encoder.onnx"
        fsdec_path = f"GPT_weights_v2/{project_name}_t2s_fsdec.onnx"
        sdec_path = f"GPT_weights_v2/{project_name}_t2s_sdec.onnx"

        if dynamo:
            export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            onnx_encoder_export_output = torch.onnx.dynamo_export(
                self.onnx_encoder,
                (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
                export_options=export_options
            )
            onnx_encoder_export_output.save(enc_path)
            return

        print(f"🚀 {project_name} T2S Encoder ONNX 추출 시작...")
        torch.onnx.export(
            self.onnx_encoder,
            (ref_seq, text_seq, ref_bert, text_bert, ssl_content),
            enc_path,
            input_names=["ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content"],
            output_names=["x", "prompts"],
            dynamic_axes={
                "ref_seq": {1: "ref_length"},
                "text_seq": {1: "text_length"},
                "ref_bert": {0: "ref_length"},
                "text_bert": {0: "text_length"},
                "ssl_content": {2: "ssl_length"},
            },
            opset_version=17,
            verbose=False,
            dynamo=False,
        )
        enc_model = onnx.shape_inference.infer_shapes(onnx.load(enc_path), data_prop=True)
        onnx.save(enc_model, enc_path)
        print(f"🎉 T2S Encoder 추출 완료 (OpenVINO 호환 적용): {enc_path}")

        x, prompts = self.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

        print(f"🚀 {project_name} T2S First Stage Decoder ONNX 추출 시작...")
        torch.onnx.export(
            self.first_stage_decoder,
            (x, prompts),
            fsdec_path,
            input_names=["x", "prompts"],
            output_names=["y", "k", "v", "y_emb", "x_example"],
            dynamic_axes={
                "x": {1: "x_length"},
                "prompts": {1: "prompts_length"},
            },
            verbose=False,
            opset_version=17,
            dynamo=False,
        )
        fsdec_model = onnx.shape_inference.infer_shapes(onnx.load(fsdec_path), data_prop=True)
        onnx.save(fsdec_model, fsdec_path)
        print(f"🎉 T2S First Stage Decoder 추출 완료 (OpenVINO 호환 적용): {fsdec_path}")

        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        print(f"🚀 {project_name} T2S Stage Decoder ONNX 추출 시작...")
        torch.onnx.export(
            self.stage_decoder,
            (y, k, v, y_emb, x_example),
            sdec_path,
            input_names=["iy", "ik", "iv", "iy_emb", "ix_example"],
            output_names=["y", "k", "v", "y_emb", "logits", "samples"],
            dynamic_axes={
                "iy": {1: "iy_length"},
                "ik": {1: "ik_length"},
                "iv": {1: "iv_length"},
                "iy_emb": {1: "iy_emb_length"},
                "ix_example": {1: "ix_example_length"},
            },
            verbose=False,
            opset_version=17,
            dynamo=False,
        )
        sdec_model = onnx.shape_inference.infer_shapes(onnx.load(sdec_path), data_prop=True)
        onnx.save(sdec_model, sdec_path)
        print(f"🎉 T2S Stage Decoder 추출 완료 (OpenVINO 호환 적용): {sdec_path}")


class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu", weights_only=False)
        self.hps = dict_s2["config"]
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = "v2"

        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

    def forward(self, text_seq, pred_semantic, ref_audio):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return self.vq_model(pred_semantic, text_seq, refer)[0, 0]


class GptSoVits(nn.Module):
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content, debug=False, project_name="zundamon"):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        audio = self.vits(text_seq, pred_semantic, ref_audio)
        if debug:
            import onnxruntime
            sess = onnxruntime.InferenceSession(f"SoVITS_weights_v2/{project_name}_vits.onnx", providers=["CPUExecutionProvider"])
            audio1 = sess.run(None, {
                "text_seq": text_seq.detach().cpu().numpy(),
                "pred_semantic": pred_semantic.detach().cpu().numpy(),
                "ref_audio": ref_audio.detach().cpu().numpy()
            })
            return audio, audio1
        return audio

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content, project_name):
        import onnx
        from onnx import helper, TensorProto

        # 1. GPT(T2S) 3단계 모델 ONNX 추출 (OpenVINO 호환 패치 포함)
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name)

        # 2. VITS 디코더 ONNX 추출 (OpenVINO 호환 패치 포함)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        os.makedirs("SoVITS_weights_v2", exist_ok=True)
        vits_out_path = f"SoVITS_weights_v2/{project_name}_vits.onnx"

        print(f"🚀 {project_name} VITS 디코더 ONNX 추출 시작...")
        torch.onnx.export(
            self.vits,
            (text_seq, pred_semantic, ref_audio),
            vits_out_path,
            input_names=["text_seq", "pred_semantic", "ref_audio"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
                "ref_audio": {1: "audio_length"},
            },
            opset_version=17,
            verbose=False,
            dynamo=False,
        )
        onnx_model = onnx.load(vits_out_path)
        freq_bins = self.vits.hps.data.filter_length // 2 + 1
        onnx_model.graph.value_info.append(
            helper.make_tensor_value_info("/STFT_output_0", TensorProto.FLOAT, [1, "frames", freq_bins, 2])
        )
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
        onnx.save(onnx_model, vits_out_path)
        print(f"🎉 VITS 디코더 추출 완료 (OpenVINO 호환 적용): {vits_out_path}")


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl = ssl_model

    def forward(self, ref_audio_16k):
        return self.ssl.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)


def export(vits_path, gpt_path, project_name, vits_model="v2", debug=False):
    # 0. VITS Latent 추출기 먼저 추출
    export_latent_only(vits_path, project_name)

    vits = VitsModel(vits_path)
    gpt = T2SModel(gpt_path, vits)
    gpt_sovits = GptSoVits(vits, gpt)
    ssl = SSLModel()
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(["n", "i2", "h", "ao3", ",", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"], version=vits_model)])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"], version=vits_model)])
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, vits.hps.data.sampling_rate).float()

    ssl_content = ssl(ref_audio_16k).float()

    # GPT(T2S) 3종 + VITS ONNX 전체 추출 실행
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content, project_name)

    if debug:
        a, b = gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content, debug=debug, project_name=project_name)
        soundfile.write("out1.wav", a.cpu().detach().numpy(), vits.hps.data.sampling_rate)
        soundfile.write("out2.wav", b[0], vits.hps.data.sampling_rate)

    MoeVSConf = {
        "Folder": f"{project_name}",
        "Name": f"{project_name}",
        "Type": "GPT-SoVits",
        "Rate": vits.hps.data.sampling_rate,
        "NumLayers": gpt.t2s_model.num_layers,
        "EmbeddingDim": gpt.t2s_model.embedding_dim,
        "Dict": "BasicDict",
        "BertPath": "chinese-roberta-wwm-ext-large",
        "AddBlank": False,
    }

    os.makedirs("GPT_weights_v2", exist_ok=True)
    with open(f"GPT_weights_v2/{project_name}.json", 'w', encoding="utf-8") as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)
    print(f"✅ 전체 ONNX 모델(GPT 3종 + VITS 2종) 추출 및 OpenVINO 호환 패치가 완료되었습니다!")


def export_vits_only(vits_path, gpt_path, project_name, vits_model="v2"):
    vits = VitsModel(vits_path)
    gpt = T2SModel(gpt_path, vits)

    # 1. 추출을 위한 더미 텐서 생성 (v2 버전 규격에 맞춤)
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(["n", "i2", "h", "ao3"], version=vits_model)])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["w", "o3", "sh", "i4"], version=vits_model)])
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()

    # 오디오 더미 텐서 및 샘플링 레이트 변환
    ref_audio = torch.randn((1, 48000 * 5)).float()
    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, vits.hps.data.sampling_rate).float()

    ssl = SSLModel()
    ssl_content = ssl(ref_audio_16k).float()

    # 2. GPT(T2S) 모델을 1회 통과시켜 VITS 입력용 pred_semantic(의미론적 토큰) 더미 생성
    pred_semantic = gpt(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

    os.makedirs("SoVITS_weights_v2", exist_ok=True)

    print(f"🚀 {project_name} VITS 디코더 ONNX 추출 시작...")

    onnx_out_path = f"SoVITS_weights_v2/{project_name}_vits.onnx"

    # 3. VITS 모델 단독 ONNX 추출
    torch.onnx.export(
        vits,
        (text_seq, pred_semantic, ref_audio_sr),
        onnx_out_path,
        input_names=["text_seq", "pred_semantic", "ref_audio"],
        output_names=["audio"],
        dynamic_axes={
            "text_seq": {1: "text_length"},
            "pred_semantic": {2: "pred_length"},
            "ref_audio": {1: "audio_length"},
        },
        opset_version=17,
        verbose=False,
        dynamo=False,
    )

    # 4. OpenVINOExecutionProvider 호환을 위한 정적 Rank 및 Shape 추론 정보 주입
    import onnx
    from onnx import helper, TensorProto
    onnx_model = onnx.load(onnx_out_path)
    freq_bins = vits.hps.data.filter_length // 2 + 1
    onnx_model.graph.value_info.append(
        helper.make_tensor_value_info("/STFT_output_0", TensorProto.FLOAT, [1, "frames", freq_bins, 2])
    )
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
    onnx.save(onnx_model, onnx_out_path)
    print(f"🎉 VITS 단독 추출 완료 (OpenVINO 호환 적용): {onnx_out_path}")


# latent 가중치 추출 코드.
def export_latent_only(vits_path, project_name):
    # 1. 원본 VITS 모델 로드
    vits = VitsModel(vits_path)

    # 2. HuBERT 출력 규격에 맞춘 더미 텐서 (배치, 차원, 시퀀스 길이)
    # ssl_dim은 v2 모델 기준 768 고정
    dummy_ssl_content = torch.randn(1, 768, 100).float()

    # 3. ONNX Export를 위해 extract_latent만 감싸는 래퍼(Wrapper) 모듈 생성
    class LatentExtractor(nn.Module):
        def __init__(self, vq_model):
            super().__init__()
            self.vq_model = vq_model

        def forward(self, x):
            # vq_model(SynthesizerTrn)의 extract_latent 강제 호출
            return self.vq_model.extract_latent(x)

    extractor = LatentExtractor(vits.vq_model)
    extractor.eval()

    os.makedirs("SoVITS_weights_v2", exist_ok=True)

    print(f"🚀 {project_name} VITS Latent Extractor ONNX 추출 시작...")
    latent_out_path = f"SoVITS_weights_v2/{project_name}_latent.onnx"

    # 4. ONNX로 굽기
    torch.onnx.export(
        extractor,
        (dummy_ssl_content,),
        latent_out_path,
        input_names=["ssl_content"],
        output_names=["codes"],
        dynamic_axes={
            "ssl_content": {2: "ssl_length"},
            "codes": {1: "codes_length"}
        },
        opset_version=17,
        verbose=False,
        dynamo=False,
    )
    import onnx
    latent_model = onnx.shape_inference.infer_shapes(onnx.load(latent_out_path), data_prop=True)
    onnx.save(latent_model, latent_out_path)
    print(f"🎉 Latent 추출기 단독 완료: {latent_out_path}")


# --- 실행부 ---
if __name__ == "__main__":
    gpt_path = "GPT_weights_v2/zudamon_style_1-e15.ckpt"
    vits_path = "SoVITS_weights_v2/zudamon_style_1_e8_s96.pth"
    exp_path = "zundamon"

    export(vits_path, gpt_path, exp_path, vits_model="v2")
