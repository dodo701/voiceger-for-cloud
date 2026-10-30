from copy import deepcopy
import math
import os, sys, gc
import random
import traceback

from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)
from typing import Any, List, Tuple, Union
import numpy as np
import yaml

from time import time as ttime
from TTS_infer_pack.text_segmentation_method import splits
from TTS_infer_pack.TextPreprocessor import TextPreprocessor


def set_seed(seed: int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    print(f"Set seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


class TTS_Config:
    default_configs = {
        "default": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "default_v2": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
    }
    configs: dict = None
    v2_languages: list = ["auto", "en", "ja", "ko", "all_ja", "all_ko"]
    languages: list = v2_languages

    def __init__(self, configs: Union[dict, str] = None):
        configs_base_path: str = "GPT_SoVITS/configs/"
        os.makedirs(configs_base_path, exist_ok=True)
        self.configs_path: str = os.path.join(configs_base_path, "tts_infer.yaml")

        if configs in ["", None]:
            if not os.path.exists(self.configs_path):
                self.save_configs()
                print(f"Create default config file at {self.configs_path}")
            configs: dict = deepcopy(self.default_configs)

        if isinstance(configs, str):
            self.configs_path = configs
            configs: dict = self._load_configs(self.configs_path)

        assert isinstance(configs, dict)
        self.default_configs["default"] = configs.get("default", self.default_configs["default"])
        self.default_configs["default_v2"] = configs.get("default_v2", self.default_configs["default_v2"])

        default_config_key = "default_v2" if "default_v2" in configs else "default"
        self.configs: dict = configs.get("custom", deepcopy(self.default_configs[default_config_key]))

        self.device = str(self.configs.get("device", "cpu"))
        self.is_half = False
        self.version = "v2"
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.languages = self.v2_languages

        if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
            self.t2s_weights_path = self.default_configs[default_config_key]['t2s_weights_path']
        if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
            self.vits_weights_path = self.default_configs[default_config_key]['vits_weights_path']
        if (self.cnhuhbert_base_path in [None, ""]) or (not os.path.exists(self.cnhuhbert_base_path)):
            self.cnhuhbert_base_path = self.default_configs[default_config_key]['cnhuhbert_base_path']
        self.update_configs()

        self.max_sec = None
        self.hz: int = 50
        self.semantic_frame_rate: str = "25hz"
        self.segment_size: int = 20480
        self.filter_length: int = 2048
        self.sampling_rate: int = 32000
        self.hop_length: int = 640
        self.win_length: int = 2048
        self.n_speakers: int = 300

    def _load_configs(self, configs_path: str) -> dict:
        if not os.path.exists(configs_path):
            print("설정 파일 경로가 존재하지 않아 기본 설정을 사용합니다.")
            self.save_configs(configs_path)
        with open(configs_path, 'r', encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        return configs

    def save_configs(self, configs_path: str = None) -> None:
        configs = deepcopy(self.default_configs)
        if self.configs is not None:
            configs["custom"] = self.update_configs()

        if configs_path is None:
            configs_path = self.configs_path
        with open(configs_path, 'w', encoding='utf-8') as f:
            yaml.dump(configs, f)

    def update_configs(self):
        self.config = {
            "device": str(self.device),
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": self.t2s_weights_path,
            "vits_weights_path": self.vits_weights_path,
            "bert_base_path": self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def update_version(self, version: str) -> None:
        self.version = "v2"
        self.languages = self.v2_languages

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, '-') + '\n'
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + '\n'
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.configs_path)

    def __eq__(self, other):
        return isinstance(other, TTS_Config) and self.configs_path == other.configs_path


class TTS:
    def __init__(self, configs: Union[dict, str, TTS_Config]):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs: TTS_Config = TTS_Config(configs)

        self.t2s_model: Any = None
        self.vits_model: Any = None
        self.bert_tokenizer: Any = None
        self.bert_model: Any = None
        self.cnhuhbert_model: Any = None

        self._init_models()

        self.text_preprocessor: TextPreprocessor = TextPreprocessor(
            self.bert_model,
            self.bert_tokenizer,
            self.configs.device,
        )

        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }

        self.stop_flag: bool = False
        self.precision = np.float32

    def _init_models(self):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

    def init_cnhuhbert_weights(self, base_path: str):
        self.cnhuhbert_model = None

    def init_bert_weights(self, base_path: str):
        self.bert_tokenizer = None
        self.bert_model = None

    def init_vits_weights(self, weights_path: str):
        self.vits_model = None

    def init_t2s_weights(self, weights_path: str):
        self.t2s_model = None

    def enable_half_precision(self, enable: bool = False, save: bool = False):
        self.configs.is_half = False
        self.precision = np.float32

    def set_device(self, device: str, save: bool = True):
        self.configs.device = str(device)
        if save:
            self.configs.save_configs()

    def set_ref_audio(self, ref_audio_path: str):
        self._set_ref_audio_path(ref_audio_path)

    def _set_ref_audio_path(self, ref_audio_path: str):
        self.prompt_cache["ref_audio_path"] = ref_audio_path

    def batch_sequences(self, sequences: List[np.ndarray], axis: int = 0, pad_value: int = 0, max_length: int = None):
        seq = sequences[0]
        ndim = seq.ndim
        if axis < 0:
            axis += ndim
        seq_lengths = [s.shape[axis] for s in sequences]
        if max_length is None or max_length < max(seq_lengths):
            max_length = max(seq_lengths)

        padded_sequences = []
        for s, length in zip(sequences, seq_lengths):
            pad_width = [(0, 0)] * ndim
            pad_width[axis] = (0, max_length - length)
            padded_seq = np.pad(s, pad_width, mode="constant", constant_values=pad_value)
            padded_sequences.append(padded_seq)
        return np.stack(padded_sequences, axis=0)

    def to_batch(
        self,
        data: list,
        prompt_data: dict = None,
        batch_size: int = 5,
        threshold: float = 0.75,
        split_bucket: bool = True,
        device: str = "cpu",
        precision=np.float32,
    ):
        _data: list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos < index_and_len_list.shape[0]:
                pos_end = min(pos + batch_size, index_and_len_list.shape[0])
                while pos < pos_end:
                    batch = index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score = batch[(pos_end - pos) // 2] / (batch.mean() + 1e-8)
                    if (score >= threshold) or (pos_end - pos == 1):
                        batch_index = index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end = pos_end - 1

            assert batch_index_list_len == len(data)
        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = np.concatenate(
                        [prompt_data["bert_features"], item["bert_features"]], axis=1
                    ).astype(precision, copy=False)
                    all_phones = np.asarray(list(prompt_data["phones"]) + list(item["phones"]), dtype=np.int64)
                    phones = np.asarray(item["phones"], dtype=np.int64)
                else:
                    all_bert_features = np.asarray(item["bert_features"], dtype=precision)
                    phones = np.asarray(item["phones"], dtype=np.int64)
                    all_phones = phones

                all_bert_max_len = max(all_bert_max_len, all_bert_features.shape[-1])
                all_phones_max_len = max(all_phones_max_len, all_phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            max_len = max(all_bert_max_len, all_phones_max_len)

            batch = {
                "phones": phones_list,
                "phones_len": np.asarray(phones_len_list, dtype=np.int64),
                "all_phones": all_phones_list,
                "all_phones_len": np.asarray(all_phones_len_list, dtype=np.int64),
                "all_bert_features": all_bert_features_list,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data: list, batch_index_list: list) -> list:
        length = len(sum(batch_index_list, []))
        _data = [None] * length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def stop(self):
        self.stop_flag = True

    def run(self, inputs: dict):
        ########## variables initialization ###########
        self.stop_flag: bool = False
        text: str = inputs.get("text", "")
        text_lang: str = inputs.get("text_lang", "")
        ref_audio_path: str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths: list = inputs.get("aux_ref_audio_paths", [])
        prompt_text: str = inputs.get("prompt_text", "")
        prompt_lang: str = inputs.get("prompt_lang", "")
        top_k: int = inputs.get("top_k", 5)
        top_p: float = inputs.get("top_p", 1)
        temperature: float = inputs.get("temperature", 1)
        text_split_method: str = inputs.get("text_split_method", "cut0")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)

        if parallel_infer:
            print("병렬 추론 모드가 활성화되었습니다.")
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        else:
            print("병렬 추론 모드가 비활성화되었습니다.")
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        if return_fragment:
            print("구간별 반환 모드가 활성화되었습니다.")
            if split_bucket:
                split_bucket = False
                print("구간별 반환 모드는 버킷 처리를 지원하지 않아 버킷 처리를 비활성화합니다.")

        if split_bucket and speed_factor == 1.0:
            print("버킷 처리 모드가 활성화되었습니다.")
        elif speed_factor != 1.0:
            print("배속 조절은 버킷 처리를 지원하지 않아 버킷 처리를 비활성화합니다.")
            split_bucket = False
        else:
            print("버킷 처리 모드가 비활성화되었습니다.")

        if fragment_interval < 0.01:
            fragment_interval = 0.01
            print("구간 간격이 너무 작아 자동으로 0.01로 설정되었습니다.")

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if ref_audio_path in [None, ""] and (
            (self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])
        ):
            raise ValueError("ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()")

        ###### setting reference audio and prompt text preprocessing ########
        t0 = ttime()
        if (ref_audio_path is not None) and (ref_audio_path != self.prompt_cache["ref_audio_path"]):
            self.set_ref_audio(ref_audio_path)

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            print("실제 입력된 참고 텍스트:", prompt_text)
            if self.prompt_cache["prompt_text"] != prompt_text:
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text,
                    prompt_lang,
                    self.configs.version,
                )
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text

        ###### text preprocessing ########
        t1 = ttime()
        data: list = None
        if not return_fragment:
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate), dtype=np.int16)
                return

            batch_index_list: list = None
            data, batch_index_list = self.to_batch(
                data,
                prompt_data=self.prompt_cache if not no_prompt_text else None,
                batch_size=batch_size,
                threshold=batch_threshold,
                split_bucket=split_bucket,
                device=self.configs.device,
                precision=self.precision,
            )
        else:
            print("############ 텍스트 분할 ############")
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i % batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

            def make_batch(batch_texts):
                batch_data = []
                print("############ 텍스트 특징 추출 ############")
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                        text, text_lang, self.configs.version
                    )
                    if phones is None:
                        continue
                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(
                    batch_data,
                    prompt_data=self.prompt_cache if not no_prompt_text else None,
                    batch_size=batch_size,
                    threshold=batch_threshold,
                    split_bucket=False,
                    device=self.configs.device,
                    precision=self.precision,
                )
                return batch[0]

        t2 = ttime()
        try:
            print("############ 추론 ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            for item in data:
                t3 = ttime()
                if return_fragment:
                    item = make_batch(item)
                    if item is None:
                        continue

                batch_phones: List[np.ndarray] = item["phones"]
                batch_phones_len: np.ndarray = item["phones_len"]
                all_phoneme_ids: List[np.ndarray] = item["all_phones"]
                all_phoneme_lens: np.ndarray = item["all_phones_len"]
                all_bert_features: List[np.ndarray] = item["all_bert_features"]
                norm_text: str = item["norm_text"]
                max_len = item["max_len"]

                print("프론트엔드 처리 후 텍스트(문장별):", norm_text)
                if no_prompt_text:
                    prompt = None
                else:
                    prompt_sem = np.asarray(self.prompt_cache["prompt_semantic"], dtype=np.int64)
                    prompt = np.broadcast_to(prompt_sem, (len(all_phoneme_ids), prompt_sem.shape[-1]))

                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                t4 = ttime()
                t_34 += t4 - t3

                refer_audio_spec = self.prompt_cache["refer_spec"]
                batch_audio_fragment = []

                if speed_factor == 1.0:
                    pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                    upsample_rate = math.prod(self.vits_model.upsample_rates)
                    audio_frag_idx = [
                        pred_semantic_list[i].shape[0] * 2 * upsample_rate
                        for i in range(0, len(pred_semantic_list))
                    ]
                    audio_frag_end_idx = [sum(audio_frag_idx[: i + 1]) for i in range(0, len(audio_frag_idx))]
                    all_pred_semantic = np.concatenate(pred_semantic_list, axis=0)[None, None, :]
                    _batch_phones = np.concatenate(batch_phones, axis=0)[None, :]
                    _batch_audio_fragment = self.vits_model.decode(
                        all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor
                    )
                    if _batch_audio_fragment.ndim == 3:
                        _batch_audio_fragment = _batch_audio_fragment[0, 0, :]
                    audio_frag_end_idx.insert(0, 0)
                    batch_audio_fragment = [
                        _batch_audio_fragment[audio_frag_end_idx[i - 1] : audio_frag_end_idx[i]]
                        for i in range(1, len(audio_frag_end_idx))
                    ]
                else:
                    for i, idx in enumerate(idx_list):
                        phones = batch_phones[i][None, :]
                        _pred_semantic = pred_semantic_list[i][-idx:][None, None, :]
                        audio_fragment = self.vits_model.decode(
                            _pred_semantic, phones, refer_audio_spec, speed=speed_factor
                        )
                        if audio_fragment.ndim == 3:
                            audio_fragment = audio_fragment[0, 0, :]
                        batch_audio_fragment.append(audio_fragment)

                t5 = ttime()
                t_45 += t5 - t4
                if return_fragment:
                    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield self.audio_postprocess(
                        [batch_audio_fragment],
                        self.configs.sampling_rate,
                        None,
                        speed_factor,
                        False,
                        fragment_interval,
                    )
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate), dtype=np.int16)
                    return

            if not return_fragment:
                print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                if len(audio) == 0:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate), dtype=np.int16)
                    return
                yield self.audio_postprocess(
                    audio,
                    self.configs.sampling_rate,
                    batch_index_list,
                    speed_factor,
                    split_bucket,
                    fragment_interval,
                )

        except Exception as e:
            traceback.print_exc()
            yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate), dtype=np.int16)
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        gc.collect()

    def audio_postprocess(
        self,
        audio: List[List[np.ndarray]],
        sr: int,
        batch_index_list: list = None,
        speed_factor: float = 1.0,
        split_bucket: bool = True,
        fragment_interval: float = 0.3,
    ) -> Tuple[int, np.ndarray]:
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * fragment_interval),
            dtype=np.float32,
        )

        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                audio_fragment = np.asarray(audio_fragment, dtype=np.float32)
                max_audio = np.abs(audio_fragment).max()
                if max_audio > 1:
                    audio_fragment = audio_fragment / max_audio
                audio[i][j] = np.concatenate([audio_fragment, zero_wav], axis=0)

        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)
        else:
            audio = sum(audio, [])

        audio = np.concatenate(audio, axis=0)
        audio = (audio * 32768).astype(np.int16)
        return sr, audio
