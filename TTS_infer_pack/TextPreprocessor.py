import os
import sys
import re
from typing import Dict, List, Tuple, Callable
from tqdm import tqdm
import numpy as np
import LangSegment
from text.symbols2 import symbols as symbols_v2

# ==============================================================================
# Symbols & Sequence Mapping (symbols2)
# ==============================================================================
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2)}

def cleaned_text_to_sequence(cleaned_text, version="v2"):
    unk_id = _symbol_to_id_v2.get("UNK", 0)
    return [_symbol_to_id_v2.get(symbol, unk_id) for symbol in cleaned_text]


# ==============================================================================
# Text Segmentation Methods (previously text_segmentation_method.py)
# ==============================================================================
punctuation = set(['!', '?', '…', ',', '.', '-', " "])
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
METHODS = dict()

def get_method(name: str) -> Callable:
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError(f"Method {name} not found")
    return method

def get_method_names() -> list:
    return list(METHODS.keys())

def register_method(name):
    def decorator(func):
        METHODS[name] = func
        return func
    return decorator

def split_big_text(text, max_len=510):
    punc_pattern = "".join(splits)
    segments = re.split('([' + punc_pattern + '])', text)
    result = []
    current_segment = ''
    for segment in segments:
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment
    if current_segment:
        result.append(current_segment)
    return result

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

@register_method("cut0")
def cut0(inp):
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "\n"

@register_method("cut1")
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

@register_method("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

@register_method("cut3")
def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

@register_method("cut4")
def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

@register_method("cut5")
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []
    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)
    if items:
        mergeitems.append("".join(items))
    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts: list, threshold: int) -> list:
    if len(texts) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def replace_consecutive_punctuation(text: str) -> str:
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    return re.sub(pattern, r'\1', text)


# ==============================================================================
# Japanese G2P (pyopenjtalk with user.dict pre-loaded)
# ==============================================================================
import pyopenjtalk

_userdic_initialized = False
def _init_japanese_userdict():
    global _userdic_initialized
    if _userdic_initialized:
        return
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(cur_dir)
        userdic_bin = os.path.join(root_dir, "text", "ja_userdic", "user.dict")
        if os.path.exists(userdic_bin):
            pyopenjtalk.update_global_jtalk_with_user_dict(userdic_bin)
    except Exception:
        pass
    _userdic_initialized = True

_init_japanese_userdict()

_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)
_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]

def _ja_post_replace_ph(ph):
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": ".", "·": ",", "、": ",", "...": "…",
    }
    return rep_map.get(ph, ph)

def _symbols_to_japanese_fn(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text

def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

def pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True):
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)
    phones = []
    for n in range(N):
        lab_curr = labels[n]
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                phones.append("?" if e3 == 1 else "$")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        elif a2 == 1 and a2_next == 2:
            phones.append("[")
    return phones

def preprocess_jap(text, with_prosody=False):
    text = _symbols_to_japanese_fn(text).lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text_out = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if with_prosody:
                text_out += pyopenjtalk_g2p_prosody(sentence)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text_out += p.split(" ")
        if i < len(marks):
            if marks[i] == " ":
                continue
            text_out += [marks[i].replace(" ", "")]
    return text_out

def japanese_g2p(norm_text, with_prosody=True):
    phones = preprocess_jap(norm_text, with_prosody)
    return [_ja_post_replace_ph(i) for i in phones]


# ==============================================================================
# Korean G2P (g2pk2 + jamo + ko_pron)
# ==============================================================================
from jamo import h2j, j2hcj
import ko_pron
from g2pk2 import G2p

_g2p_ko = G2p()

_korean_classifiers = '군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통'
_hangul_divided = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㅘ', 'ㅗㅏ'), ('ㅙ', 'ㅗㅐ'), ('ㅚ', 'ㅗㅣ'), ('ㅝ', 'ㅜㅓ'),
    ('ㅞ', 'ㅜㅔ'), ('ㅟ', 'ㅜㅣ'), ('ㅢ', 'ㅡㅣ'), ('ㅑ', 'ㅣㅏ'),
    ('ㅒ', 'ㅣㅐ'), ('ㅕ', 'ㅣㅓ'), ('ㅖ', 'ㅣㅔ'), ('ㅛ', 'ㅣㅗ'), ('ㅠ', 'ㅣㅜ')
]]
_latin_to_hangul = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', '에이'), ('b', '비'), ('c', '시'), ('d', '디'), ('e', '이'),
    ('f', '에프'), ('g', '지'), ('h', '에이치'), ('i', '아이'), ('j', '제이'),
    ('k', '케이'), ('l', '엘'), ('m', '엠'), ('n', '엔'), ('o', '오'),
    ('p', '피'), ('q', '큐'), ('r', '아르'), ('s', '에스'), ('t', '티'),
    ('u', '유'), ('v', '브이'), ('w', '더블유'), ('x', '엑스'), ('y', '와이'), ('z', '제트')
]]

def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i:i+3] == 'ㅇㅡㄹ' or text[i:i+3] == 'ㄹㅡㄹ') and text[i+3] == ' ' and text[i+4] == 'ㄹ':
            new_text += text[i:i+3] + ' ' + 'ㄴ'
            i += 5
        else:
            new_text += text[i]
            i += 1
    new_text += text[i:]
    return new_text

def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text

def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text

def hangul_number(num, sino=True):
    num = re.sub(',', '', num)
    if num == '0':
        return '영'
    if not sino and num == '20':
        return '스무'
    digits = '123456789'
    names = '일이삼사오육칠팔구'
    digit2name = {d: n for d, n in zip(digits, names)}
    modifiers = '한 두 세 네 다섯 여섯 일곱 여덟 아홉'
    decimals = '열 스물 서른 마흔 쉰 예순 일흔 여든 아흔'
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}
    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, '')
            elif i == 1:
                name = digit2name.get(digit, '') + '십'
                name = name.replace('일십', '십')
        else:
            if i == 0:
                name = digit2mod.get(digit, '')
            elif i == 1:
                name = digit2dec.get(digit, '')
        if digit == '0':
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)):]
                if ''.join(last_three) == '':
                    spelledout.append('')
                    continue
            else:
                spelledout.append('')
                continue
        if i == 2:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 3:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 4:
            name = digit2name.get(digit, '') + '만'
            name = name.replace('일만', '만')
        elif i == 5:
            name = digit2name.get(digit, '') + '십'
            name = name.replace('일십', '십')
        elif i == 6:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 7:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 8:
            name = digit2name.get(digit, '') + '억'
        elif i == 9:
            name = digit2name.get(digit, '') + '십'
        elif i == 10:
            name = digit2name.get(digit, '') + '백'
        elif i == 11:
            name = digit2name.get(digit, '') + '천'
        elif i == 12:
            name = digit2name.get(digit, '') + '조'
        elif i == 13:
            name = digit2name.get(digit, '') + '십'
        elif i == 14:
            name = digit2name.get(digit, '') + '백'
        elif i == 15:
            name = digit2name.get(digit, '') + '천'
        spelledout.append(name)
    return ''.join(elem for elem in spelledout)

def number_to_hangul(text):
    tokens = set(re.findall(r'(\d[\d,]*)([\uac00-\ud71f]+)', text))
    for token in tokens:
        num, classifier = token
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f'{num}{classifier}', f'{spelledout}{classifier}')
    digits = '0123456789'
    names = '영일이삼사오육칠팔구'
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text

def _ko_post_replace_ph(ph):
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": ".", "·": ",", "、": ",", "...": "…", " ": "空",
    }
    if ph in rep_map:
        ph = rep_map[ph]
    if ph in symbols_v2:
        return ph
    return "停"

def korean_g2p(text):
    text = latin_to_hangul(text)
    text = _g2p_ko(text)
    text = divide_hangul(text)
    text = fix_g2pk2_error(text)
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    return [_ko_post_replace_ph(i) for i in text]


# ==============================================================================
# English G2P (Lazy loaded on actual English text demand)
# ==============================================================================
_english_module = None
def get_english_module():
    global _english_module
    if _english_module is None:
        from text import english
        _english_module = english
    return _english_module


# ==============================================================================
# Direct Clean Text (replaces dynamic __import__ in cleaner.py)
# ==============================================================================
def clean_text_direct(text: str, language: str, version: str = "v2"):
    lang = language.lower().replace("all_", "")
    if lang in {"ko", "korean"}:
        norm_text = text
        phones = korean_g2p(norm_text)
        word2ph = None
    elif lang in {"ja", "japanese"}:
        norm_text = replace_consecutive_punctuation(text)
        phones = japanese_g2p(norm_text)
        word2ph = None
    elif lang in {"en", "english"}:
        en_mod = get_english_module()
        norm_text = text
        phones = en_mod.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        norm_text = text
        phones = korean_g2p(norm_text)
        word2ph = None

    phones = ['UNK' if ph not in symbols_v2 else ph for ph in phones]
    return phones, word2ph, norm_text


# ==============================================================================
# TextPreprocessor Class (100% Pure NumPy)
# ==============================================================================
class TextPreprocessor:
    def __init__(self, bert_model=None, tokenizer=None, device: str = "cpu"):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        print("############ 텍스트 분할 ############")
        text = replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print("############ 텍스트 특징 추출 ############")
        for t in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(t, lang, version)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if (text[0] not in splits and len(get_first(text)) < 4):
            text = "。" + text if lang != "en" else "." + text
        print("실제 입력된 목표 텍스트:")
        print(text)

        seg_method = get_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for item in _texts:
            if len(item.strip()) == 0:
                continue
            if not re.sub(r"\W+", "", item):
                continue
            if item[-1] not in splits:
                item += "。" if lang != "en" else "."
            if len(item) > 510:
                texts.extend(split_big_text(item))
            else:
                texts.append(item)

        print("실제 입력된 목표 텍스트(문장 분리 후):")
        print(texts)
        return texts

    def segment_and_extract_feature_for_text(self, text: str, language: str, version: str = "v2") -> Tuple[list, np.ndarray, str]:
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str = "v2", final: bool = False):
        clean_lang = language.replace("all_", "").lower()

        # Fast path 1: Single language mode or no English in text
        if language.startswith("all_") or clean_lang in {"ja", "ko"}:
            has_english = bool(re.search(r'[a-zA-Z]', text))
            if not has_english or language.startswith("all_"):
                phones, word2ph, norm_text = self.clean_text_inf(text, clean_lang, version)
                bert = np.zeros((1024, len(phones)), dtype=np.float32)
                if not final and len(phones) < 6:
                    return self.get_phones_and_bert("." + text, language, version, final=True)
                return phones, bert, norm_text

        # Path 2: Multilingual / Mixed mode (Korean/Japanese with embedded English)
        textlist = []
        langlist = []
        LangSegment.setfilters(["ja", "en", "ko"])
        for tmp in LangSegment.getTexts(text):
            seg_text = tmp["text"]
            if tmp["lang"] == "en" and re.search(r'[a-zA-Z]', seg_text):
                langlist.append("en")
            else:
                langlist.append(clean_lang if clean_lang in {"ja", "ko"} else tmp["lang"])
            textlist.append(seg_text)

        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            seg_lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], seg_lang, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, seg_lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)

        bert = np.concatenate(bert_list, axis=1) if bert_list else np.zeros((1024, 0), dtype=np.float32)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert, norm_text

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        phones, word2ph, norm_text = clean_text_direct(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        return np.zeros((1024, len(phones)), dtype=np.float32)

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("유효한 텍스트를 입력하세요")
        for text in texts:
            if text not in [None, " ", ""]:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        return replace_consecutive_punctuation(text)
