import os
import traceback
import numpy as np
import soundfile as sf
import soxr
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language=os.environ.get("language", "Auto"))


class GradioMock:
    @staticmethod
    def Warning(message):
        print(f"警告: {message}")


gr = GradioMock()


def load_audio(file, sr):
    try:
        file = clean_path(file)
        if not os.path.exists(file):
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        audio, orig_sr = sf.read(file, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = soxr.resample(audio, orig_sr, sr)
    except Exception:
        traceback.print_exc()
        raise RuntimeError(i18n("音频加载失败"))

    return np.asarray(audio, dtype=np.float32).flatten()


def clean_path(path_str: str):
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return path_str.strip(" ").strip("'").strip("\n").strip('"').strip(" ").strip("\u202a")
