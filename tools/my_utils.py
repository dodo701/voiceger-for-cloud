import os
import traceback
import numpy as np
import soundfile as sf
import soxr


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
        raise RuntimeError("오디오 로드 실패")

    return np.asarray(audio, dtype=np.float32).flatten()


def clean_path(path_str: str):
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return path_str.strip(" ").strip("'").strip("\n").strip('"').strip(" ").strip("\u202a")
