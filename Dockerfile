# ==============================================================================
# 아직 onnx inferencee 파이프라인이 미완성 상태임.
# ==============================================================================
FROM python:3.11-slim-bookworm

# Python 및 런타임 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# 1. 시스템 필수 패키지 설치 (ffmpeg 제거, 가벼운 libsndfile1만 사용)
# - libsndfile1: soundfile C 백엔드 라이브러리
# - build-essential, cmake: pyopenjtalk / python-mecab-ko 빌드 폴백 대비
# - libgomp1: OpenMP 런타임 (PyTorch, ONNX Runtime, OpenVINO 멀티스레딩)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 2. 의존성 파일 복사 및 설치 (런타임 전용, torch 미포함)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
        && pip install -r requirements.txt

# 3. 런타임 외부 다운로드 원천 차단 (Pre-baking)
# - NLTK 품사 태거 및 CMU 사전 다운로드 (/usr/local/share/nltk_data)
# - pyopenjtalk 기본 일본어 사전(open_jtalk_dic_utf_8-1.11) 다운로드 및 압축 해제
RUN mkdir -p /usr/local/share/nltk_data && python -c "\
import nltk, pyopenjtalk; \
nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/share/nltk_data'); \
nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/local/share/nltk_data'); \
nltk.download('cmudict', download_dir='/usr/local/share/nltk_data'); \
pyopenjtalk._lazy_init()"

# 4. 프로젝트 소스 코드 복사 (.dockerignore에 의해 무거운 로컬 가중치/venv 제외)
COPY . /app

# 5. 모델 다운로드 스크립트(download.py) 실행하여 컨테이너 이미지 내에 가중치 내장
RUN python download.py

# 6. 모델 다운로드 완료 후 런타임에는 외부 통신을 하지 않도록 오프라인 모드 고정
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 7. 일본어 사전 심볼릭 링크 연결 및 LangSegment 패키지 오류 패치
RUN python -c "\
import os, pyopenjtalk; \
dict_dir = pyopenjtalk.OPEN_JTALK_DICT_DIR; \
dict_dir = dict_dir.decode('utf-8') if isinstance(dict_dir, bytes) else dict_dir; \
mecab_dir = os.path.dirname(dict_dir); \
target = os.path.join(mecab_dir, 'user.dict'); \
src = '/app/GPT_SoVITS/text/ja_userdic/user.dict'; \
(os.remove(target) if os.path.exists(target) or os.path.islink(target) else None); \
os.symlink(src, target) if os.path.exists(src) else None" && \
    python -c "import site, os; p = os.path.join(site.getsitepackages()[0], 'LangSegment', '__init__.py'); open(p, 'w').write(open(p).read().replace('setLangfilters,getLangfilters,', ''))"

# Cloud Run 기본 포트
ENV PORT=8080
EXPOSE 8080

# 기본 실행 명령어 (API 서버 엔드포인트 추가 시 변경 가능)
CMD ["python", "infer_tts_minimal.py"]
