# 1. 패키지 설치 (예시)

RUN pip install LangSegment==0.2.0

# 2. LangSegment 버그 자동 패치 (site-packages 경로를 동적으로 찾아 문자열 삭제)

RUN python -c "import site, os; p = os.path.join(site.getsitepackages()[0], 'LangSegment', '__init__.py'); open(p, 'w').write(open(p).read().replace('setLangfilters,getLangfilters,', ''))"
