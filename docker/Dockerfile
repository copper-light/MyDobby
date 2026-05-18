# CUDA 지원 Python 베이스 이미지 (PyTorch 2.10 + CUDA 13.0 기준)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 시스템 패키지 설치 (MeCab 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# 의존성 파일만 먼저 복사 (레이어 캐시 활용)
COPY pyproject.toml poetry.lock ./

# 가상 환경 없이 시스템에 직접 설치
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --without dev

# 소스 코드 복사 (모델 바이너리 제외)
COPY apiserver/ ./apiserver/
COPY config/ ./config/
COPY external/ ./external/
COPY gather_data/ ./gather_data/
COPY models/ ./models/
COPY retrieve/ ./retrieve/
COPY session/ ./session/
COPY utils/ ./utils/

# 모델 파일은 볼륨 마운트로 제공 (docker-compose.yaml의 sft 볼륨 참고)

EXPOSE 18000

CMD ["uvicorn", "apiserver.main:app", "--host", "0.0.0.0", "--port", "18000"]
