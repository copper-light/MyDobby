# Docker 실행 가이드

## 전체 구성

```
docker-compose.yaml          # MongoDB + Qdrant + Langfuse + API 서버
docker-compose.airflow.yaml  # Airflow (스케줄러, 웹서버)
Dockerfile                   # MyDobby API 서버 이미지
```

## 사전 준비

루트의 `.env` 파일이 있어야 합니다. (`.env.example` 참고)

Airflow에서 사용할 Fernet 키 생성:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

`.env`에 아래 항목 추가:
```
AIRFLOW_FERNET_KEY=<위에서 생성한 키>
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin
AIRFLOW_SECRET_KEY=<임의 문자열>
```

## 실행 순서

### 1단계 — 인프라 서비스 시작 (MongoDB · Qdrant · Langfuse)

```bash
cd docker
docker compose up -d mongodb qdrant langfuse-db langfuse
```

### 2단계 — API 서버 빌드 & 시작

```bash
docker compose up -d --build apiserver
```

> GPU가 없는 환경: `docker-compose.yaml`의 `deploy.resources.reservations` 블록을 제거하세요.

### 3단계 — Airflow 시작 (데이터 수집 스케줄링 필요 시)

```bash
# 먼저 1단계 네트워크가 생성되어 있어야 합니다
docker compose -f docker-compose.airflow.yaml up -d
```

## 서비스별 접속 주소

| 서비스 | 주소 |
|--------|------|
| MongoDB | `mongodb://localhost:27017` |
| Qdrant | `http://localhost:6333` |
| Langfuse | `http://localhost:3000` |
| MyDobby API | `http://localhost:18000` |
| Airflow | `http://localhost:8080` |

## 종료

```bash
# 인프라 + API 서버
docker compose down

# Airflow
docker compose -f docker-compose.airflow.yaml down

# 볼륨(데이터)까지 삭제
docker compose down -v
```

## 데이터 볼륨

| 볼륨 | 내용 |
|------|------|
| `mongodb_data` | MongoDB 문서 데이터 |
| `qdrant_data` | Qdrant 벡터 데이터 |
| `langfuse_db` | Langfuse PostgreSQL 데이터 |
| `airflow_db` | Airflow 메타데이터 |
| `airflow_logs` | Airflow 실행 로그 |
