# Airflow 설정 가이드

## 설치

```bash
pip install apache-airflow
```

## 프로젝트 모듈 경로 설정

DAG에서 `gather_data`, `retrieve`, `external`, `config` 등 프로젝트 모듈을 import하려면
Airflow 워커가 프로젝트 루트를 Python 경로로 인식해야 합니다.

```bash
# 방법 1: 프로젝트를 패키지로 설치 (권장)
cd /path/to/MyDobby
poetry install

# 방법 2: PYTHONPATH 환경변수 설정
export PYTHONPATH=/path/to/MyDobby:$PYTHONPATH
```

## DAG 등록

```bash
# AIRFLOW_HOME의 dags 폴더에 심볼릭 링크 또는 복사
cp airflow/dags/data_collection_dag.py $AIRFLOW_HOME/dags/

# 또는 airflow.cfg의 dags_folder를 이 디렉토리로 지정
```

## DAG 목록

| DAG ID | 스케줄 | 설명 |
|--------|--------|------|
| `mydobby_data_collection` | 매일 03:00 | Notion → MongoDB → Qdrant 전체 수집 파이프라인 |

## 태스크 흐름

```
collect_notion → reset_qdrant → process_chunking
```

| 태스크 | 설명 |
|--------|------|
| `collect_notion` | Notion BFS 크롤링 후 새 문서를 MongoDB에 저장, 수집 수를 XCom으로 반환 |
| `reset_qdrant` | Qdrant의 child_chunks / parent_chunks 전체 초기화 (새 문서 없으면 건너뜀) |
| `process_chunking` | MongoDB 전체 문서를 청킹 + 임베딩 후 Qdrant 재적재 (새 문서 없으면 건너뜀) |

## 수동 실행

```bash
airflow dags trigger mydobby_data_collection
```
