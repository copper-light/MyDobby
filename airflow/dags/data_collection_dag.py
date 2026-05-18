"""
MyDobby 데이터 수집 파이프라인 DAG (증분 처리)

Task 흐름:
  collect_notion → process_chunking

실행 이력은 MongoDB에 기록되며, 각 단계는 마지막 실행 이후 변경된 데이터만 처리합니다.
  - collect_notion : history_gather_data 컬렉션 기준으로 새 Notion 문서만 수집
  - process_chunking: history_process_rag  컬렉션 기준으로 새로 수집된 문서만 청킹/임베딩

전체 재구축이 필요할 때는 mydobby_full_rebuild DAG를 사용하세요.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mydobby",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


with DAG(
    dag_id="mydobby_data_collection",
    default_args=default_args,
    description="Notion → MongoDB → Qdrant 증분 수집 파이프라인",
    schedule="0 3 * * *",   # 매일 새벽 3시
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mydobby", "rag", "incremental"],
) as dag:

    def _collect_notion(**context):
        from gather_data.gather import collect_notion

        count = collect_notion()
        print(f"[collect_notion] 수집 완료: {count}개 문서")
        return count

    def _process_chunking(**context):
        from retrieve.embedding import process_chunking

        count = context["ti"].xcom_pull(task_ids="collect_notion")
        if count == 0:
            print("[process_chunking] 새 문서 없음 — 건너뜀")
            return

        process_chunking()
        print("[process_chunking] 청킹 + 임베딩 + Qdrant 적재 완료")

    collect_notion = PythonOperator(
        task_id="collect_notion",
        python_callable=_collect_notion,
    )

    process_chunking = PythonOperator(
        task_id="process_chunking",
        python_callable=_process_chunking,
        execution_timeout=timedelta(hours=2),
    )

    collect_notion >> process_chunking


with DAG(
    dag_id="mydobby_full_rebuild",
    default_args=default_args,
    description="Qdrant 초기화 후 MongoDB 전체 문서 재임베딩",
    schedule=None,   # 수동 트리거 전용
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mydobby", "rag", "rebuild"],
) as rebuild_dag:

    def _reset_qdrant(**context):
        from external.connect import get_qdrant_client
        from external.qdrant import reset_collection

        client = get_qdrant_client()
        reset_collection(client)
        print("[reset_qdrant] Qdrant 컬렉션 초기화 완료")

    def _rebuild_chunking(**context):
        from retrieve.embedding import process_chunking
        from external.connect import get_mongo_db
        from external.mongo import find_raw_documents_for_rag

        # history_process_rag를 비워 전체 문서가 대상이 되도록 함
        db = get_mongo_db()
        db["history_process_rag"].drop()
        print("[rebuild_chunking] history_process_rag 초기화 완료")

        process_chunking()
        print("[rebuild_chunking] 전체 재구축 완료")

    reset_qdrant = PythonOperator(
        task_id="reset_qdrant",
        python_callable=_reset_qdrant,
    )

    rebuild_chunking = PythonOperator(
        task_id="rebuild_chunking",
        python_callable=_rebuild_chunking,
        execution_timeout=timedelta(hours=6),
    )

    reset_qdrant >> rebuild_chunking
