from datetime import datetime
from external.mongo import find_session, insert_session, push_session_message, find_session_messages


def get_or_create_session(db, user_id):
    """세션이 없으면 생성, 있으면 반환"""
    session = find_session(db, user_id)
    if not session:
        insert_session(db, user_id, datetime.now())
        session = find_session(db, user_id)
    return session


def append_message(db, user_id, question, answer, model):
    """질문과 답변을 세션에 저장"""
    user_message = {
        "role": "user",
        "content": question,
        "model": model,
        "timestamp": datetime.now()
    }
    push_session_message(db, user_id, user_message)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "model": model,
        "timestamp": datetime.now()
    }
    push_session_message(db, user_id, assistant_message)


def get_recent_messages(db, user_id, n=5):
    """최근 N개 대화 조회 (최근 N개 = 2N개 메시지)"""
    messages = find_session_messages(db, user_id)
    if not messages:
        return []

    recent = messages[-n*2:]  # n개 대화 = 2n개 메시지 (user + assistant)

    return [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in recent
    ]
