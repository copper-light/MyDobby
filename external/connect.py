from qdrant_client import QdrantClient
from pymongo import MongoClient
from langfuse import Langfuse
import config.settings as settings

def get_qdrant_client():
    return QdrantClient(url=settings.QDRANT_URL)

def get_mongo_db():
    client = MongoClient(settings.MONGO_URL)
    return client[settings.MONGO_DB_NAME]

def get_langfuse_client():
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )


if __name__ == "__main__":
    client = get_langfuse_client()
    span = client.start_observation(
        as_type="span",
        name="step-1")
    span.update(input="하하하 질문?")
    span.update(output={"keywords": ["Langfuse", "연동", "방법"]})

    span2 = span.start_observation(
        as_type="span",
        name= "step-2")
    span2.update(output={"keywords": ["Langfuse", "2222", "2222"]})
    span2.end()

    span.end()
    client.flush()
    