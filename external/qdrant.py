from uuid import uuid5, NAMESPACE_DNS
from qdrant_client import models
from datetime import datetime, timezone
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
import config.settings as settings

def insert_parent_chunk(client, doc_id, chunk_id, title, chunks, doc_path, collection_name = "parent_chunks"):
    points = [
        models.PointStruct(
            id = uuid5(NAMESPACE_DNS, chunk_id),
            vector=[0],
            payload={
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title,
                "text": chunks,
                "doc_path": doc_path,
                "create_at": datetime.now(timezone.utc)
            }   
        ),
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

def insert_child_chunk(client, dense_vector, keywords, doc_id, parent_id, chunk_id, title, chunks, collection_name = "child_chunks"):
    points = [
        models.PointStruct(
            id = uuid5(NAMESPACE_DNS, chunk_id),
            vector={
                "dense": dense_vector,
                "sparse": models.Document(
                    text=keywords,
                    model="Qdrant/bm25",
                    options={
                        "k1": settings.BM25_K1,
                        "b": settings.BM25_B,
                    },
                ),
            },
            
            payload={
                "doc_id": doc_id,
                "parent_id": parent_id,
                "chunk_id": chunk_id,
                "title": title,
                "text": chunks,
                "keywords": keywords,
                "create_at": datetime.now(timezone.utc)
            }   
        ),
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

def find_parent_chunk(client, chunk_id, collection_name = "parent_chunks"):
    points, next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter= Filter(
            must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
        ),
        with_vectors=False,
        with_payload=True
    )
    return points

# def find_child_chunk(client, vector, topk= 20, collection_name = "child_chunks"):
#     result = client.query_points(
#         collection_name=collection_name,
#         query=vector,
#         limit=topk,
#         with_payload=True,
#         with_vectors=False,
#     )
#     return result

def find_child_chunk(client, dense_vector, keywords, topk= 20, collection_name = "child_chunks"):
    sparse_vector = models.Document(
        text=keywords,
        model="Qdrant/bm25",
        options={
            # "avg_len": avg_document_length,  # |D| and avgdl 역할
            "k1": 1.2,
            "b": 0.75,
        },
    )

    result = client.query_points(
        collection_name=collection_name,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=topk,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=topk,
            ),
        ],
        limit=topk,
        with_payload=True,
        with_vectors=False,
    )
    return result


def reset_collection(client, collection_names = ["child_chunks", "parent_chunks"]):
    for name in collection_names:
        client.delete(
            collection_name=name,
            points_selector=FilterSelector(
                filter=Filter(must=[])
            ),
            wait=True
        )


"""
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

dense_results = client.search(
    collection_name="hybrid_docs",
    query_vector=dense_query_vec,
    using="dense",
    limit=20,
)

sparse_results = client.search(
    collection_name="hybrid_docs",
    query_vector=sparse_query_doc,
    using="sparse",
    limit=20,
)

# dense 비중 0.7, sparse 비중 0.3 사용
EMBED_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

k = 60  # RRF k 값
scores = []
for point_id in set(p.id for p in dense_results + sparse_results):
    dense_rank = next((i+1 for i, p in enumerate(dense_results) if p.id == point_id), 999)
    sparse_rank = next((i+1 for i, p in enumerate(sparse_results) if p.id == point_id), 999)

    rrf_embed = 1.0 / (k + dense_rank)
    rrf_keyword = 1.0 / (k + sparse_rank)

    score = EMBED_WEIGHT * rrf_embed + KEYWORD_WEIGHT * rrf_keyword
    scores.append((point_id, score))

top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

for pid, score in top5:
    print(pid, score)
"""


"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "bm25 keyword search with embedding"

dense_query_vec = dense_model.encode(query).tolist()
sparse_query_doc = models.Document(
    text=query,
    model="Qdrant/bm25",
    options={"avg_len": avg_document_length, "k1": 1.2, "b": 0.75},
)

result = client.query_points(
    collection_name=collection_name,
    prefetch=[
        models.Prefetch(
            query=dense_query_vec,
            using="dense",
            limit=20,
        ),
        models.Prefetch(
            query=sparse_query_doc,
            using="sparse",
            limit=20,
        ),
    ],
    query=models.FusionQuery(
        fusion=models.Fusion.RRF
    ),
    limit=5,
    with_payload=True,
)

for p in result.points:
    print(f"ID: {p.id}, Score: {p.score:.4f}")
    print(f"Text: {p.payload['text']}\n")
"""


if __name__ == "__main__":
    from qdrant_client import QdrantClient
    client = QdrantClient()

    result = client.query_points(
        collection_name="parent_chunks",
        with_vectors=False,
        with_payload=True,
        limit=1000
    )
    # print(points[0])

    import pandas as pd
    # pd.DataFrame(points).to_csv("parent_chunks.csv", index=False)

    df = pd.DataFrame()

    for p in result.points:
        df = pd.concat([df, pd.DataFrame([{
            "chunk_id": p.payload["chunk_id"],
            "doc_id": p.payload["doc_id"],
            "title": p.payload["title"],
            "text": p.payload["text"],
            "create_at": p.payload["create_at"]
        }])])
    
    df.to_csv("parent_chunks.csv", index=False)