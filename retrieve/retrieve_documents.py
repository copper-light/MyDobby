from external.connect import get_qdrant_client
from external.qdrant import find_child_chunk, find_parent_chunk
from models.load_model import get_model
from retrieve.embedding import extract_keyword
import config.settings as settings
import json

def retrieve_child_chunks(client, query, topk = 20, score_threshold = settings.SCORE_THRESHOLD):
    emb_model = get_model("koelectra")
    # emb_model = get_model("qwen3_vl")
    
    keywords = query['keywords']
    ret = []
    searched_ids = set()
    for q in query["questions"]:
        embed = emb_model.embed(q)
        result = find_child_chunk(client, embed, keywords, topk)
        for point in result.points:
            if point.id in searched_ids:
                continue
            if point.score < score_threshold:
                continue
            searched_ids.add(point.id)
            ret.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            })

    return ret, keywords


def retrieve_documents(query, retrieve_topk = settings.RETRIEVE_TOPK, rerank_topk = settings.RERANK_TOPK, trace = None):
    print(f"retrieve_documents query : {query}")
    client = get_qdrant_client()

    with trace.start_as_current_observation(as_type="retrieval", name="retrieve_child") as span:
        span.update(input={"query": query, "retrieve_topk": retrieve_topk})
        child_chunks, keywords = retrieve_child_chunks(client, query, retrieve_topk)
        if len(child_chunks) == 0:
            return None
        span.update(output={"child_chunks": child_chunks, "keywords": keywords})
    
    with trace.start_as_current_observation(as_type="retrieval", name="reranking") as span:
        documents = []
        for child_chunk in child_chunks:
            documents.append(child_chunk["payload"]["title"] + "\n" + child_chunk["payload"]["text"])

        span.update(input={"query": query, "documents": documents, "rerank_topk": rerank_topk})
        model = get_model("qwen3_reranker")
        selected_docs = model.rerank(query["questions"][0], documents, topk=retrieve_topk)
        child_chunks = [child_chunks[i] for i, score in selected_docs]

        for i, (_,score) in enumerate(selected_docs):
            child_chunks[i]["rank_score"] = score
        span.update(output=child_chunks)

        # child_chunks = child_chunks[-rerank_topk:]
        reranked_child_chunks = [child_chunk for child_chunk in child_chunks if settings.RANK_SCORE_THRESHOLD <= child_chunk["rank_score"]]

        # , topk = rerank_topk, threshold = 0.15


    with trace.start_as_current_observation(as_type="retrieval", name="retrieve_parent") as span:
        parent_chunks = {}
        for child_chunk in reranked_child_chunks:
            parent_id = child_chunk["payload"]["parent_id"]
            parent_chunks[parent_id] = parent_chunks.get(parent_id, 0) + child_chunk["rank_score"]
        sorted_parent_chunks = sorted(parent_chunks.items(), key=lambda x: x[1], reverse=True)[:rerank_topk]

        span.update(input={
            "parent_chunks": sorted_parent_chunks
        })
        
        
        ret = []
        for chunk_id, score in sorted_parent_chunks:
            parent_chunk = find_parent_chunk(client, chunk_id)
            
            for p in parent_chunk:
                ret.append({
                    "title": p.payload["title"],
                    "text": p.payload["text"],
                    "doc_path": p.payload["doc_path"] + " > " + p.payload["title"],
                    "pick_number": float(score)
                })

        span.update(output=ret)

    return ret

if __name__ == "__main__":
    from external.connect import get_langfuse_client
    client = get_langfuse_client()
    query = {
        "questions": [
            "파이썬으로 웹 크롤링하는 방법"
        ],
        "keywords": "파이썬 크롤링 웹 방법"
    }
    retrieve_documents(query, trace= client)
