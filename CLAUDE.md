# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MyDobby is a local RAG (Retrieval-Augmented Generation) system optimized for Korean language, integrating Notion as a data source, MongoDB + Qdrant as storage, Slack as a user interface, and a FastAPI server with an OpenAI-compatible API.

## Commands

```bash
# Install dependencies
poetry install

# Start the API server (loads Llama model on startup; takes time)
python -m apiserver.main
# or
uvicorn apiserver.main:app --host 0.0.0.0 --port 18000 --reload

# Collect data from Notion and store to MongoDB
python -m gather_data.gather

# Run Jupyter for notebooks (SFT training, embedding experiments)
jupyter lab
```

External services that must be running locally:
- MongoDB on `localhost:27017`
- Qdrant on `localhost:6333`
- Langfuse on `localhost:3000`

## Architecture

### Two Pipelines

**Data Collection (offline):**
`gather_data/gather.py` → MongoDB (`raw_documents`) → chunking/embedding → Qdrant (`parent_chunks`, `child_chunks`)

- Notion blocks are traversed breadth-first; only blocks >50 chars and newer than the last collection timestamp are stored.
- `parent_chunking()` splits by `[HEADER]` markers; `chunk_to_children()` does sentence-level chunking with ~100-150 char overlap.
- Each child chunk gets two vectors: dense (Qwen3-VL-Embedding-8B) and sparse (BM25 keywords from MeCab POS tagger).

**Inference (online):**
User query → query expansion → keyword extraction → hybrid retrieval → reranking → parent chunk aggregation → LLM generation → response

- `apiserver/service.py:request_chat()` orchestrates the full pipeline with Langfuse tracing at each step.
- Query expansion uses Llama to generate 2 additional query variants.
- Hybrid search in Qdrant uses RRF (Reciprocal Rank Fusion, k=60) to combine dense and sparse results.
- Qwen3-Reranker-0.6B cross-encoder re-scores candidates; those below `rank_score=0.1` are dropped.
- Top 5 parent chunks (aggregated by summing child rerank scores per `parent_id`) form the context.

### Key Modules

| Module | Role |
|---|---|
| `apiserver/main.py` | FastAPI entry point; endpoints `/rag`, `/chat/completions` (OpenAI-compatible), `/context` |
| `apiserver/service.py` | RAG pipeline orchestrator (`request_chat()`) |
| `apiserver/pompt.py` | Prompt templates; loads from Langfuse or falls back to hardcoded strings |
| `apiserver/slack.py` | Slack bot via async socket mode; replies in thread |
| `external/connect.py` | Connection factories for Qdrant, MongoDB, Langfuse |
| `external/qdrant.py` | Hybrid insert/search; parent and child chunk collections |
| `external/mongo.py` | Raw document storage and collection-timestamp tracking |
| `models/load_model.py` | Lazy-loading model registry (`get_model(name)`) |
| `retrieve/retrieve_documents.py` | Three-stage retrieval: child search → rerank → parent aggregation |
| `retrieve/embedding.py` | `parent_chunking()`, `chunk_to_children()`, `extract_keyword()` |
| `gather_data/gather.py` | Notion crawler + chunking/embedding orchestration |
| `sft/` | LoRA fine-tuning notebooks and checkpoints |

### Model Registry

`models/load_model.py` caches loaded models in a dict. Models are loaded on first call to `get_model(name)`:

| Name | Model |
|---|---|
| `"llama"` | `meta-llama/Llama-3.1-8B-Instruct` (main LLM, torch.compile'd) |
| `"koelectra"` | KoELECTRA base (dense embedding, older) |
| `"qwen3_reranker"` | `Qwen/Qwen3-Reranker-0.6B` |
| `"keyword_extractor"` | MeCab POS tagger (Korean nouns: NNP, NNG, SL) |
| `"qwen3_vl"` | `Qwen/Qwen3-VL-Embedding-8B` (current embedding model) |

### Qdrant Collections

- `child_chunks`: Dense + sparse dual vectors, searched during retrieval.
- `parent_chunks`: Full-text context blocks retrieved after reranking by `parent_id`.

### Langfuse Integration

Prompts are versioned and fetched from Langfuse at request time (`apiserver/pompt.py`). All pipeline stages emit traces. Prompt names: `answer`, `query_expansion`, `keyword_extraction`, `like_dobby`, `not_found_context`.

### OpenAI Compatibility

The `/chat/completions` endpoint follows the OpenAI schema, allowing standard OpenAI client libraries to point at this server. Authentication is Bearer token-based (`apiserver/common.py`).

### Session Management

- 세션은 사용자 아이디에 따라서 독립적으로 대화 이력이 관리됨
- `session/` 디렉토리에 세션 관련 기능 구현:
  - `session/session.py`: `get_or_create_session()`, `append_message()`, `get_recent_messages(n=5)`
- MongoDB `chat_sessions` 컬렉션에 저장:
  ```
  {
    "user_id": str,
    "created_at": datetime,
    "messages": [
      {"role": "user|assistant", "content": str, "model": str, "timestamp": datetime}
    ]
  }
  ```
- Slack: `event.get("user")`로 user_id 추출 후 `request_chat(user_id=...)`에 전달
- HTTP (`/rag`, `/chat/completions`): `user_id` 및 `history_limit` 파라미터 지원
- `apiserver/service.py:request_chat()`:
  - `user_id` 있을 때 — 세션에서 최근 N개 메시지 조회
  - 응답 후 — 질문+답변을 세션에 저장
  - `history_limit` 파라미터로 조정 가능 (기본값 5개 대화 = 10개 메시지)
- `apiserver/pompt.py`: `create_answer_prompt(history=None)` 파라미터로 확장 가능