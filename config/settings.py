import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys & Tokens ──────────────────────────────────────────────────────────
SLACK_APP_TOKEN     = os.getenv("SLACK_APP_TOKEN", "")
SLACK_BOT_TOKEN     = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SECRET_KEY    = os.getenv("SLACK_SECRET_KEY", "")
NOTION_TOKEN        = os.getenv("NOTION_TOKEN", "")
API_KEY             = os.getenv("API_KEY", "handh")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

# ── Service URLs ───────────────────────────────────────────────────────────────
QDRANT_URL    = os.getenv("QDRANT_URL",    "http://localhost:6333")
MONGO_URL     = os.getenv("MONGO_URL",     "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "mydobby")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# ── Model Paths ────────────────────────────────────────────────────────────────
LLAMA_MODEL_PATH    = os.getenv("LLAMA_MODEL_PATH",    "sft/dobby-style-lora/final")
QWEN_RERANKER_MODEL = os.getenv("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
QWEN_EMBED_MODEL    = os.getenv("QWEN_EMBED_MODEL",    "Qwen/Qwen3-VL-Embedding-8B")
KOELECTRA_MODEL     = os.getenv("KOELECTRA_MODEL",     "monologg/koelectra-base-v3-discriminator")

# ── Notion ─────────────────────────────────────────────────────────────────────
NOTION_ROOT_BLOCK_IDS = [
    ("기술", "b8ae1e18cc724d7180da1cfd697cff11"),
    ("업무", "2d27cf73b6878013aff4e441251ff928"),
    ("취업", "3257cf73b68780d0b8fefe75ed318098"),
]

# ── Retrieval Tuning ───────────────────────────────────────────────────────────
RETRIEVE_TOPK        = int(os.getenv("RETRIEVE_TOPK",         "30"))
RERANK_TOPK          = int(os.getenv("RERANK_TOPK",           "5"))
SCORE_THRESHOLD      = float(os.getenv("SCORE_THRESHOLD",     "0.15"))
RANK_SCORE_THRESHOLD = float(os.getenv("RANK_SCORE_THRESHOLD","0.1"))

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── BM25 ───────────────────────────────────────────────────────────────────────
BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B  = float(os.getenv("BM25_B",  "0.75"))
