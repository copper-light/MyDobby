from datetime import datetime, timezone
from models.load_model import get_model
from tqdm import tqdm
from external.connect import get_qdrant_client, get_mongo_db, get_langfuse_client
from external.mongo import find_raw_documents_for_rag, insert_process_rag_time
from external.qdrant import insert_parent_chunk, insert_child_chunk, reset_collection
from apiserver.pompt import create_extract_keywords_prompt
import config.settings as settings
import re


def get_documents():
    mongo_db = get_mongo_db()
    documents = find_raw_documents_for_rag(mongo_db)
    print("loaded documents: ", len(documents))
    return documents

def extract_keyword(text):
    ex_model = get_model("keyword_extractor")
    langfuse = get_langfuse_client()
    keywords = ex_model.generate(text)
    # keywords = [w.strip() for w in keywords.split(",") if len(w.strip()) > 1]
    keywords = " ".join(keywords)
    # keywords = keywords.lower().replace(",", " ").replace(".", " ").replace("-", " ").replace("_", " ")
    return keywords


def chunk_to_children(title, text, chunk_size = settings.CHUNK_SIZE, overlap = settings.CHUNK_OVERLAP, max_overlap= 150):
    sentences = re.split(r"(?<!\w.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s|\n", text)
    chunks = []
    current_chunk = ""
    overlap_chunk = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = re.sub(r'^\* ', '', sentence)
        if len(sentence) == 0:
            continue

        overlap_chunk.append(sentence)
        while len(overlap_chunk) > 1 and len(" ".join(overlap_chunk)) > overlap:
            overlap_chunk.pop(0)
            
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += (sentence + "\n")
        else:
            # overlap_chunk_str = " ".join(overlap_chunk)
            chunks.append(current_chunk.strip())
            overlap_text = "\n".join(overlap_chunk)
            if len(overlap_text) > max_overlap:
                overlap_text = overlap_text[:max_overlap]

            current_chunk = overlap_text + "\n" + sentence # overlap_chunk_str

    if len(current_chunk) > 0:
        chunks.append(current_chunk.strip())

    return chunks


def preprocess_documents(chunk):
    chunk = re.sub(r'[ \t]+', ' ', chunk.strip())
    chunk = re.sub(r'\n+', '\n', chunk)

    return chunk.strip()


def parent_chunking(document):
    chunks = document.strip().split("[HEADER]")

    ret = []

    # for i in range(len(chunks)):
    #     ret += chunks[i].split("\n\n")

    for i, c in enumerate(chunks):
        # first element is header
        chunks[i] = preprocess_documents(c)
    return chunks


def process_chunking(chunk_size = settings.CHUNK_SIZE, overlap = settings.CHUNK_OVERLAP):
    start_time = datetime.now(timezone.utc)

    client = get_qdrant_client()
    db = get_mongo_db()
    documents = get_documents()
    eb_model = get_model("koelectra")
    # eb_model = get_model("qwen3_vl")
    langfuse = get_langfuse_client()

    processed_pages = 0
    processed_chunks = 0

    for document in tqdm(documents):
        doc_id = document["doc_id"]
        title = document["title"]
        page_path = document["page_path"]
        p_chunks = parent_chunking(document["text"])

        for i, p_chunk in enumerate(p_chunks):
            if len(p_chunk) <= 0: continue

            p_id = f"{doc_id}_{i+1}_{len(p_chunks)}"
            insert_parent_chunk(client, doc_id, p_id, title, page_path + "\n" + p_chunk, doc_path=page_path)

            children_chunks = chunk_to_children(title, p_chunk, chunk_size=chunk_size, overlap= overlap)
            for j, c in enumerate(children_chunks):
                embeddings = eb_model.embed(page_path + ". " + c)
                keywords = extract_keyword(c + " " + page_path)
                insert_child_chunk(
                    client,
                    embeddings,
                    keywords,
                    doc_id,
                    p_id,
                    f"{p_id}_{j+1}_{len(children_chunks)}",
                    title,
                    c)
                processed_chunks += 1
        processed_pages += 1

    insert_process_rag_time(db, start_time, processed_pages)
    print(f"Processed {processed_pages} pages, chunks: {processed_chunks}")

if __name__ == "__main__":
    reset_collection(get_qdrant_client())
    process_chunking(chunk_size=256, overlap=50)