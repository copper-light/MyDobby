from pymongo import DESCENDING

def _insert_one(collection, data):
    collection.insert_one(data)

def insert_raw_document(db, doc_id, doc_source, title, page_path, content_type, text, edit_at, collect_at):
    collection = db['raw_documents']
    _insert_one(collection, {
        "doc_id": doc_id,
        "doc_source": doc_source,
        "title": title,
        "page_path": page_path,
        "content_type": content_type,
        "text": text,
        "edit_at": edit_at,
        "collect_at": collect_at,
    })

def insert_gather_time(db, collect_at, collected_pages):
    collection = db['history_gather_data']
    _insert_one(collection, {
        "process_at": collect_at,
        "collected_pages": collected_pages,
    })

def find_latest_gather_time(db):
    collection = db['history_gather_data']
    latest_collect = collection.find_one(sort=[("process_at", DESCENDING)])
    if latest_collect and "process_at" in latest_collect:
        return latest_collect["process_at"]
    else:
        return None

def insert_process_rag_time(db, process_at, processed_pages):
    collection = db['history_process_rag']
    _insert_one(collection, {
        "process_at": process_at,
        "processed_pages": processed_pages,
    })


def find_raw_documents_for_rag(db):
    pipeline = [
        {
            "$lookup": {
                "from": "history_process_rag",
                "pipeline": [{
                    "$group": {
                        "_id": None,
                        "max_date": {"$max": "$process_at"}
                    }
                }],
                "as": "tmp"
            }
        },
        {
            "$match": {
                "$expr": {
                    "$gt": [
                        "$collect_at",
                        {"$arrayElemAt": ["$tmp.max_date", 0]},
                    ]
                }
            }
        }
    ]
    return list(db['raw_documents'].aggregate(pipeline))


def find_session(db, user_id):
    collection = db['chat_sessions']
    return collection.find_one({"user_id": user_id})


def insert_session(db, user_id, created_at):
    from datetime import datetime
    collection = db['chat_sessions']
    _insert_one(collection, {
        "user_id": user_id,
        "created_at": created_at,
        "messages": []
    })


def push_session_message(db, user_id, message):
    collection = db['chat_sessions']
    collection.update_one({"user_id": user_id}, {"$push": {"messages": message}})


def find_session_messages(db, user_id):
    collection = db['chat_sessions']
    session = collection.find_one({"user_id": user_id})
    if session:
        return session.get("messages", [])
    return []


# def insert_many(collection, data):
#     collection.insert_many(data)

# def find_one(collection, query):
#     return collection.find_one(query)

# def find(collection, query):
#     return collection.find(query)

# def update_one(collection, query, data):
#     collection.update_one(query, data)

# def update_many(collection, query, data):
#     collection.update_many(query, data)

# def delete_one(collection, query):
#     collection.delete_one(query)

# def delete_many(collection, query):
#     collection.delete_many(query)