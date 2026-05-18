from datetime import datetime, timezone
from gather_data.notion import get_data, collect_content, ROOT_BLOCK_IDS
from external.connect import get_mongo_db
import external.mongo as mongo 
from collections import deque


def collect_notion():
    db = get_mongo_db()
    collected_count = 0
    start_time = datetime.now(timezone.utc)

    # 한번실행되도록할 것
    # db["collect_time"].create_index([("collect_at", DESCENDING)], name="idx_collect_at_desc")

    latest_collect_time = mongo.find_latest_gather_time(db)
    if latest_collect_time is None:
        latest_collect_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
    else:
        latest_collect_time = latest_collect_time.replace(tzinfo=timezone.utc)

    for content_type, root_block_id in ROOT_BLOCK_IDS:
        q = deque([(content_type, content_type, root_block_id)])
        while q:
            print("remaining.. ", len(q))
            title, page_path, block_id = q.popleft()
            data = get_data(block_id)
            if data:
                text, last_edited_time, children_blocks = collect_content(data)
                if len(text) > 50 and latest_collect_time < last_edited_time:
                    now_utc = datetime.now(timezone.utc)
                    mongo.insert_raw_document(db, block_id, "notion", title, page_path, content_type, text, last_edited_time, now_utc)
                    for (sub_title, block_id) in children_blocks:
                        q.append((sub_title, page_path+" > "+sub_title, block_id))
                    collected_count += 1
                    print(f"find page: {title}, chlidren: {len(children_blocks)}")

    mongo.insert_gather_time(db, start_time, collected_count)

    return collected_count

if __name__ == "__main__": 
    collected_count = collect_notion()
    print(f"Collected {collected_count} documents")