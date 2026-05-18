import requests
from datetime import datetime
import config.settings as settings

ROOT_BLOCK_IDS = settings.NOTION_ROOT_BLOCK_IDS

def get_data(block_id):
    NOTION_API_URL = f"https://api.notion.com/v1/blocks/{block_id}/children"
    headers = {
        "Authorization": f"Bearer {settings.NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
    }
    response = requests.get(NOTION_API_URL, headers=headers)
    if response.status_code == 200:
        data = response.json()
    else:
        data = None

    return data

def collect_content(data):
    text = ""
    children_page_ids = []
    
    last_edited_time = None
    results = data.get("results", None)
    if results is None: return "", last_edited_time, children_page_ids

    for block in results:
        if last_edited_time is None:
            last_edited_time = datetime.fromisoformat(block["last_edited_time"])
        else:
            new_date = datetime.fromisoformat(block["last_edited_time"])
            if last_edited_time < new_date:
                last_edited_time = new_date

        block_type = block["type"]
        has_children = block.get("has_children", False)

        if block_type == "child_page":
            title = block[block_type]["title"]
            children_page_ids.append((title, block["id"]))
            text += "* " + title + "\n"
        else:

            if block_type.startswith("heading"):
                text += "[HEADER] "
            content = block.get(block_type, {})
            if len(content.get("rich_text", [])) == 0:
                text += "\n"
            else:
                for rich_text in content["rich_text"]:
                    if block_type.find("list_item") != -1:
                        text += "* "
                    text += rich_text["plain_text"] + "\n"

            if has_children:
                data = get_data(block["id"])
                children_text, _, _ = collect_content(data)
                text += children_text + "\n"
        
    return text, last_edited_time, children_page_ids

if __name__ == "__main__":
    from collections import deque

    q = deque(ROOT_BLOCK_IDS)
     
    while q:
        title, block_id = q.popleft()
        data = get_data(block_id)
        if data:
            text, children_blocks = collect_content(data)
            print(text)
            q.extend(children_blocks)