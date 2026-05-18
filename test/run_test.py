from scipy.stats import describe
from external.connect import get_langfuse_client

import json
from tqdm import tqdm

if __name__ == '__main__':
    with open("test/result_test.json") as f:
        o = json.load(f)
    
    client = get_langfuse_client()
  
    o = o["result"]
    for raw in tqdm(o, desc="Validation"):
        try:
            prompt_obj = client.get_prompt("test01")
            with client.start_as_current_observation(as_type="generation", name="validation", input=raw["query"], prompt=prompt_obj) as trace:
                trace.update(output=raw, metadata ={"rag": True,})
        except Exception as e:
            print(e)
        client.flush()

        