from models.load_model import BaseModel
import torch
from transformers import ElectraModel, ElectraTokenizer


class KoElectraModel(BaseModel):
    def __init__(self):
        from config import settings
        self.model = ElectraModel.from_pretrained(settings.KOELECTRA_MODEL)
        self.tokenizer = ElectraTokenizer.from_pretrained(settings.KOELECTRA_MODEL)

    def config(self):
        return self.config
    
    def generate(self, messages, span = None, check_token_cnt = False):
        return None
    
    def embed(self, messages):
        with torch.no_grad():
            ids = self.tokenizer.encode(messages)
            embeddings = self.model(torch.tensor(ids).unsqueeze(0))
            embeddings = torch.mean(embeddings.last_hidden_state, dim=1).squeeze(0)

        return embeddings.tolist()


if __name__ == "__main__":
    model = KoElectraModel()

    import pandas as pd
    from apiserver.pompt import create_extract_keywords_prompt

    # df = pd.read_csv("eval_01.csv")
    print(model.embed("안녕하세요"))