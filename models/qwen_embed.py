from models.load_model import BaseModel
import torch
from sentence_transformers import SentenceTransformer

class Qwen3VLModel(BaseModel):
    def __init__(self):
        from config import settings
        self.model = SentenceTransformer(settings.QWEN_EMBED_MODEL)
        self.model.cuda()

    def config(self):
        return self.config
    
    def generate(self, messages, span = None, check_token_cnt = False):
        return None
    
    def embed(self, messages):
        embeddings = self.model.encode(messages, batch_size=32, convert_to_tensor=False, device='cuda')
        return embeddings.tolist()

if __name__ == "__main__":
    model = Qwen3VLModel()

    print(model.embed("안녕하세요"))