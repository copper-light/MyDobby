import torch
from sentence_transformers import CrossEncoder
from models.load_model import BaseModel

class Qwen3RerankerModel(BaseModel):
    def __init__(self):
        from config import settings
        self.model = CrossEncoder(settings.QWEN_RERANKER_MODEL)
        self.model.cuda()

    def config(self):
        return self.config
    
    def generate(self, messages, span = None, check_token_cnt = False):
        return None
    
    def embed(self, messages):
        return None

    def rerank(self, query, documents, topk=100, threshold = 0):
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, batch_size=32, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
        # print(scores)
        indexes = scores.argsort()[-topk:]

        ret = []
        for i in indexes:
            if threshold <= scores[i]:
                ret.append((i, scores[i]))

        return ret


if __name__ == "__main__":
    reranker = Qwen3RerankerModel()
    documents = ["hello", "test", "world"]
    query = "test"
    print(reranker.rerank(query, documents, 2))
    