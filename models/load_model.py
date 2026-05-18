from abc import *


ActiveModel = {}

class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def config(self):
        pass
    
    @abstractmethod
    def generate(self, message):
        pass

    @abstractmethod
    def embed(self, message):
        pass

def get_model(model_name = "llama", reload = False):

    if model_name in ActiveModel and not reload:
        return ActiveModel[model_name]
    
    if model_name == "phi":
        from models.phi import PhiModel
        model = PhiModel()
    elif model_name == "llama":
        from models.llama import LlamaModel
        model = LlamaModel()
    elif model_name == "koelectra": # KoElectraModel
        from models.koelectra_embed import KoElectraModel
        model = KoElectraModel()
    elif model_name == "qwen3_reranker": # Qwen3RerankerModel
        from models.qwen3_reranker import Qwen3RerankerModel
        model = Qwen3RerankerModel()
    elif model_name == "keyword_extractor": # KeywordExtractorModel
        from models.keyword_extractor import KeywordExtractorModel
        model = KeywordExtractorModel()
    elif model_name == "qwen3_vl": # Qwen3VLModel
        from models.qwen_embed import Qwen3VLModel
        model = Qwen3VLModel()
    # elif model_name == "gemma":
    #     self.model = GemmaModel()
    else:
        raise ValueError("Invalid model name")

    ActiveModel[model_name] = model

    return model
