from models.load_model import BaseModel
from mecab import MeCab

class KeywordExtractorModel(BaseModel):
    def __init__(self):
        self.model = MeCab(user_dictionary_path="retrieve/nouns.dic")


    def config(self):
        pass

    def generate(self, message):
        tokens = self.model.pos(message.lower())
        ret = []
        for word, t in tokens:
            if t in ["NNP", "NNG", "SL"]:
                ret.append(word)
        return ret

    def embed(self, message):
        pass
        