import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.load_model import BaseModel

torch.random.manual_seed(0)


class PhiModel(BaseModel):
    def __init__(self, model_path="microsoft/Phi-4-mini-instruct"):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.config = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 10.0,
            "do_sample": False,
        }

    def config(self):
        return self.config
    
    def generate(self, messages, span=None):
        if span:
            span = span.start_observation(as_type="span", name="generate")
            span.update(input=messages)

        prompt = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
            {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
            {"role": "user", "content": messages},
        ]

        result = self.pipe(prompt, **self.config)

        if span:
            span.update(output=result)
            span.end()

        return result