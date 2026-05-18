from langfuse._client import span_processor
from langfuse._client import span_processor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.load_model import BaseModel


class LlamaModel(BaseModel):
    def __init__(self):

        # quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="fp4",  # 또는 "fp4"
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        # )

        # model_id = "sft/dobby-style-lora/final"
        model_id = "unsloth/gemma-4-26B-A4B-it-GGUF"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
        # self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def config(self):
        return self.config
    
    def generate(self, messages, max_new_tokens: int = 1024, top_p: float = 0.95, temperature: float = 0.2, repetition_penalty=1.2):
        with torch.no_grad():
            inputs = self.tokenizer(messages, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            input_token_len = len(inputs.input_ids[0])

            outputs = self.model.generate(
                **inputs,
                # eos_token_id=[
                #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # ],
                pad_token_id = self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                top_p = top_p,
                repetition_penalty=1.2,
                max_new_tokens=max_new_tokens, 
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                # output_scores=True,
                use_cache=True
            )

            output_ids = outputs.sequences[0][input_token_len:]
            output_token_len = len(output_ids)
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return output_text.strip("\n"), input_token_len, output_token_len 

    # def batch_generate(self, messages):
    #     messages_template = [
    #         {"role": "user", "content": messages},
    #     ]
    #     results = self.pipe(messages_template, batch_size=4)
        
    #     return results

    def embed(self, messages):
        return None

    def rerank(self, query, documents, topk=5):
        return None


if __name__ == "__main__":
    # model = LlamaModel()
    # print(model.generate("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n날씨를 알려주는 기상캐스터야\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n오늘은 날씨 어때?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", top_p=0.9, temperature=0.1))

    # from transformers import pipeline
    # pipe = pipeline("text-generation", max_new_tokens = 1024, top_p = 0.9, temperature = 0.1, model="meta-llama/Llama-3.1-8B-Instruct")

    # messages_template = [
    #         {"role": "system", "content": "날씨를 알려주는 기상캐스터야"},
    #         {"role": "user", "content": "오늘은 날씨 어때?"},
    #     ]
    # results = pipe(messages_template)
    # print(results)


    from transformers import AutoProcessor, AutoModelForCausalLM
       
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]

    model_id = "google/gemma-4-26B-A4B-it"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

    # Process input
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse output
    print(processor.parse_response(response))