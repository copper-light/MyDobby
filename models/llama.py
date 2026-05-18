import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.load_model import BaseModel


class LlamaModel(BaseModel):
    def __init__(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        from config import settings
        model_id = settings.LLAMA_MODEL_PATH
        # model_id = "meta-llama/Llama-3.1-8B-Instruct"
        # model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            # low_cpu_mem_usage=True
        )
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def config(self):
        return self.config
    
    def generate(self, messages, max_new_tokens: int = 512, top_p: float = 0.95, temperature: float = 0.2, repetition_penalty=1.2):
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
                repetition_penalty=repetition_penalty,
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

    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    # 1. 양자화 설정 정의
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 3. pipeline에 로드된 모델 전달
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    messages_template = [
            {"role": "system", "content": "날씨를 알려주는 기상캐스터야"},
            {"role": "user", "content": "오늘은 날씨 어때?"},
        ]
    results = pipe(messages_template)
    print(results)