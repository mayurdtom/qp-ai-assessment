from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
from config.model_config import MISTRAL_MODEL_CONFIG

login(token='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')

class MistralService:
    def __init__(self):
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MISTRAL_MODEL_CONFIG['model_name'],
            padding_side="left",
            truncation=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_CONFIG['model_name'],
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    def generate_response(self, context, query):
        prompt = f"""Below is a piece of context information followed by a question. Please analyze the context carefully and provide a specific, relevant answer to the question. If the context doesn't contain enough information to answer the question fully, acknowledge this in your response.

Context: {context}

Question: {query}

Please provide a clear and concise answer based on the above context. Focus on the most relevant information and maintain a natural, helpful tone:"""
        
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MISTRAL_MODEL_CONFIG['max_tokens'],
            temperature=MISTRAL_MODEL_CONFIG['temperature'],
            do_sample=True,
            top_p=0.9,
            top_k=50,
            num_beams=1, 
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        return response