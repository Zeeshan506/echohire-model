from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class Predictor(BasePredictor):
    def setup(self):
        # Use Hugging Face Hub model instead of local path
        model_name = "Zeeshan506/echohire-qgen-distilgpt2"  # <-- HF repo
        # If private, uncomment the next line and add your token to .env
        # os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_API_TOKEN")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, prompt: str = Input(..., description="Input prompt")) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_length=512,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
