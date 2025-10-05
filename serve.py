from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

load_dotenv()
# -------------------------------
# Define Allowed Routes
# -------------------------------
SERVER_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

origins = [
    "http://localhost:3000",
    SERVER_API_URL,
    "http://localhost:3001",
]





# -------------------------------
# Define request model
# -------------------------------
class JobRequest(BaseModel):
    job_title: str
    skills: str
    max_new_tokens: int = 300  # optional, default 300
    num_beams: int = 3         # optional, default 3

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(title="EchoHire QGen API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all origins (not recommended in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------
# Load model & tokenizer
# -------------------------------
MODEL_PATH = "./models/echohire-qgen-distilgpt2"  # inside container
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu")

# Optional: set model to eval mode for inference
model.eval()

# -------------------------------
# Helper function to generate questions
# -------------------------------
def generate_questions(prompt: str, max_new_tokens=300, num_beams=3):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# -------------------------------
# GET endpoint
# -------------------------------
@app.get("/generate-questions")
def generate(
    job_title: str = Query(..., description="Job title for the question generation"),
    skills: str = Query(..., description="Skills for the question generation"),
    max_new_tokens: int = Query(300, description="Max new tokens to generate"),
    num_beams: int = Query(3, description="Number of beams for generation")
):
    prompt = f"Job Title: {job_title}\nSkills: {skills}\nQuestions Only. No Answers:"
    questions = generate_questions(prompt, max_new_tokens, num_beams)
    return {"questions": questions}