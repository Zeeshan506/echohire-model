import time
from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_path = r"D:\Echohire\model training\model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)  # CPU only

job_title = "Cybersecurity Engineer"
skills = "Endpoint Security, Docker, Cloud Security, DevSecOps, Security Automation, Vulnerability Management"

prompt = f"Job Title: {job_title}\nSkills: {skills}\nQuestions Only. No Answers"

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = (input_ids != tokenizer.pad_token_id).long()

start_time = time.time()  # start timer

output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=250,
    num_beams=3,
    no_repeat_ngram_size=2
)

end_time = time.time()  # end timer

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
print("\nTime taken (seconds):", end_time - start_time)
