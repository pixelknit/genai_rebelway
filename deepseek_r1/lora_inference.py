from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

base_model = "deepseek-ai/deepseek-llm-7b-base"
finetuned_model = "lora-finetuned-deepseek"

tokenizer = AutoTokenizer.from_pretrained(finetuned_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", load_in_4bit=True, trust_remote_code=True)
model = PeftModel.from_pretrained(model, finetuned_model)

prompt = "Write an amzon review (Product: headphones):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


