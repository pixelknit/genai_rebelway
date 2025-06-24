from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
print("Data loaded")

model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
        )

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
        )

model = get_peft_model(model, lora_config)

def tokenize(sample):
    prompt = sample["prompt"]
    completion = sample["completion"]
    full_text = prompt + " " + completion
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, remove_columns=["prompt", "completion"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#training args

training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_dir="./logs",
        save_strategy="epoch",
        report_to="none"
        )

#trainer
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
        )

trainer.train()

model.save_pretrained("lora-finetuned-deepseek")
tokenizer.save_pretrained("lora-finetuned-deepseek")
