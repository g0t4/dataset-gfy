import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch import return_types
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, pipeline
import rich
import torch

use_device = torch.device("cuda:0")
torch.set_default_device(use_device)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_name = "Qwen/Qwen2.5-Coder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # or torch.bfloat16
    trust_remote_code=True
).to(use_device)

# %%

test = "roses are red, violets"
# TODO after the fine tune, see how it answers this same prompt... if it says smth unexpected (esp. that rhymes) then I know the fine tune is working!

for i in range(1, 10):
    inputs = tokenizer(test, return_tensors="pt")
    inputs.input_ids
    response = model(**inputs)
    logits = response.logits
    logits.shape
    last = logits[0,-1:][0]
    max_token_id_next = last.argmax()
    next = tokenizer.decode(max_token_id_next)
    test = test + next
    print(test)
    # print(next)


# %%

# untrained = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt_recursion = "Explain recursion simply:"
# rich.print(untrained(prompt_recursion)[0]["generated_text"])

# %%


# FYI for now, use all examples to verify fine tune is operational... then later split out test/train and reset adapter/model
ds = load_dataset("json", data_files="../../out/gfy.nocomments.jsonl")["train"]

def format(sample):
    text = sample["prompt"] + "\n" + sample["completion"]
    tokenized = tokenizer(text, truncation=True, max_length=512)
    # augment dataset with labels for training (for causal LM finetune)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized = ds.map(format)

# %%
def dump_model_info(when, model):
    for name, param in list(model.named_parameters())[:10]:
        rich.print(f"{when} {name:40} {param.device} {param.dtype}")
dump_model_info("before train", model) # shows bfloat16 on my nvidia GPU


# %%

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# %%

model = get_peft_model(model, config)


args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-5,
    num_train_epochs=10,
    logging_steps=10,
    bf16=True
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)



trainer.train()

# %%

def compare(prompt):
    untrained = pipeline("text-generation", model=model, tokenizer=tokenizer)
    finetuned = pipeline("text-generation", model=model, tokenizer=tokenizer)
    rich.print("\n[bold green]prompt[/]")
    rich.print(prompt)
    rich.print("\n[bold red]untrained[/]")
    rich.print(untrained(prompt)[0]["generated_text"])
    rich.print("\n[bold blue]finetuned[/]")
    rich.print(finetuned(prompt)[0]["generated_text"])

compare("Explain why humans have a sense of self:")
