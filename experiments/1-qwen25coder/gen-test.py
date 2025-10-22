import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch import return_types
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, pipeline
import rich
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_device = torch.device("cuda:0")
assert torch.cuda.is_available(), "NVIDIA GPU required."

model_name = "Qwen/Qwen2.5-Coder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
assert tokenizer.pad_token is not None

original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype="auto",  # or torch.bfloat16
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" ... LATER
)
original_model.to(use_device)

# %%

test = "roses are red, violets"
# TODO after the fine tune, see how it answers this same prompt... if it says smth unexpected (esp. that rhymes) then I know the fine tune is working!

for i in range(1, 10):
    inputs = tokenizer(test, return_tensors="pt").to(original_model.device)
    inputs.input_ids
    response = original_model(**inputs)
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

original_model.config.use_cache = False  # must be False for gradient checkpointing
original_model.gradient_checkpointing_enable()

# FYI for now, use all examples to verify fine tune is operational... then later split out test/train and reset adapter/model
train_ds = load_dataset("json", data_files="../../out/gfy.nocomments.jsonl")["train"]

def format(sample):
    text = sample["prompt"] + "\n" + sample["completion"]
    tokenized = tokenizer(text, truncation=True, padding=False, max_length=512)
    # augment dataset with labels for training (for causal LM finetune)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized = train_ds.map(format)

# %%
def dump_model_info(when, model):
    for name, param in list(model.named_parameters())[:10]:
        rich.print(f"{when} {name:40} {param.device} {param.dtype}")
dump_model_info("before train", original_model) # shows bfloat16 on my nvidia GPU


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

finetuned_model = get_peft_model(original_model, config)
# dump_model_info("lora before bf16 to", model) # shows float32! for lora layers
finetuned_model.to(torch.bfloat16)
dump_model_info("lora after bf16 to", finetuned_model) # shows bfloat16 now

from transformers import DataCollatorForLanguageModeling

# collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False  # because it's a causal LM, not masked LM
# )

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=finetuned_model,
    args=args,
    train_dataset=tokenized,
    # data_collator=collator,
)

trainer.train()

# %%

# use pipeline to infer
def compare(prompt):
    finetuned_model.eval()

    rich.print("\n\n[bold green]prompt[/]")
    rich.print(prompt)

    rich.print("\n[bold red]untrained[/]")
    untrained = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer, device=0, torch_dtype=torch.bfloat16)
    rich.print(untrained(prompt)[0]["generated_text"])

    rich.print("\n[bold blue]finetuned[/]")
    finetuned = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer, device=0, torch_dtype=torch.bfloat16)
    rich.print(finetuned(prompt)[0]["generated_text"])

compare("Explain why humans have a sense of self:")

# %%

# manual inference (like mine above)

finetuned_model.eval()
prompt = "roses are red, violets"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    for i in range(10):
        outputs = finetuned_model(**inputs)
        next_id = outputs.logits[0, -1].argmax().unsqueeze(0)
        inputs = {
            "input_ids": torch.cat([inputs["input_ids"], next_id.unsqueeze(0)], dim=1),
        }
        test = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        print(test)

