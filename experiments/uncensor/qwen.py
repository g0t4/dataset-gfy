from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    trust_remote_code=True,
).eval()

# %%

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Explain how attention works."
    },
]

prompt_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

print(tokenizer.eos_token_id)
print(tokenizer.decode(tokenizer.eos_token_id))

outputs = model.generate(
    prompt_ids,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
