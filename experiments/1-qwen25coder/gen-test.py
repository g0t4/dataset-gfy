from torch import return_types
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_name = "Qwen/Qwen2.5-Coder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# %%

import numpy as np
test = "roses are red, violets"

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



# %%

# from transformers import pipeline
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# pipe("Explain recursion simply:")

