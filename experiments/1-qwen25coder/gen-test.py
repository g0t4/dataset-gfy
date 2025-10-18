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
test = "roses are red, violets are "
inputs = tokenizer(test, return_tensors="pt")
inputs.input_ids
inputs

response = model(**inputs)

response

logits = response.logits

logits
logits.shape




# %%



# %%

from transformers import pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe("Explain recursion simply:")

