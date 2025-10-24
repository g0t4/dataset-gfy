import cuda_env

cuda_env.use_6000()
# cuda_env.list_gpus()

# %%
"""### Load with vanilla transformers - test this works first"""
# MODEL_PATH = 'Qwen/Qwen-1_8B-chat'
MODEL_PATH = 'Qwen/Qwen-1_8B'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# PRN test with cuda device not using device_map="auto"?
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()
# ImportError: cannot import name 'BeamSearchScorer' from 'transformers' (.venv/lib/python3.12/site-packages/transformers/__init__.py)
#  apparently this was removed in transformers... so I need to use a different

# %%

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, dtype=torch.bfloat16)
response = pipe("test")

print(response)
