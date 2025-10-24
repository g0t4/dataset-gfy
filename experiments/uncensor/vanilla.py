import cuda_env

cuda_env.use_6000()
# cuda_env.list_gpus()

# %%
"""### Load with vanilla transformers - test this works first"""
MODEL_PATH_CHAT = 'Qwen/Qwen-1_8B-chat'
MODEL_PATH_BASE = 'Qwen/Qwen-1_8B'
DEVICE = 'cuda:0'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

chat_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_CHAT, trust_remote_code=True)
chat_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_CHAT, trust_remote_code=True,
    # fp16=True, # TODO! figure out what is native for this model (IIUC its bf16)
)
chat_model.to(DEVICE)
chat_model.eval()
# ImportError: cannot import name 'BeamSearchScorer' from 'transformers' (.venv/lib/python3.12/site-packages/transformers/__init__.py)
#  apparently this was removed in transformers... so I need to use a different

# %%


def manual_inference(model, tokenizer, text, max_tokens=10):
    for _ in range(max_tokens):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        inputs.input_ids
        response = model(**inputs)
        logits = response.logits
        logits.shape
        last = logits[0,-1:][0]
        max_token_id_next = last.argmax()
        #             "input_ids": torch.cat([inputs["input_ids"], next_id.unsqueeze(0)], dim=1),
        #             TODO switch to not re-encode all tokens on every iteration
        next = tokenizer.decode(max_token_id_next, skip_special_tokens=True)
        text = text + next
        print(f'{next=}')

        # check EOS
    print(text)

# %%

base_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_BASE, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_BASE, trust_remote_code=True)
base_model.to(DEVICE)
base_model.eval()

# %%

manual_inference(base_model, base_tokenizer, "foo the")

# %%

chat_model.chat(query="what is a cuck?", history=[], tokenizer=chat_tokenizer)

# %%

# test manual chat prompt

# TEMPLATE from qwen repo
#
# {% for message in messages %}
# {% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}
# {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
# {% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
# {% endfor %}

def make_prompt(query):
    return f"""<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""


query = "name one of the most popular programming languages... just give me the name"
manual_inference(chat_model, chat_tokenizer, make_prompt(query), max_tokens=100)

# %%

# use base unless using chat template:
pipe = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer, device=DEVICE)
response = pipe("test")

print(response)
