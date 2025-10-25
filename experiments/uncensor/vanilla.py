import cuda_env
from rich import print

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
chat_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_CHAT,
    trust_remote_code=True,
    # fp16=True, # TODO! figure out what is native for this model (IIUC its bf16)
)
chat_model.to(DEVICE)
chat_model.eval()
# ImportError: cannot import name 'BeamSearchScorer' from 'transformers' (.venv/lib/python3.12/site-packages/transformers/__init__.py)
#  apparently this was removed in transformers... so I need to use a different

# %%

base_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_BASE, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_BASE, trust_remote_code=True)
base_model.to(DEVICE)
base_model.eval()

# %%

def summarize_layer(name, param):
    print(f"{name} shape={tuple(param.shape)} {param.dtype}")

for name, param in chat_model.named_parameters():
    # if "wte" in name:
    summarize_layer(name, param)

# for m in chat_model.modules():
#     print("---")
#     print(m)

# %% * inspect hiddens w/o transformer_lens HookedTransformer

def inspect_hiddens_in_forward_pass(tokenizer, model, text, max_tokens=1):
    for _ in range(max_tokens):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print("inputs", inputs)
        summarize_layer("inputs.input_ids", inputs.input_ids)
        response = model(**inputs, output_hidden_states=True)
        # TODO look into what I can get at with just output_hidden_states and not Transformer Lens
        #   IIUC won't get names here
        #   also obviously can't modify, only inspect
        logits = response.logits
        summarize_layer("logits", logits)
        logits.shape
        last = logits[0, -1:][0]
        max_token_id_next = last.argmax()
        print("max_token_id_next", max_token_id_next)
        if max_token_id_next.item() == tokenizer.eos_token_id:
            break
        #             "input_ids": torch.cat([inputs["input_ids"], next_id.unsqueeze(0)], dim=1),
        #             TODO switch to not re-encode all tokens on every iteration
        next = tokenizer.decode(max_token_id_next, skip_special_tokens=True)
        text = text + next


        # ** recompute next token off final hidden state + weights to map hidden space => token space
        # print("hidden states:", response.hidden_states)
        for number, hidden in enumerate(response.hidden_states):
            summarize_layer(f"hidden: layer {number}", hidden)
        # TODO take final hidden layer and matmul wte.T to re-create the logits and see if they line up
        print("REDO LOGITS off last hidden:")
        last_hidden = response.hidden_states[-1]
        # last_hidden = last_hidden.squeeze()[-1]
        # wte = model.transformer.wte.weight # this must just be for input side (does not match)
        # summarize_layer("  wte", wte)
        weights_lm_head = model.lm_head.weight  # this results in a match!
        summarize_layer("  weights_lm_head", weights_lm_head)
        #    FYI dump named_parameters (layers) and that's how I found lm_head at end with the right dimension
        last_tok_last_hidden = last_hidden[:, -1:, :]
        summarize_layer("  last_tok_last_hidden", last_tok_last_hidden)
        hidden_vector_last_token = last_tok_last_hidden.squeeze()
        summarize_layer("  hidden_vector_last_token", hidden_vector_last_token)  # remove both 1-D dimensions
        redo_logits = hidden_vector_last_token.matmul(weights_lm_head.T)
        # redo_logits = weights_lm_head.matmul(hidden_vector_last_token) # s/b same result
        summarize_layer("  redo_logits", redo_logits)
        redo_max_token_id_next = redo_logits.argmax()
        print("  redo max_token_id_next:", redo_max_token_id_next)
        redo_decoded = tokenizer.decode(redo_max_token_id_next, skip_special_tokens=True)
        print("  redo decoded: '" + redo_decoded + "'")

    print(text)
    return text

inspect_hiddens_in_forward_pass(base_tokenizer, base_model, "foo the")
# FYI base model => " bar" is next token
# %%

# found stop token here (151645)
#   https://github.com/QwenLM/Qwen/blob/b5529b8/recipes/inference/vllm/README.md#L169
#   not finding it anywhere in huggingface repo's regular files?!
chat_tokenizer.eos_token_id = 151645
chat_tokenizer.pad_token_id = 151645

base_tokenizer.eos_token_id = 151645
base_tokenizer.pad_token_id = 151645

# %%

def manual_inference(model, tokenizer, text, max_tokens=10):
    for _ in range(max_tokens):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        inputs.input_ids
        response = model(**inputs)
        logits = response.logits
        logits.shape
        last = logits[0, -1:][0]
        max_token_id_next = last.argmax()
        if max_token_id_next.item() == tokenizer.eos_token_id:
            break
        #             "input_ids": torch.cat([inputs["input_ids"], next_id.unsqueeze(0)], dim=1),
        #             TODO switch to not re-encode all tokens on every iteration
        next = tokenizer.decode(max_token_id_next, skip_special_tokens=True)
        text = text + next
        # print(f'{next=}')

    print(text)

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

# %%

dir(base_model)
type(base_model)

# %%
from rich import print

for n in base_model.parameters():
    print(n)

# %%

def summarize_named_params(model):
    for name, param in model.named_parameters():
        summarize_layer(name, param)

summarize_named_params(base_model)
