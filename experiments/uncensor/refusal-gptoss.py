import torch
import contextlib
from dataclasses import dataclass

@dataclass
class HookPoint:
    name: str

class GPTOSSWithHooks(torch.nn.Module):
    """
    A minimal HookedTransformer-style wrapper for GPT-OSS.
    Provides:
      - run_with_cache
      - hook registration
      - unified hook naming
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.cfg = base_model.config
        self.layers = base_model.model.layers
        self.emb = base_model.model.embed_tokens
        self.ln_f = base_model.model.norm  # todo rename? there was no model.final_layernorm... it does have model.norm
        self.lm_head = base_model.lm_head

        # External hook registry
        self._fwd_hooks = {}
        self._temp_fwd_handles = []

    # -----------------------------
    # Hook utilities
    # -----------------------------
    def add_hook(self, name, fn):
        if name not in self._fwd_hooks:
            self._fwd_hooks[name] = []
        self._fwd_hooks[name].append(fn)

    @contextlib.contextmanager
    def hooks(self, fwd_hooks=None):
        """Temporarily install hooks like HookedTransformer.hooks()."""
        if fwd_hooks:
            for (name, fn) in fwd_hooks:
                self.add_hook(name, fn)
            try:
                yield
            finally:
                # Clear temporary registry
                self._fwd_hooks.clear()
        else:
            yield

    def _apply_hooks(self, name, output):
        """Apply all hooks for a given hook point. Hooks may *modify* output."""
        if name not in self._fwd_hooks:
            return output
        for fn in self._fwd_hooks[name]:
            output = fn(output, HookPoint(name))
        return output

    # -----------------------------
    # Run with caching
    # -----------------------------
    @torch.no_grad()
    def run_with_cache(self, input_ids, names_filter=None):
        """
        Runs the model and returns:
          logits, cache
        cache is a dict[str → tensor] for hook points.
        """
        cache = {}

        def save(name, tensor):
            if (names_filter is None) or (names_filter(name)):
                cache[name] = tensor.detach().clone()

        hidden_states = self.emb(input_ids)

        for i, layer in enumerate(self.layers):
            # resid_pre
            name = f"resid_pre/{i}"
            save(name, hidden_states)
            hidden_states = self._apply_hooks(name, hidden_states)

            # attention
            normed = layer.input_layernorm(hidden_states)
            attn_out = layer.self_attn(normed)[0]

            # resid_mid (after attn write)
            name = f"resid_mid/{i}"
            post_attn = hidden_states + attn_out
            save(name, post_attn)
            post_attn = self._apply_hooks(name, post_attn)

            # MLP
            normed2 = layer.post_attention_layernorm(post_attn)
            mlp_out = layer.mlp(normed2)

            hidden_states = post_attn + mlp_out

            # resid_post
            name = f"resid_post/{i}"
            save(name, hidden_states)
            hidden_states = self._apply_hooks(name, hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, cache

    # -----------------------------
    # Direct forward (no caching)
    # -----------------------------
    def forward(self, input_ids):
        return self.model(input_ids).logits

# %% 2. Load GPT-OSS with this wrapper
#
from transformers import AutoModelForCausalLM, AutoTokenizer

GPTOSS_20B = 'openai/gpt-oss-20b'
GPTOSS_120B = 'openai/gpt-oss-120b'

MODEL_PATH = GPTOSS_20B
DEVICE = "cuda"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # torch_dtype=torch.bfloat16,
    device_map="cuda")

model = GPTOSSWithHooks(base).to(DEVICE)  # broken final_layernorm

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# %% test base model
#
# test_query = "what is your name?"
# inputs = tokenizer(test_query, return_tensors="pt").to(base.device)
# out = base.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(out[0], skip_special_tokens=True))

# %% test hooked model (pre-hooks)

test_query = "what is your name?"
inputs = tokenizer(test_query, return_tensors="pt").to(base.device)
token_ids = inputs.input_ids
for i in range(0, 30):
    out = model.forward(token_ids)
    print(out.shape)
    next_token_id = out[0][-1].argmax()
    print(next_token_id.item(), tokenizer.decode(next_token_id))
    token_ids = torch.cat([token_ids, next_token_id[None, None]], dim=1)
    print(token_ids)
    print("-" * 80)
    print(tokenizer.decode(token_ids[0], skip_special_tokens=True))
    # if next_token_id == 2:
    #     break

print("""

    TODO STOPPED HERE... just spent some time evaluating if the base/model models work to generate text, they do...
    now I need to evaluate how the hooks work (ChatGPT generated this based on my qwen version that uses transformer_lens given TL doesn't work with gptoss yet)
    though now that I checked:  https://github.com/TransformerLensOrg/TransformerLens/releases/tag/v3.0.0a5
    it is coming in v3!

    """)

# %% 3. Capture residual activations (identical to TL flow)

def resid_filter(name):
    return "resid_" in name

# TypeError: GptOssAttention.forward() missing 2 required positional arguments: 'position_embeddings' and 'attention_mask'
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(DEVICE)

logits, cache = model.run_with_cache(input_ids, names_filter=resid_filter)

for k, v in cache.items():
    print(k, v.shape)

# %% 4. Direction extraction (no change from your Qwen code)
harmful_mean = cache_harm['resid_pre/14'][:, -1, :].mean(0)
harmless_mean = cache_safe['resid_pre/14'][:, -1, :].mean(0)

refusal_dir = harmful_mean - harmless_mean
unit_refusal_dir = refusal_dir / refusal_dir.norm()

# %% 5. Ablation hooks (identical pattern to TL)
def remove_direction(direction):
    direction = direction / direction.norm()

    def hook(out, hookpoint: HookPoint):
        proj = (out @ direction) * direction
        return out - proj

    return hook

hooks = []
for layer in range(model.cfg.num_hidden_layers):
    for name in ["resid_pre", "resid_mid", "resid_post"]:
        hooks.append((f"{name}/{layer}", remove_direction(unit_refusal_dir)))
with model.hooks(hooks):
    logits, _ = model.run_with_cache(input_ids)

# %% 6. Weight orthogonalization (fully supported)
def orthogonalize_weight(W, v):
    proj = (W @ v)[:, None] * v[None, :]
    return W - proj

for layer in model.layers:
    layer.self_attn.o_proj.weight.data = orthogonalize_weight(
        layer.self_attn.o_proj.weight.data,
        unit_refusal_dir,
    )

    # mlp: down_proj writes to residual
    layer.mlp.down_proj.weight.data = orthogonalize_weight(
        layer.mlp.down_proj.weight.data,
        unit_refusal_dir,
    )
