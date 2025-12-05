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
        self.ln_f = base_model.model.final_layernorm
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
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = GPTOSSWithHooks(base).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# %% 3. Capture residual activations (identical to TL flow)
def resid_filter(name):
    return "resid_" in name

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

