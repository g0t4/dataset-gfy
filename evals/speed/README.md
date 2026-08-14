# Speed eval

Raw speed testing: how fast does a given model/hardware/llama-server-flags
combination actually run, on real prompts pulled from real usage. The goal
isn't correctness (that's what `../fim`, `../rewrite`, `../agent` are for)
-- it's finding good params for my actual workloads instead of hand-tuning
`--spec-draft-n-max` one value at a time and eyeballing tokens/sec. This
matters more now that MTP (multi-token prediction / self-speculative
decoding) ships baked into most of the local models I run -- there's a real
parameter space to sweep, not just "on vs off."

Same trace format as the other tracks (`request_body.messages`/`tools`),
same "prompt = everything up to the first assistant message" convention as
`../agent`. No grading at all -- a case's completion is captured and shown
for sanity-checking, but the only thing that matters is llama-server's own
`timings` block (via `extra_body={"verbose": True}`, same mechanism
`../fim/run_eval.py`'s `stream_completion` uses) plus client-observed
wall-clock time-to-first-token and total latency.

## What's different about this track

Unlike fim/rewrite/agent, this one manages a server's *lifecycle*, not just
its API: `run_eval.py` SSHes into a host, checks GPU state, optionally kills
any already-running `llama-server`, launches a new one in the background
with whatever flags you're sweeping, waits for it to come up, runs the
cases, downloads its log, then tears it down again. Plain `ssh`/`scp`
subprocess calls, no paramiko/fabric -- `~/.ssh/config` already has host
aliases (`build21`, `paxy.lan`) with `ControlMaster`/`ControlPersist`, so
repeat ssh calls in one run reuse a single connection for free.

**Safety default:** it will never kill someone else's already-running
`llama-server` unless you pass `--kill-existing` explicitly. The GPU box
this was built against (`build21`) is routinely mid-session with a real
model loaded for manual use -- the default behavior is to print what it
found and start the new server alongside it (on GPU capacity that's left
over), not clear the board.

## Two categories, two axes

- **`generation`**: short prompt, long reference completion. Prompt
  processing finishes almost instantly, so total time is dominated by
  decode speed -- and, with speculative decoding on, by draft-accept rate.
  `speed-gen-lua-tower-of-hanoi` is this: a 217-token prompt, ~1824 tokens
  of reference output.
- **`prefill`**: huge prompt, near-empty reference completion. Isolates
  prompt-processing throughput almost by itself.
  `speed-prefill-agent-semantic-grep-what-time-is-it` is this: a real
  ~11.3K-token semantic-grep RAG dump under a trivial "what time is it?"
  question, answered with a single tool call.

Mixing both concerns in one case just averages two different numbers
together and tells you less about either. When adding a case, decide which
axis it's meant to isolate and lean into that -- don't reach for a "medium"
prompt with a "medium" completion; the extremes are what make a number
legible.

## Cases and what to think about when adding one

- **Cold cache is load-bearing for `prefill` cases.** llama-server caches
  the KV state of a prompt it's already seen (`cache_n` in the timings
  block). Run a `prefill` case against a server that's already processed
  that exact prompt (e.g. a second `--only` run right after the first) and
  you'll measure cache-hit speed, not real prefill throughput -- an
  order-of-magnitude different number. Fresh server per prefill sweep, or
  at least a prompt that's actually novel to that server, or you're not
  measuring what you think you're measuring.
- **Decide which axis before picking a trace**, per above -- check
  `prompt_n` and `predicted_n` in the source trace's own captured timings
  (or estimate from `len(content)` if the trace predates verbose capture)
  before committing to a case.
- **`agent/` and `fim/`/`rewrite/` traces are fair game as source
  material here too** -- an agent trace especially can have a *lot* of
  accumulated context by its later turns (some in this repo top 200K+
  tokens by the last turn), which is a great prefill stress test even
  though that trace was never meant to become a correctness eval. Use
  `prompt_end_idx` to pick any point in a longer trace, not just the
  first-assistant-message default.
- **This eval only exists to serve real usage patterns** -- the traces
  worth turning into cases are the shapes of prompt you actually send
  (a bare question, a heavy-RAG agent turn, a FIM with a big surrounding
  file, ...), not synthetic worst-cases. If a param only matters for a
  prompt shape you never actually send, it's not worth sweeping.
- **A case's `notes` field should state which real timings the source
  trace captured**, if any (prompt_n/predicted_n/prompt_per_second/etc) --
  useful as a sanity baseline for "does this number look like the same
  ballpark as when this was captured for real."
- Not built yet, worth thinking about before it's needed: concurrent
  requests (server-side batching behavior under load, not just single-
  stream throughput), and CPU contention -- if something else on the box is
  compiling/indexing/downloading while a sweep runs, decode/prefill numbers
  will be noisy for reasons that have nothing to do with the params being
  swept. `check_gpus()` only looks at GPU state today; worth adding a CPU
  load check (or just a printed warning) if noisy runs turn out to be a
  real problem in practice.

## Speculative decoding / MTP params worth sweeping

llama-server's speculative-decoding flags apply uniformly whether the
"draft" comes from a separate small model (`--spec-draft-model`) or a
built-in MTP head baked into the same GGUF (`--spec-type draft-mtp`, no
draft model needed -- confirmed against a captured trace's
`generation_settings.speculative.types: "none,draft-mtp"`). Valid
`--spec-type` values (from `llama.cpp`'s server README): `none`,
`draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`,
`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`.

First axis to explore, per your framing: **MTP on/off, and `n-max` swept
across a range** (e.g. `none` as baseline, then `draft-mtp` at
`--spec-draft-n-max 2,3,4,...16`) -- that alone should surface where
diminishing returns or actual regressions kick in for a given model.

Other flags in the same family worth varying once `n-max` is mapped out:

- `--spec-draft-n-min` (default 0) -- minimum draft tokens per step; a
  floor under `n-max`, only matters combined with it.
- `--spec-draft-p-min` (default 0.00) -- minimum acceptance probability
  (greedy) for a draft token to count; raising it should trade acceptance
  rate for draft quality.
- `--spec-draft-p-split` (default 0.10) -- speculative decoding split
  probability; less intuitive, worth a coarse sweep (e.g. 0.05/0.10/0.20)
  once n-max/n-min look stable.
- `--spec-type` **itself**, across whichever of `draft-eagle3`/`draft-mtp`/
  `draft-dspark` a given model actually ships weights/heads for -- these are
  different draft mechanisms, not just different tuning of the same one, so
  comparing them is a separate question from tuning one of them.

## Dev-loop model

Use a small, fast model (e.g. `ggml-org/Qwen3.5-0.8B-GGUF`) while iterating
on the harness itself -- the point is fast turnaround while finding bugs in
`run_eval.py`, not a meaningful number. Real sweeps (bigger models, the full
`n-max` range, multiple `--spec-*` combinations) come after the harness is
trusted.

```
uv run --project .. python run_eval.py --host build21 --port 8097 \
    --model ggml-org/Qwen3.5-0.8B-GGUF:BF16 --save

uv run --project .. python run_eval.py --host build21 --port 8097 \
    --model ggml-org/Qwen3.5-0.8B-GGUF:BF16 \
    --llama-args "--spec-type draft-mtp --spec-draft-n-max 4" --save
```

`--llama-server-bin` defaults to the path found running on `build21`
(`/home/wes/repos/github/ggml-org/llama.cpp/build/bin/llama-server`) --
override it for any other host/build layout (e.g. `paxy.lan`).

Server logs are always downloaded back to `logs/` after a run (cheap,
useful for confirming which flags actually took effect -- llama-server logs
its resolved generation settings on startup). `--save` additionally writes
full per-case timings JSON to `results/` for comparing across sweeps later.
