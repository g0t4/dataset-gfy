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

## Three categories

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
- **`summary`**: huge prompt AND a real (not near-empty) completion --
  deliberately mixes both axes because that's what the actual workload is
  (e.g. summarizing a long conversation), not an accidental blur of two
  clean measurements. Doesn't replay a trace's own captured prompt like the
  other two categories do -- `load_summary_prompt`/`render_trace_transcript`
  wrap the trace's rendered conversation inside a fresh "please summarize
  the following conversation" instruction instead. `speed-summary-short-trace`
  (~600-token document) and `speed-summary-long-trace` (~110K-token
  document, a real multi-hour agent session) pair a short and long input at
  the same task, so prefill-time-vs-input-size can be read off directly
  while holding the task constant. Motivated by summarization being a
  real prefill-dominant workload where MTP's draft-accept rate might behave
  very differently on extractive/paraphrase-heavy output than on the
  code-generation content `speed-gen-lua-tower-of-hanoi` produces --
  not yet confirmed either way, worth a dedicated sweep later. Timing
  only, same as every other case here -- summary *quality* isn't graded;
  a real quality-graded summarization eval is a deliberately deferred
  future track, once there's a feel for what's actually wanted from one.

`generation`/`prefill` isolate one mechanism each -- mixing both concerns
in a case built for that purpose just averages two numbers together and
tells you less about either. When adding a case in one of those two
categories, decide which axis it's meant to isolate and lean into that --
don't reach for a "medium" prompt with a "medium" completion; the extremes
are what make a number legible. `summary` is the deliberate exception:
it's not accidentally mixing axes, it's a distinct real workload shape
that happens to need both at once.

## Repeats and seed

Every case runs `--repeats` times (default 5, matching `llama-bench`'s own
`-r`/`--repetitions` default) and the report shows mean ± stdev per case,
not a single number. One run is luck of the draw -- GPU contention, thermal
drift, OS scheduling can all put a single measurement noticeably off from
what the params actually deliver on average. `llama-bench` itself defaults
to 5 repetitions even though its workload is 100% synthetic (dummy token
counts, no real sampling) -- proof this kind of noise is worth averaging
over even in the best case; a shared dev box like `build21` (routinely
mid-session with someone else's model loaded) has more of it, not less.

Also matching `llama-bench`: an untimed `--warmup` run (default on,
`--no-warmup` to skip) before each case's timed repeats. The very first
request against a freshly-started server can genuinely diverge from the
rest -- cuBLAS algorithm selection and GPU clock ramp-up both tend to
land on whichever call happens to go first, and one-time buffer
allocation only happens once. Observed directly: a 1.7B baseline run
(before this flag existed) showed `predicted_n` at `2141, 2731, 2731,
2731, 2731` across 5 repeats of the exact same case/seed/params -- repeat
1 the outlier, repeats 2-5 self-consistent. `--warmup` moves that
cold-start cost onto a discarded run instead of letting it skew repeat 1
of the counted set.

Every request also carries a fixed `--seed` (default 42). The theory was
that llama.cpp's speculative decoding is sampling-equivalent to the target
model alone -- drafts are verified against the target's own probabilities,
so a fixed seed should produce the *same* output tokens whether MTP is on
or off, just via a different number of forward passes, making an
across-`--llama-args` comparison for the same case apples-to-apples.

**In practice this only holds within one `--llama-args` combo, not across
them.** A real Qwen3.5-0.8B n-max sweep (seed pinned at 7) showed
`speed-gen-lua-tower-of-hanoi` finishing at a *different* token count per
`--spec-draft-n-max` value -- 572 tokens with no spec decoding, 1221 tokens
at n-max=2 and n-max=3, then diverging into the 4096-token `--max-tokens`
cap at n-max=4/5/6 -- despite every repeat *within* a given n-max value
landing on the exact same token count every time. Best working theory:
verifying a draft batch runs matmuls at a different batch size than
single-token decode, and floating-point addition isn't associative, so the
computed logits differ by enough to flip an argmax/threshold decision at
some point in the sequence -- after which the two trajectories are
unrelated completions, not "the same text, a bit faster." Repeats *within*
a `--llama-args` combo are still a clean read on environmental noise (seed
+ params together are still fully deterministic -- same token count, same
draft_n, every time). But an across-combo tok/s comparison should be read
with that caveat: n-max=4/5/6's much higher decode tok/s and draft-accept%
in that sweep likely partly reflect that they ended up decoding a longer,
more repetitive completion (which is *easier* to draft correctly), not
purely a "higher n-max is faster" effect. Treat a big jump in `predicted_n`
alongside a big jump in accept% across a sweep as a sign the comparison
isn't apples-to-apples anymore, not as a clean win.

One real failure mode this surfaced during validation, worth knowing about:
a fixed seed doesn't prevent a bad generation, it just makes it
*deterministic* -- if seed 42 happens to send a given model/case into a
repetition loop instead of hitting EOS, every single repeat hits that same
loop (unlike unseeded runs, where only some fraction of random draws would).
`--max-tokens` (default 4096) exists specifically to bound the damage; if a
case's mean looks suspiciously pinned at exactly `--max-tokens`, that's the
tell -- try a different `--seed` for that case, not a sign the harness is
broken.

## Reasoning models

A reasoning-capable model can spend its *entire* `--max-tokens` budget on
`reasoning_content` and never reach an actual answer -- `Result` has a
`reasoning` field (full text, same as `completion`) specifically so this
is visible instead of silently showing up as an empty/truncated
`completion` with no explanation. In `--verbose` output, a case that hits
this shows up with `[N chars reasoning, 0 chars content -- likely ran out
of budget mid-thought]` tacked onto its done-line.

This isn't hypothetical: `speed-summary-long-trace` (~110K-token document)
against Qwen3.5-0.8B burned through 12000 tokens of pure reasoning and
never emitted a single answer token, despite that model's own card
claiming it's "non-thinking by default" -- that claim describes a
different serving stack's default, not llama-server's raw jinja template
rendering, which reasoned anyway with nothing set. Use `--disable-thinking`
(sets `chat_template_kwargs.enable_thinking=false`, the request-body field
llama-server forwards straight into the chat template -- confirmed against
a real captured trace using this exact mechanism) / `--enable-thinking` to
force one or the other rather than trusting either doc's claimed default;
leave both unset to get whatever the template's own default is. Forcing
thinking off doesn't guarantee a *good* answer, though -- same case,
same model, thinking off, produced 17K chars of the model hallucinating
and looping a fake git-commit-replay tool call instead of ever attempting
a summary. That's a capability limit (0.8B on a 110K-token document), not
a reasoning-toggle bug -- the same model handled `speed-summary-short-trace`
(~600 tokens) fine.

## Cases and what to think about when adding one

- **Cold cache is load-bearing for `prefill` cases.** llama-server caches
  the KV state of a prompt it's already seen (`cache_n` in the timings
  block). `run_eval.py` handles this automatically between `prefill`
  repeats via `POST /slots/{id}?action=erase` (that's why the server is
  always started with `--slot-save-path` -- llama-server gates the entire
  `/slots` POST route, erase included, behind that flag being set at all,
  even though erase itself never writes a file: see `server-context.cpp`'s
  `post_slots` handler). If you ever bypass `run_eval.py`'s own case loop
  (e.g. `--reuse-server` plus manually replaying a prompt twice against the
  same server), the underlying problem is still real: a second hit on an
  already-cached prompt measures cache-hit speed, not real prefill
  throughput -- an order-of-magnitude different number.
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
    --model ggml-org/Qwen3.5-0.8B-GGUF:BF16 --verbose --save

uv run --project .. python run_eval.py --host build21 --port 8097 \
    --model ggml-org/Qwen3.5-0.8B-GGUF:BF16 \
    --llama-args "--spec-type draft-mtp --spec-draft-n-max 4" --verbose --save
```

`-v`/`--verbose` prints prompt size, time-to-first-token, and a per-repeat
timing summary as each case streams (to stderr) instead of only the final
report table -- useful for a sweep that's going to take a while, or for
noticing a stuck/looping repeat before it burns through `--max-tokens`.

`--llama-server-bin` defaults to the path found running on `build21`
(`/home/wes/repos/github/ggml-org/llama.cpp/build/bin/llama-server`) --
override it for any other host/build layout (e.g. `paxy.lan`).

Server logs are always downloaded back to `logs/` after a run (cheap,
useful for confirming which flags actually took effect -- llama-server logs
its resolved generation settings on startup). `--save` additionally writes
full per-case timings JSON to `results/` for comparing across sweeps later.
