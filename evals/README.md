# Evals

Small, hand-built evals grown from real usage traces of my `ask-openai.nvim`
plugin (backed by local llama-server instances on `paxy.lan`), not a public
benchmark. Each subdirectory is a self-contained eval track with its own
`run_eval.py`, `cases.jsonl`, and README with the technical details:

- **[`fim/`](fim/README.md)** -- fill-in-the-middle completions. Graded by
  exact match or LLM judge against the human-accepted completion baked into
  each trace.
- **[`rewrite/`](rewrite/README.md)** -- AskRewrite (select code, describe a
  change, get a replacement) traces. Graded execution-first: the candidate
  is spliced into a runnable test harness and actually run; an optional LLM
  judge only weighs in on qualitative stuff (style, whether an explicit
  constraint was honored) once execution has already confirmed it works.
- **[`speed/`](speed/README.md)** -- raw speed testing, not correctness.
  Boots a llama-server instance over SSH on a remote GPU box with a
  specific set of server flags (the point: sweeping speculative-decoding/
  MTP params like `--spec-draft-n-max`), replays precanned prompts, and
  captures llama-server's own reported prompt/decode tokens-per-second and
  draft-accept rate.

These tracks intentionally duplicate code (client setup, judge-prompt
plumbing, trace-dump helpers) rather than sharing a library -- not worth
factoring out until there's a real need for the same plumbing across more
than one or two of them.

## What this is actually for

In rough priority order, based on what's turned out to matter in practice:

1. **Regression-testing my own tooling.** Changing `run_eval.py`, the
   grading logic, or a rubric and re-running the suite is a fast smoke test
   that I didn't break something -- independent of whether any model's
   answers are "good."
2. **Triage: is a model worth further investment?** "Does this model clear
   the bar enough that tuning its prompt is worth my time" matters far more
   than a precise ranking.
3. **Model-vs-model comparison.** A distant third, if it matters at all.
   This is *not* a leaderboard project.

## What I've learned so far

- A captured trace already contains its own gold label for free --
  `messages[:-1]` is the prompt, `messages[-1]` is what a human actually
  accepted. No manual labeling step needed to turn a trace into a case.
- Tiered grading (exact-match fast path -> partial credit -> LLM judge)
  captures more nuance than a binary pass/fail, and writing the rubric is
  itself the real eval-design work -- it forces spelling out what "partial
  credit" even means for a given case.
- Dumping full request/response traces (including judge reasoning) per run
  is cheap and makes troubleshooting truncation or prompt changes trivial
  after the fact -- didn't need it to be convenient to access, just present.
- For tasks without one canonical answer (rewrite requests especially),
  **execution-based grading beats text/LLM-judge matching** for the
  correctness axis. Proof: the same rewrite case, run against 3 different
  local models, produced 3 different subtly-broken candidates (an invalid
  fish multi-variable `set`, and two variants that mishandled a no-slash
  edge case) -- none of which would've looked obviously wrong under
  text-matching, and execution caught every one cleanly.
- `--max-tokens -1`/`0` to remove the token cap entirely turned "guess a
  limit, hit it, rerun" into a non-issue when troubleshooting a model that's
  clearly still mid-thought.
- This whole approach (real accept/reject telemetry from actual usage) is
  structurally closer to what companies like GitHub/Cursor/Sourcegraph use
  *internally* for eval than to how public academic FIM benchmarks (SAFIM,
  HumanEval-Infilling) are built -- those mine static, already-written code
  and retroactively mask "interesting" spans; they have no signal on
  whether a human would've actually wanted that completion.

## What I'm deliberately not doing (yet)

- **Not building a public benchmark.** That means a frozen held-out set,
  contamination-proofing, and scrubbing personal info (real paths, home
  dir, git author, whatever the semantic-grep context pulls in) out of
  traces before anything could be shared -- a distinct, much bigger
  decision than growing this organically.
- **Not sharing code between eval tracks.** Duplication is fine until a
  third track needs the same plumbing.
- **Not testing the live plugin's prompt-construction path.** Traces are
  frozen JSON -- replaying one tests the harness/grading code, not whether
  `ask-openai.nvim`'s *current* context-injection (semantic-grep matches,
  etc) still builds prompts correctly. Only capturing a fresh trace after a
  plugin change would catch that.
- **Not chaining dependent cases.** The two `fim-fish-typed-dots-*` cases
  were captured back-to-back on the same file, so the second trace's
  context already contains the *correct* answer to the first rather than
  whatever a model under test would have actually produced. Grading them
  independently is fine for "does the model reach the right answer given
  fixed context" but doesn't test self-consistency across the pair.

## Open questions / what I want to dig into next

- **`gemma-4-26B-A4B-it` (port 8011) reasons unusually long** relative to
  how capable it otherwise seems. Flagged during the first round of
  rewrite-eval runs; not root-caused yet -- worth comparing
  `completion_tokens`/`reasoning_content` length against the other local
  models on the same cases (the committed
  `fim/results/20260726T070445Z-google_gemma-4-26B-A4B-it-qat-q4_0-gguf.json`
  result already has some of this data) before assuming it's just how the
  model behaves rather than a sampler/template setting.
- More traces worth mining into cases -- see `TRACES_TO_TURN_INTO_EVALS.md`.
- Whether an existing benchmark (SAFIM looked like the best fit, being
  syntax-aware and multi-language) is ever worth running alongside this for
  an external comparison point.
- Whether "did I break my tooling" ever wants to be a real CI-style check
  once there's enough case coverage to make that meaningful, vs. the
  ad-hoc "just rerun it" approach that's sufficient right now.
