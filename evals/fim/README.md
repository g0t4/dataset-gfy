# FIM eval

Turns real FIM (fill-in-the-middle) traces from `ask_traces/fims/` into a
small eval you can run against any OpenAI-compatible chat completions
endpoint (llama-server, etc) to check whether a model gets the same answer
a human actually accepted.

## Why this works with zero manual labeling

Each trace file already contains its own gold label: `request_body.messages`
is `[system, ...user context, user FIM request, assistant]`, where the final
`assistant` message is the completion that was actually shown to the user in
Neovim. So for eval purposes:

- prompt = `messages[:-1]` (send exactly this)
- expected = `messages[-1].content`

No re-labeling needed -- just decide, per case, *how* to compare a new
model's answer to that expected value.

## Grading tiers

Two graders are supported, chosen per case in `cases.jsonl`:

- **`exact_normalized`** -- for cases with essentially one correct answer
  (e.g. the `||` case: only one Rust token makes that boolean condition
  valid). Whitespace-trimmed string compare against an `accepted` list.
  Always correct/incorrect, no partial credit -- if you find yourself
  wanting partial credit here, it should probably be an `llm_judge` case.

- **`llm_judge`** -- for cases where "correct" requires reasoning about the
  surrounding code (right variable name, right value, consistent with
  sibling code elsewhere in the file). A separate judge model is given the
  original FIM task, the reference answer, and a hand-written per-case
  `rubric`, and returns `{"verdict": "correct"|"partial"|"incorrect",
  "reason": "..."}`. The rubric is the actual eval-design work here --
  writing it forces you to spell out what "partial credit" even means for
  that case (e.g. right value, vague name -> partial; right name, wrong
  value -> partial).

Judge model is intentionally a separate connection (`--judge-port`) from the
model under test (`--port`) -- grading with the same small model you're
evaluating is a weak signal.

## Running

Model selection is by **port, not name**: `paxy.lan` runs one model per port
(static allocation via systemd user services), so `--port` is all you need
to pick which model to test. The model name itself is never sent in the
request -- it's read back from `response_metadata["model_name"]` on the
first completion and used to label the report and the saved results file,
so a run always reflects whatever model actually answered on that port
(not whatever you assumed was running there).

Connects via `ChatLlamaServer` (from the sibling
[`langchain-llama-server`](../../../langchain-llama-server) project) rather
than a raw `openai` client, at `http://paxy.lan:<port>/v1`.

```sh
cd evals
uv run python fim/run_eval.py --port 8012

# needed only if a case uses grader:llm_judge
uv run python fim/run_eval.py --port 8012 --judge-port 8013

# just one case, e.g. while iterating on a rubric
uv run python fim/run_eval.py --port 8012 --only fim-rust-boolean-or

# save a results/<timestamp>-<model>.json for tracking across models/runs
uv run python fim/run_eval.py --port 8012 --judge-port 8013 --save
```

Runs at `temperature=0` by default (deterministic grading > sampling
fidelity to the original trace, which was captured at temperature=1).

## Adding a case

1. Find or capture a trace in `ask_traces/fims/` -- see
   `../TRACES_TO_TURN_INTO_EVALS.md` for candidates already flagged as
   interesting.
2. Read it (`jq .request_body.messages`) and decide: is there basically one
   right answer (`exact_normalized`), or does grading require reading
   surrounding code (`llm_judge`)?
3. Append one line to `cases.jsonl`: `id`, `source_trace` (path relative to
   the repo root, i.e. starting with `ask_traces/...`), `language`,
   `grader`, and either `accepted` (list of exact strings) or `rubric`
   (free text judge instructions). `notes` is optional context for future
   you about *why* this case is interesting.

## Known limitation

The two `typed_dots` cases (`fim-fish-typed-dots-set` and
`fim-fish-typed-dots-expand`) were captured back-to-back on the same file --
the second trace's surrounding code already contains the human-accepted
answer to the first, not whatever a model under test would have produced.
Right now they're graded independently, which is fine for "does the model
reach the right answer given fixed context" but doesn't test whether a
model is *self-consistent* across the pair (e.g. names the variable
`typed_dots` in the first and forgets to reuse the same name in the second
if it had generated its own answer). Chaining the model's own case-1 output
into case-2's prompt before grading would test that, but isn't implemented
yet -- see `../../ask_traces/fims/2026-07/2026-07-25_001/NOTES.yml` for the
related "combine both FIMs into one edit prediction" idea.
