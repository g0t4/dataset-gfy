# Rewrite eval

Turns real AskRewrite traces from `ask_traces/rewrite/` into an eval you can
run against any OpenAI-compatible chat completions endpoint. Same trace
format as `../fim` (`messages[:-1]` is the prompt, `messages[-1]` is the
human-accepted rewrite) -- but graded very differently, because a rewrite
request rarely has one canonical answer.

## Two grading tiers, chosen per-case via `grader`

FIM completions usually have a small, enumerable set of correct answers --
text/LLM-judge comparison against a reference works well. Rewrite requests
often don't: "simplify without rejoin" can be satisfied by any number of
genuinely different approaches, so comparing candidate text against the one
answer a human happened to accept would unfairly penalize equally-valid
alternatives. But some rewrites *do* have one obviously-correct answer --
not every case is worth the overhead of a subprocess harness. Pick whichever
fits the case:

### `grader: "execute"` (default)

Each case supplies:

- **`test_harness`** -- a runnable script with a `<<<CANDIDATE>>>` marker.
  The candidate's rewritten code is spliced in at that marker and the whole
  script is actually executed.
- **`runner`** -- the command used to run it, e.g. `["fish"]`.
- **`rubric`** (optional) -- handed to an LLM judge, but *only* if execution
  already passed. The judge is told correctness is already settled and
  should only weigh in on things execution can't see: did the rewrite honor
  an explicit stylistic constraint, is it a genuine simplification, etc.

A non-zero exit (or a timeout) from the harness is an automatic `incorrect`
-- no judge call needed. This caught real, distinct bugs in every model
tried on the first case (an invalid fish multi-variable `set`, and two
variants that mishandled a no-slash edge case) that text-matching or a
judge alone likely would have missed or scored inconsistently.

### `grader: "exact_normalized"`

For the simple, unambiguous cases -- FIM-style tiered grading instead of a
harness:

- **`accepted`** -- list of exact-match candidates. Comparison only trims
  trailing whitespace; **leading indentation is kept significant**, since
  for a whole-line rewrite, dropping it would break splicing the answer
  back into the file at that exact call site.
- **`partial_accepted`** -- list of `{value, reason}` deterministic
  partial-credit matches (e.g. the same rewrite but missing the leading
  indent).
- **`rubric`** -- required here, not optional. If the candidate matches
  neither `accepted` nor `partial_accepted`, it falls to an LLM judge --
  but unlike the `execute` tier's judge, this one has to decide correctness
  itself (nothing executed the candidate), so the rubric needs to spell out
  what counts as correct/partial/incorrect, not just the qualitative extras.

## Writing a test harness

Keep it self-contained and cheap to run -- it executes on every eval run,
with a real subprocess, a scratch cwd, and a timeout
(`EXEC_TIMEOUT_SECONDS` in `run_eval.py`). Prefer covering a couple of
input vectors (including an edge case) inside the harness itself rather
than one hardcoded example, so a candidate that happens to work for the
original trace's specific input but not in general still gets caught. See
`rewrite-fish-path-split-max` in `cases.jsonl` for the pattern: wrap the
splice point in a function taking the input + expected outputs as
arguments, then call it multiple times with `and` to chain assertions.

Since this executes model-generated code, don't write harnesses that could
plausibly do anything destructive if a candidate went sideways -- these are
meant to be small, pure logic checks (string manipulation, arithmetic),
not anything touching real files/network/state outside the scratch dir.

## Running

Same conventions as `../fim` -- port-based model selection, judge is a
separate connection, `--max-tokens -1`/`0` removes the token cap, traces
dump to `debug_dumps/<run-ts>/` (gitignored).

```sh
cd evals
uv run python rewrite/run_eval.py --port 8012

# needed only if a case has a rubric (qualitative judge check)
uv run python rewrite/run_eval.py --port 8012 --judge-port 8013

uv run python rewrite/run_eval.py --port 8012 --only rewrite-fish-path-split-max --save
```

## Adding a case

1. Find or capture a trace in `ask_traces/rewrite/`.
2. Decide which `grader` fits: if there are meaningfully different valid
   rewrites, or you'd need to actually run the code to know it's right, use
   `"execute"` and write a `test_harness` that exercises the rewritten code
   against at least one real input (plus an edge case if one exists) and
   exits non-zero on failure -- splice point is the literal string
   `<<<CANDIDATE>>>`. If there's essentially one correct answer (not worth a
   subprocess), use `"exact_normalized"` with `accepted`/`partial_accepted`
   instead.
3. Write a `rubric`: optional for `"execute"` (only if there's a
   qualitative constraint execution can't verify -- an explicit "don't do
   X" in the request, a "simplify" ask that needs a judgment call on
   whether it actually got simpler); required for `"exact_normalized"`
   (it's the judge's only fallback criteria when nothing matches exactly).
4. Append one line to `cases.jsonl`.
