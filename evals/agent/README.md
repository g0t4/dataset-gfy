# Agent eval

Turns a real multi-turn tool-calling trace from `ask_traces/agents/` into
an eval: replay the initial prompt against a model under test, let it
drive its own `run_process` tool-calling loop against an isolated sandbox
copy of the starting files, then check whether it actually solved the
task -- by running the code it produced, for real, and comparing output.

This is a different shape of eval from `../fim` and `../rewrite`. Those
grade a single completion. Here the trace itself is a full agent session
(read the file, write it, compile-check it, run it, commit), so the
harness has to actually run an agent loop: send the conversation so far,
execute whatever tool calls come back, feed the results in as `tool`
messages, and repeat until the model stops calling tools or `max_turns` is
hit.

## Trace format

Same underlying trace JSON as fim/rewrite (`request_body.messages`), but
instead of `messages[:-1]` / `messages[-1]`, `load_trace_prompt_and_tools()`
finds the **first assistant message** and treats everything before it
(system prompt + preferences + semantic-grep context + the actual request)
as the prompt. Everything from the first assistant message on was the
*reference* model's own trajectory -- not replayed, not compared
token-for-token -- it's just evidence of what a working solution looked
like when this case was built.

## Sandbox

Each case lists `fixture_files`: a `{dest_name: path_relative_to_repo_root}`
map, copied into a fresh temp dir before the loop starts. Only
`run_process` is wired up for real (subprocess, `cwd` **always** pinned to
the sandbox dir -- a model-generated shell command's blast radius stays
contained to the disposable sandbox no matter what `cwd` it asks for). The
other tools a trace's system prompt might advertise (`fetch`, `screencap`,
`delegate`, `locate_anything`, `semantic_grep`) return a stub "not
available in this offline eval" tool result instead of erroring the whole
run -- there's no real index/network/screen to back them, and a case
should be scoped so it doesn't actually need them.

## Grading

One tier right now, no LLM judge: after the loop ends, independently run
`run_command` against whatever the model left in the sandbox (**not**
however the model itself happened to invoke or verify it) and compare
stdout to `expected_stdout` -- exact match, or match after stripping
leading/trailing whitespace from both sides. Anything else (non-zero exit,
timeout, real mismatch) is `incorrect`.

Get `expected_stdout` by actually running the known-good reference file
against the fixtures yourself, not by hand-copying it out of a captured
terminal log -- a hand capture can silently lose/mangle bytes (this
happened on the first case: the captured log was missing a whole output
line and had turned a literal `\n` inside a Python `repr()` into a real
newline byte).

## Running

```sh
cd evals
uv run python agent/run_eval.py --port 8014

uv run python agent/run_eval.py --port 8014 --only agent-python-verify-jinja-tokens-typed-dto-red-diff -v

# inspect exactly what the model left behind instead of trusting the report
uv run python agent/run_eval.py --port 8014 --only agent-python-verify-jinja-tokens-typed-dto-red-diff --keep-sandbox
```

## Adding a case

1. Capture a trace in `ask_traces/agents/`, including the actual fixture
   files the agent needs to read/write (not just the before/after of the
   one file being edited -- anything else the code under test depends on
   at runtime, e.g. data files it reads).
2. Write a `run_command` that deterministically proves the task was done
   (not just "the file compiles" -- actually exercises the resulting code).
3. Generate `expected_stdout` by running the known-good reference version
   of the file through that exact command yourself -- don't trust a
   hand-captured log.
4. Pick a `max_turns` with some headroom over how many turns the reference
   trajectory actually took.
5. Append one line to `cases.jsonl`.
