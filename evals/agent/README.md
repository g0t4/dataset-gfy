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

A trace can be a *continuation* of a longer real session -- one capture
with several follow-up asks chained onto the same file (verify.py, then
"now build pretty.py", then "now build render_check.py", ...). A case
sourced from a later follow-up sets `prompt_end_idx` explicitly
(`messages[:prompt_end_idx]`) instead of relying on the first-assistant-
message default, which only makes sense for a case starting at message 0.
Point it just past the user message that kicks off the follow-up you're
turning into a case -- everything before that (including the earlier
sub-tasks' back-and-forth) becomes prompt context, same as the reference
session had it.

## Sandbox: real containment via Docker

`run_process` runs every model-driven command via `docker exec` into a
per-case container (built from `agent/docker/Dockerfile`, image
`agent-eval-sandbox:latest`), bind-mounted to the sandbox dir at
`/workspace` and always executed from there. This is real filesystem
containment -- the container has no access to the host filesystem beyond
that one bind-mounted directory -- not just a pinned-`cwd` convention a
shell command could `cd` its way out of.

That distinction isn't hypothetical: it's how `agent-python-render-check-
dual-template-jinja-match` was first found to be broken. Before Docker was
wired in, `run_process` just ran `subprocess.run(..., cwd=sandbox_dir)` --
pinning where a command *started*, not what it could reach. The trace's
source repo (`ask-openai.nvim`) was a real, currently-checked-out working
directory on the eval machine, with the reference commit already sitting on
`master`. The model under test `cd`'d straight there, ran the *already-
finished* `render_check.py`, and declared victory -- its actual sandbox
stayed empty the whole time. Grading still caught it (it only ever looks at
the sandbox), but the eval measured nothing. The first fix attempt --
`prompt_patches`, a `{old_substring: new_substring}` text substitution
across the whole replayed prompt, redacting the real path -- wasn't enough
either: denied the stated path, the model got suspicious and ran `fd -t d
muse` against the real home directory, using one of the very tools its own
system prompt advertised, and found the real repo anyway under its
unredacted name. Redacting text a model reads can't stop a model that
decides to search the real filesystem from scratch. Docker closes that off
structurally: inside the container there is nothing to find.

`prompt_patches` still exists and still gets applied (see `Case.prompt_patches`
in `run_eval.py`) -- it's just no longer load-bearing for containment. Use it
for narrative cleanliness (a stated cwd that doesn't match where fixtures
actually landed can confuse a model into wasted turns) rather than as a
security boundary.

Each case lists `fixture_files`: a `{dest_name: path_relative_to_repo_root}`
map, copied into a fresh temp dir before the loop starts, then bind-mounted
into the container. `dest_name` can include subdirectories (e.g.
`tools/chat_viewer/browser/__main__.py`, needed when the fixture is a real
package module with sibling `__init__.py`/import-chain files) --
`setup_sandbox()` creates parent directories as needed before copying. The
other tools a trace's system prompt might advertise
(`fetch`, `screencap`, `delegate`, `locate_anything`, `semantic_grep`) return
a stub "not available in this offline eval" tool result instead of erroring
the whole run -- there's no real index/network/screen to back them, and a
case should be scoped so it doesn't actually need them.

One-time setup: `docker build -t agent-eval-sandbox agent/docker` (needs
Docker running). `run_eval.py` fails fast at startup if the daemon isn't
reachable or the image hasn't been built.

## Grading

One tier right now, no LLM judge: after the loop ends, independently run
`run_command` against whatever the model left in the sandbox (**not**
however the model itself happened to invoke or verify it) and compare
stdout to `expected_stdout` -- exact match, or match after stripping
leading/trailing whitespace from both sides. Anything else (non-zero exit,
timeout, real mismatch) is `incorrect`.

Grading runs on the **host**, not through `docker exec` -- only the model's
own tool-calling loop is containerized. `resolve_run_command()` remaps a
`"python3"`/`"python"` head to `sys.executable` (the interpreter running
`run_eval.py` itself, i.e. this project's `uv`-managed venv), and
`subprocess.run(..., cwd=sandbox_dir)` runs directly against the sandbox
dir on disk -- there's no `/workspace` mount at grading time, and any
package `run_command` needs (e.g. `rich`, `argcomplete`) has to be
available in the host venv, not just the Docker image. A `run_command` or
fixture script that hardcodes a container path like `/workspace/...` will
fail here even though it works fine when the model runs the equivalent
command through its own sandboxed `run_process` calls -- use a
`cwd`-relative path instead (`agent-python-trace-browser-jq-dump-key`'s
`grade_driver.py` hit exactly this).

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
