#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Run agent eval cases (cases.jsonl) against a model under test: replay a
captured multi-turn tool-calling trace's initial prompt, let the model
under test drive its own real tool-calling loop against an isolated
sandbox copy of the fixture files, then independently execute a grading
command against whatever final state the model left behind.

Each case in cases.jsonl points at a trace file captured from real
ask-openai.nvim agent usage (`ask_traces/agents/`). Unlike fim/rewrite,
these traces are a full multi-turn tool-calling session, not a single
request/response. `load_trace_prompt_and_tools()` finds the first
assistant message and treats everything before it as the initial prompt
(system + preferences + semantic-grep context + the actual request) --
everything from there on was the reference model's own trajectory
(reading the file, writing it, running it, committing), which is exactly
what the model under test now has to reproduce on its own, for real,
against a private sandbox.

Only `run_process` is wired up for real (subprocess, cwd pinned to the
sandbox dir regardless of what cwd the model asks for). The other tools
the trace's system prompt advertises (fetch/screencap/delegate/
locate_anything/semantic_grep) return a stub "not available" tool result
-- there's no real index/network/screen to back them in an offline eval,
and this case doesn't need them.

Grading is deliberately simple for now (single tier, no LLM judge): after
the agent loop ends (model stops calling tools, or `max_turns` is hit),
independently run `run_command` against the sandbox and compare stdout to
`expected_stdout` -- exact match, or match after stripping leading/
trailing whitespace from both sides. A non-zero exit, a timeout, or a
stdout mismatch is `incorrect`. No partial credit tier yet.

Usage:
    uv run --project .. python run_eval.py --port 8014
    uv run --project .. python run_eval.py --port 8014 --only agent-python-verify-jinja-tokens-typed-dto-red-diff --save
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import argcomplete

if TYPE_CHECKING:
    import openai

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parent

PAXY_HOST = "paxy.lan"

EXEC_TIMEOUT_SECONDS = 15
RUN_PROCESS_TIMEOUT_CAP_MS = 60_000

# a case's run_command/model tool calls say "python3"/"python" -- swap in
# the interpreter actually running this harness (which has the eval
# project's deps, e.g. jinja2/rich, installed) instead of relying on
# whatever bare "python3" resolves to on PATH.
INTERPRETER_ALIASES = {"python3": sys.executable, "python": sys.executable}


@dataclass
class Case:
    id: str
    source_trace: str
    language: str
    # dest filename (in the sandbox) -> path relative to REPO_ROOT to copy from
    fixture_files: dict[str, str]
    # command to independently execute against the sandbox after the agent loop,
    # to produce output for grading -- NOT what the model itself runs
    run_command: list[str]
    expected_stdout: str
    grader: str = "execute_stdout_match"
    max_turns: int = 8
    # answer-constraint tightness, for slicing reports separately from verdict
    # (see fim/run_eval.py for the scale)
    constraint: str | None = None
    notes: str | None = None


@dataclass
class Result:
    id: str
    verdict: str
    reason: str
    turns_used: int
    hit_max_turns: bool
    tool_calls_made: list[str]
    final_stdout: str
    final_returncode: int | None
    expected_stdout: str
    completion_tokens: int | None
    finish_reason: str | None
    model_name: str
    constraint: str | None = None


def make_client(port: int) -> openai.OpenAI:
    import openai
    return openai.OpenAI(base_url=f"http://{PAXY_HOST}:{port}/v1", api_key="none", timeout=120)


def ping_server(port: int, label: str) -> None:
    """See fim/run_eval.py -- same fail-fast rationale."""
    import httpx
    url = f"http://{PAXY_HOST}:{port}/v1/models"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        sys.exit(f"startup check failed: {label} on port {port} did not respond to GET {url} ({e}) -- "
                  f"is llama-server up on that port?")


def dump_json(dump_dir: Path, name: str, data: dict) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))


def load_cases(path: Path) -> list[Case]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(Case(**json.loads(line)))
    return cases


def complete_case_ids(prefix: str, parsed_args: argparse.Namespace, **kwargs) -> list[str]:
    cases_path = Path(getattr(parsed_args, "cases", None) or (AGENT_DIR / "cases.jsonl"))
    try:
        return [c.id for c in load_cases(cases_path)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def load_trace_prompt_and_tools(source_trace: str) -> tuple[list[dict], list[dict]]:
    trace = json.loads((REPO_ROOT / source_trace).read_text())
    messages = trace["request_body"]["messages"]
    tools = trace["request_body"]["tools"]
    first_assistant_idx = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return messages[:first_assistant_idx], tools


def setup_sandbox(case: Case) -> Path:
    sandbox_dir = Path(tempfile.mkdtemp(prefix=f"agent-eval-{case.id}-"))
    for dest_name, src_rel in case.fixture_files.items():
        shutil.copy(REPO_ROOT / src_rel, sandbox_dir / dest_name)
    return sandbox_dir


def execute_run_process(args: dict, sandbox_dir: Path) -> dict:
    command_line = args.get("command_line")
    argv = args.get("argv")
    timeout_ms = min(args.get("timeout_ms") or 30_000, RUN_PROCESS_TIMEOUT_CAP_MS)
    # NOTE: cwd is always pinned to the sandbox dir, regardless of any `cwd`
    # arg the model passes -- keeps a model-generated shell command's blast
    # radius contained to the disposable sandbox no matter what it asks for.
    try:
        if command_line:
            proc = subprocess.run(command_line, shell=True, cwd=sandbox_dir, capture_output=True,
                                   text=True, timeout=timeout_ms / 1000, input=args.get("stdin_text"))
        elif argv:
            proc = subprocess.run(argv, cwd=sandbox_dir, capture_output=True,
                                   text=True, timeout=timeout_ms / 1000, input=args.get("stdin_text"))
        else:
            return {"content": [{"name": "ERROR", "type": "text", "text": "run_process called with neither command_line nor argv"}]}
    except subprocess.TimeoutExpired:
        return {"content": [{"name": "ERROR", "type": "text", "text": f"process timed out after {timeout_ms}ms"}]}

    content = [{"name": "EXIT_CODE", "type": "text", "text": str(proc.returncode)}]
    if proc.stdout:
        content.append({"name": "STDOUT", "type": "text", "text": proc.stdout})
    if proc.stderr:
        content.append({"name": "STDERR", "type": "text", "text": proc.stderr})
    return {"content": content}


def execute_stub_tool(name: str) -> dict:
    return {"content": [{"name": "ERROR", "type": "text",
                          "text": f"{name!r} is not available in this offline agent-eval sandbox (only run_process is wired up)"}]}


def run_agent_loop(client: openai.OpenAI, prompt_messages: list[dict], tools: list[dict], case: Case,
                    sandbox_dir: Path, max_tokens: int, temperature: float, verbose: bool) -> dict:
    messages = [dict(m) for m in prompt_messages]
    tool_calls_made: list[str] = []
    completion_tokens_total = 0
    finish_reason = None
    model_name = "unknown"
    turn = 0
    hit_max_turns = False

    for turn in range(1, case.max_turns + 1):
        resp = client.chat.completions.create(model="", messages=messages, tools=tools,
                                               temperature=temperature, max_tokens=max_tokens)
        choice = resp.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason
        model_name = resp.model or model_name
        if resp.usage:
            completion_tokens_total += resp.usage.completion_tokens or 0

        if verbose:
            names = [tc.function.name for tc in (msg.tool_calls or [])]
            print(f"  [turn {turn}] finish={finish_reason} tool_calls={names} content={msg.content!r}", file=sys.stderr)

        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls_made.append(name)
            result = execute_run_process(args, sandbox_dir) if name == "run_process" else execute_stub_tool(name)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    else:
        hit_max_turns = True

    return {
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "completion_tokens": completion_tokens_total,
        "finish_reason": finish_reason,
        "model_name": model_name,
        "turns_used": turn,
        "hit_max_turns": hit_max_turns,
    }


def resolve_run_command(run_command: list[str]) -> list[str]:
    head, *rest = run_command
    return [INTERPRETER_ALIASES.get(head, head), *rest]


def grade(case: Case, sandbox_dir: Path) -> tuple[str, str, str, int | None]:
    cmd = resolve_run_command(case.run_command)
    try:
        proc = subprocess.run(cmd, cwd=sandbox_dir, capture_output=True, text=True, timeout=EXEC_TIMEOUT_SECONDS)
        stdout, stderr, returncode, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout, stderr, returncode, timed_out = (e.stdout or ""), (e.stderr or ""), None, True
    except FileNotFoundError as e:
        return "incorrect", f"could not run grading command {cmd!r}: {e}", "", None

    if timed_out:
        return "incorrect", f"grading run timed out after {EXEC_TIMEOUT_SECONDS}s", stdout, None
    if returncode != 0:
        return "incorrect", f"grading run exited {returncode}: {stderr.strip()[:300] or '(no stderr)'}", stdout, returncode
    if stdout == case.expected_stdout:
        return "correct", "stdout matches expected exactly", stdout, returncode
    if stdout.strip() == case.expected_stdout.strip():
        return "correct", "stdout matches expected after trimming leading/trailing whitespace", stdout, returncode
    return "incorrect", "stdout did not match expected (even after trimming whitespace)", stdout, returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help=f"port of the llama-server instance to test, on {PAXY_HOST}")
    parser.add_argument("--cases", default=str(AGENT_DIR / "cases.jsonl"), help="path to cases.jsonl").completer = argcomplete.completers.FilesCompleter(allowednames=[".jsonl"])
    parser.add_argument("--max-tokens", type=int, default=8192,
                         help="cap on generated tokens per agent turn; 0 or -1 removes the cap entirely")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save", action="store_true", help="write results JSON to results/<timestamp>-<model>.json")
    parser.add_argument("--only", default=None, help="only run the case with this id").completer = complete_case_ids
    parser.add_argument("--verbose", "-v", action="store_true", help="print each agent turn (tool calls, content) as it happens")
    parser.add_argument("--keep-sandbox", action="store_true", help="don't delete the sandbox dir after grading -- print its path so you can inspect final file state")

    argcomplete.autocomplete(parser)
    import rich
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        cases = [c for c in cases if c.id == args.only]
        if not cases:
            sys.exit(f"no case with id {args.only!r}")

    ping_server(args.port, "model under test")
    client = make_client(args.port)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump_dir = AGENT_DIR / "debug_dumps" / run_ts

    results: list[Result] = []
    for case in cases:
        if args.verbose:
            print(f"running {case.id}...", file=sys.stderr)

        prompt_messages, tools = load_trace_prompt_and_tools(case.source_trace)
        sandbox_dir = setup_sandbox(case)
        try:
            loop_result = run_agent_loop(client, prompt_messages, tools, case, sandbox_dir,
                                          args.max_tokens, args.temperature, args.verbose)
            dump_json(dump_dir, case.id, {
                "sandbox_dir": str(sandbox_dir) if args.keep_sandbox else None,
                **loop_result,
            })
            verdict, reason, final_stdout, final_returncode = grade(case, sandbox_dir)
            if loop_result["hit_max_turns"]:
                reason += f" [NOTE: agent loop hit max_turns={case.max_turns} without finishing on its own]"
        finally:
            if args.keep_sandbox:
                print(f"  sandbox kept: {sandbox_dir}", file=sys.stderr)
            else:
                shutil.rmtree(sandbox_dir, ignore_errors=True)

        results.append(Result(
            id=case.id,
            verdict=verdict,
            reason=reason,
            turns_used=loop_result["turns_used"],
            hit_max_turns=loop_result["hit_max_turns"],
            tool_calls_made=loop_result["tool_calls_made"],
            final_stdout=final_stdout,
            final_returncode=final_returncode,
            expected_stdout=case.expected_stdout,
            completion_tokens=loop_result["completion_tokens"],
            finish_reason=loop_result["finish_reason"],
            model_name=loop_result["model_name"],
            constraint=case.constraint,
        ))

    model_names = {r.model_name for r in results}
    if len(model_names) > 1:
        rich.print(f"[bold white on red] WARNING [/] port {args.port} answered with more than one model name "
                   f"across cases: {model_names} -- was the server restarted with a different model mid-run?", file=sys.stderr)
    resolved_model = results[0].model_name if results else "unknown"

    print_report(resolved_model, args.port, results, dump_dir)

    if args.save:
        save_results(resolved_model, results)


def print_report(model: str, port: int, results: list[Result], dump_dir: Path):
    import rich
    icon = {"correct": "✅", "incorrect": "❌"}
    print()
    rich.print(f"agent eval -- model: [bold black on bright_yellow] {model} [/]  (port {port})")
    print(f"trace dumps: {dump_dir}")
    print("=" * 60)
    for r in results:
        constraint_tag = f", {r.constraint}" if r.constraint else ""
        max_turns_tag = " [HIT MAX TURNS]" if r.hit_max_turns else ""
        print(f"\n{icon.get(r.verdict, '?')} [{r.verdict}] {r.id}  ({r.turns_used} turns{constraint_tag}, "
              f"{r.completion_tokens} tokens, finish={r.finish_reason}){max_turns_tag}")
        print(f"   tool calls : {r.tool_calls_made}")
        print(f"   expected   : {r.expected_stdout!r}")
        print(f"   got        : {r.final_stdout!r}")
        if r.reason:
            print(f"   reason     : {r.reason}")

    total = len(results)
    correct = sum(1 for r in results if r.verdict == "correct")
    incorrect = total - correct
    print("\n" + "-" * 60)
    print(f"{correct}/{total} correct, {incorrect}/{total} incorrect")


def save_results(model: str, results: list[Result]):
    import re
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out_path = AGENT_DIR / "results" / f"{ts}-{safe_model}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "results": [asdict(r) for r in results]}, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
