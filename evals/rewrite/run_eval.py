#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Run rewrite eval cases (cases.jsonl) against a model under test, grade the
rewrites, and print a report.

Each case in cases.jsonl points at a trace file captured from real
AskRewrite usage. The trace's messages[:-1] is the exact prompt that was
sent (original selection + free-text request); messages[-1] (the assistant
message) is the human-approved rewrite that was actually accepted, kept
here only as a reference for the judge -- it is NOT compared textually,
since a rewrite request rarely has one canonical answer.

Grading is execution-first, not text-match: each case supplies a
test_harness (a runnable script with a <<<CANDIDATE>>> marker) plus a
runner command. The candidate's rewritten code is spliced into the marker
and actually executed; a non-zero exit means incorrect, full stop. Only
once execution passes does an optional rubric get handed to an LLM judge,
to weigh in on qualitative stuff execution can't see (did it honor an
explicit stylistic constraint, is it a genuine simplification, etc).

Model selection is by port, not name -- see evals/fim/run_eval.py for the
rationale (copied verbatim here, this file started as a copy of that one).

Usage:
    uv run --project .. python run_eval.py --port 8012
    uv run --project .. python run_eval.py --port 8012 --judge-port 8013 --save
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import argcomplete

if TYPE_CHECKING:
    from langchain_llama_server import ChatLlamaServer

REPO_ROOT = Path(__file__).resolve().parents[2]
REWRITE_DIR = Path(__file__).resolve().parent

PAXY_HOST = "paxy.lan"

CANDIDATE_MARKER = "<<<CANDIDATE>>>"
EXEC_TIMEOUT_SECONDS = 10


@dataclass
class Case:
    id: str
    source_trace: str
    language: str
    runner: list[str]  # command used to execute the harness script, e.g. ["fish"]
    test_harness: str  # runnable script; CANDIDATE_MARKER gets replaced with the model's rewrite
    rubric: str | None = None  # optional qualitative check, only run if execution passes
    # answer-constraint tightness, for slicing reports separately from verdict (see fim/run_eval.py for the scale)
    constraint: str | None = None
    notes: str | None = None


@dataclass
class Result:
    id: str
    verdict: str
    reason: str
    graded_by: str  # "execute" or "execute+llm_judge"
    reference: str
    candidate: str
    exec_passed: bool
    exec_returncode: int | None
    completion_tokens: int | None
    finish_reason: str | None
    model_name: str
    judge_model_name: str | None = None
    constraint: str | None = None


def make_client(port: int) -> ChatLlamaServer:
    from langchain_llama_server import ChatLlamaServer
    return ChatLlamaServer(base_url=f"http://{PAXY_HOST}:{port}/v1", api_key="none", timeout=120)


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
            data = json.loads(line)
            cases.append(Case(**data))
    return cases


def complete_case_ids(prefix: str, parsed_args: argparse.Namespace, **kwargs) -> list[str]:
    cases_path = Path(getattr(parsed_args, "cases", None) or (REWRITE_DIR / "cases.jsonl"))
    try:
        return [c.id for c in load_cases(cases_path)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def load_trace_prompt_and_reference(source_trace: str) -> tuple[list[dict], str]:
    trace_path = REPO_ROOT / source_trace
    trace = json.loads(trace_path.read_text())
    messages = trace["request_body"]["messages"]
    prompt_messages = messages[:-1]
    reference = messages[-1]["content"]
    return prompt_messages, reference


def run_harness(case: Case, candidate: str, dump_dir: Path) -> tuple[bool, str, int | None]:
    if CANDIDATE_MARKER not in case.test_harness:
        raise ValueError(f"case {case.id!r} test_harness is missing the {CANDIDATE_MARKER} marker")
    script = case.test_harness.replace(CANDIDATE_MARKER, candidate)

    with tempfile.TemporaryDirectory(prefix="rewrite-eval-") as scratch_dir:
        script_path = Path(scratch_dir) / f"harness.{case.language}"
        script_path.write_text(script)
        try:
            proc = subprocess.run(
                [*case.runner, str(script_path)],
                cwd=scratch_dir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
            returncode = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
            timed_out = False
        except subprocess.TimeoutExpired as e:
            returncode = None
            stdout, stderr = (e.stdout or ""), (e.stderr or "")
            timed_out = True

    dump_json(dump_dir, f"{case.id}.exec", {
        "script": script,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
    })

    if timed_out:
        return False, f"timed out after {EXEC_TIMEOUT_SECONDS}s", None
    if returncode != 0:
        return False, f"exited {returncode}: {stderr.strip()[:300] or '(no stderr)'}", returncode
    return True, "executed successfully", returncode


JUDGE_PROMPT_TEMPLATE = """You are grading a code-rewrite suggestion. A separate execution-based test
harness already ran this candidate and confirmed it produces correct output -- your job is NOT to
re-check functional correctness, that's already settled. Judge only the qualitative aspects called out
in the rubric below (e.g. whether it honors an explicit stylistic constraint the user gave, whether it's
a genuine simplification, whether the code looks like it's doing real general-purpose work rather than
something suspicious that just happens to satisfy the test harness).

Task given to the model under test (their instructions plus the original code they selected):
---
{rewrite_task}
---

Reference (human-approved) rewrite that was actually accepted for this request, for context only --
the candidate does NOT need to match this, other valid rewrites exist:
<<<REFERENCE>>>
{reference}
<<<END REFERENCE>>>

Rubric for this specific case:
{rubric}

Candidate rewrite to grade:
<<<CANDIDATE>>>
{candidate}
<<<END CANDIDATE>>>

Respond with ONLY a JSON object, no markdown fences, no extra commentary:
{{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}
"""


def grade_llm_judge(candidate: str, case: Case, prompt_messages: list[dict], reference: str, judge_client: ChatLlamaServer, dump_dir: Path) -> tuple[str, str, str]:
    rewrite_task = prompt_messages[-1]["content"]
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        rewrite_task=rewrite_task,
        reference=reference,
        rubric=case.rubric,
        candidate=candidate,
    )
    ai_message = judge_client.invoke([{"role": "user", "content": judge_prompt}], temperature=0)
    judge_model_name = ai_message.response_metadata.get("model_name") or "unknown"
    raw = (ai_message.content or "").strip()
    dump_json(dump_dir, f"{case.id}.judge", {
        "judge_prompt": judge_prompt,
        "content": raw,
        "reasoning_content": ai_message.additional_kwargs.get("reasoning_content"),
        "finish_reason": ai_message.response_metadata.get("finish_reason"),
        "model_name": judge_model_name,
        "usage_metadata": ai_message.usage_metadata,
    })
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return "incorrect", f"judge did not return JSON: {raw!r}", judge_model_name
    try:
        parsed = json.loads(match.group(0))
        verdict = parsed.get("verdict", "incorrect")
        reason = parsed.get("reason", "")
        if verdict not in ("correct", "partial", "incorrect"):
            return "incorrect", f"judge returned unknown verdict {verdict!r}", judge_model_name
        return verdict, reason, judge_model_name
    except json.JSONDecodeError:
        return "incorrect", f"judge returned unparseable JSON: {raw!r}", judge_model_name


def stream_completion(model_client: ChatLlamaServer, prompt_messages: list[dict], invoke_kwargs: dict, trace: bool):
    """Stream a completion, optionally echoing tokens to stderr as they arrive.

    Passes extra_body={"verbose": True} so llama-server attaches its own
    "timings" block (predicted_n, predicted_per_second, draft accept stats,
    etc) to the finish-reason chunk -- richer than the standard OpenAI
    "usage" field, and unlike usage it doesn't require a separate trailing
    chunk with an empty choices list (which older llama-server responses
    could send with stream_options.include_usage, tripping up naive chunk
    parsing).

    Returns (message, timings, error). `message` is whatever was
    accumulated -- possibly partial, if `error` is set -- so callers can
    read .content / .additional_kwargs["reasoning_content"] off it either
    way instead of getting nothing on a client-side timeout/connection drop
    mid-stream. `timings` is the raw dict off the finish-reason chunk, or
    None if the stream broke before reaching it.
    """
    import openai
    message = None
    timings = None
    error = None
    try:
        for chunk in model_client.stream(prompt_messages, extra_body={"verbose": True}, **invoke_kwargs):
            debug_info = getattr(chunk, "debug", None)
            if debug_info is not None and debug_info.timings:
                timings = debug_info.timings
            if trace:
                reasoning_delta = chunk.additional_kwargs.get("reasoning_content") or ""
                if reasoning_delta:
                    print(reasoning_delta, end="", flush=True, file=sys.stderr)
                if chunk.content:
                    print(chunk.content, end="", flush=True, file=sys.stderr)
            message = chunk if message is None else message + chunk
    except openai.APIConnectionError as e:
        error = e
    if trace:
        print(file=sys.stderr)
    return message, timings, error


def grade(candidate: str, case: Case, prompt_messages: list[dict], reference: str, judge_client: ChatLlamaServer | None, dump_dir: Path) -> tuple[str, str, str, str | None, bool, int | None]:
    exec_passed, exec_reason, exec_returncode = run_harness(case, candidate, dump_dir)
    if not exec_passed:
        return "incorrect", f"execution failed -- {exec_reason}", "execute", None, exec_passed, exec_returncode
    if not case.rubric:
        return "correct", f"execution passed -- {exec_reason} (no rubric on this case, skipping qualitative check)", "execute", None, exec_passed, exec_returncode
    verdict, reason, judge_model_name = grade_llm_judge(candidate, case, prompt_messages, reference, judge_client, dump_dir)
    return verdict, f"execution passed; judge: {reason}", "execute+llm_judge", judge_model_name, exec_passed, exec_returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help=f"port of the llama-server instance to test, on {PAXY_HOST}")
    parser.add_argument("--judge-port", type=int, default=None, help=f"port of the llama-server instance to use as judge, on {PAXY_HOST} (required if any case has a rubric)")
    parser.add_argument("--cases", default=str(REWRITE_DIR / "cases.jsonl"), help="path to cases.jsonl").completer = argcomplete.completers.FilesCompleter(allowednames=[".jsonl"])
    parser.add_argument("--max-tokens", type=int, default=4096,
                         help="cap on generated tokens; 0 or -1 removes the cap entirely "
                              "(useful when troubleshooting -- no more guessing a limit, hitting it, and rerunning)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save", action="store_true", help="write results JSON to results/<timestamp>-<model>.json")
    parser.add_argument("--only", default=None, help="only run the case with this id").completer = complete_case_ids
    parser.add_argument("--verbose", "-v", action="store_true", help="print which case is running as each one starts")
    parser.add_argument("--trace", action="store_true",
                         help="stream the completion and print tokens (content + reasoning_content) to stderr "
                              "as they arrive, instead of waiting for the full response -- also means a "
                              "client-side timeout still leaves you with whatever was streamed so far, dumped "
                              "to debug_dumps, instead of nothing")

    argcomplete.autocomplete(parser)
    import rich
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        cases = [c for c in cases if c.id == args.only]
        if not cases:
            sys.exit(f"no case with id {args.only!r}")

    model_client = make_client(args.port)

    needs_judge = any(c.rubric for c in cases)
    judge_client = None
    if needs_judge:
        if args.judge_port is None:
            sys.exit("one or more cases have a rubric (qualitative judge check) -- pass --judge-port, "
                      "or rerun with --only on a case without one")
        judge_client = make_client(args.judge_port)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump_dir = REWRITE_DIR / "debug_dumps" / run_ts

    results: list[Result] = []
    for case in cases:
        if args.verbose or args.trace:
            print(f"running {case.id}...", file=sys.stderr)

        prompt_messages, reference = load_trace_prompt_and_reference(case.source_trace)

        invoke_kwargs = {"temperature": args.temperature}
        if args.max_tokens not in (0, -1):
            invoke_kwargs["max_tokens"] = args.max_tokens
        ai_message, timings, stream_error = stream_completion(model_client, prompt_messages, invoke_kwargs, args.trace)
        candidate = (ai_message.content if ai_message else "") or ""
        finish_reason = ai_message.response_metadata.get("finish_reason") if ai_message else None
        model_name = (ai_message.response_metadata.get("model_name") if ai_message else None) or "unknown"
        completion_tokens = timings.get("predicted_n") if timings else None
        reasoning_content = (ai_message.additional_kwargs.get("reasoning_content") if ai_message else None) or ""

        dump_json(dump_dir, f"{case.id}.model", {
            "prompt_messages": prompt_messages,
            "content": candidate,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "model_name": model_name,
            "completion_tokens": completion_tokens,
            "timings": timings,
            "stream_error": str(stream_error) if stream_error else None,
        })

        if stream_error is not None:
            verdict = "incorrect"
            reason = (f"client error/timeout mid-stream: {stream_error} -- partial content dumped "
                      f"({len(candidate)} chars content, {len(reasoning_content)} chars reasoning)")
            graded_by = "execute"
            judge_model_name = None
            exec_passed, exec_returncode = False, None
        elif not candidate.strip() and finish_reason == "length":
            verdict = "incorrect"
            reason = (f"truncated before emitting any content ({completion_tokens} tokens spent, "
                      f"{'reasoning: ' + reasoning_content[:80] + '...' if reasoning_content else 'likely on reasoning/thinking'}"
                      f") -- try a higher --max-tokens, or --max-tokens -1 to remove the cap")
            graded_by = "execute"
            judge_model_name = None
            exec_passed, exec_returncode = False, None
        else:
            verdict, reason, graded_by, judge_model_name, exec_passed, exec_returncode = grade(
                candidate, case, prompt_messages, reference, judge_client, dump_dir)

        if finish_reason == "length" and candidate.strip():
            reason += " [NOTE: hit max_tokens -- may be mid-completion]"

        results.append(Result(
            id=case.id,
            verdict=verdict,
            reason=reason,
            graded_by=graded_by,
            reference=reference,
            candidate=candidate,
            exec_passed=exec_passed,
            exec_returncode=exec_returncode,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            model_name=model_name,
            judge_model_name=judge_model_name,
            constraint=case.constraint,
        ))

    model_names = {r.model_name for r in results}
    if len(model_names) > 1:
        rich.print(f"[bold white on red] WARNING [/] port {args.port} answered with more than one model name "
                   f"across cases: {model_names} -- was the server restarted with a different model mid-run?", file=sys.stderr)
    resolved_model = results[0].model_name if results else "unknown"

    judge_model_names = {r.judge_model_name for r in results if r.judge_model_name}
    if len(judge_model_names) > 1:
        rich.print(f"[bold white on red] WARNING [/] judge port {args.judge_port} answered with more than one model "
                   f"name across cases: {judge_model_names} -- was the server restarted with a different model mid-run?", file=sys.stderr)
    resolved_judge_model = next(iter(judge_model_names), None)

    print_report(resolved_model, args.port, resolved_judge_model, args.judge_port, results, dump_dir)

    if args.save:
        save_results(resolved_model, results)


def print_report(model: str, port: int, judge_model: str | None, judge_port: int | None, results: list[Result], dump_dir: Path):
    import rich
    icon = {"correct": "✅", "partial": "⚠️ ", "incorrect": "❌"}
    print()
    rich.print(f"rewrite eval -- model: [bold black on bright_yellow] {model} [/]  (port {port})")
    if judge_model:
        rich.print(f"              judge: [bold black on bright_cyan] {judge_model} [/]  (port {judge_port})")
    print(f"trace dumps: {dump_dir}")
    print("=" * 60)
    for r in results:
        constraint_tag = f", {r.constraint}" if r.constraint else ""
        exec_tag = "exec ok" if r.exec_passed else f"exec failed (rc={r.exec_returncode})"
        print(f"\n{icon.get(r.verdict, '?')} [{r.verdict}] {r.id}  ({r.graded_by}{constraint_tag}, {exec_tag}, {r.completion_tokens} tokens, finish={r.finish_reason})")
        print(f"   reference: {r.reference!r}")
        print(f"   candidate: {r.candidate!r}")
        if r.reason:
            print(f"   reason   : {r.reason}")

    total = len(results)
    correct = sum(1 for r in results if r.verdict == "correct")
    partial = sum(1 for r in results if r.verdict == "partial")
    incorrect = sum(1 for r in results if r.verdict == "incorrect")
    print("\n" + "-" * 60)
    print(f"{correct}/{total} correct, {partial}/{total} partial, {incorrect}/{total} incorrect")


def save_results(model: str, results: list[Result]):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out_path = REWRITE_DIR / "results" / f"{ts}-{safe_model}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "results": [asdict(r) for r in results]}, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
