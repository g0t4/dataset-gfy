#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Run FIM eval cases (cases.jsonl) against a model under test, grade the
completions, and print a report.

Each case in cases.jsonl points at a trace file captured from real usage.
The trace's messages[:-1] is the exact prompt that was sent; messages[-1]
(the assistant message) is the human-approved completion that was actually
shown/accepted, used here as the reference answer.

Model selection is by port, not name: my llama-server instances are one
model per port (static allocation), so which model you're testing is
entirely a function of which port you point at. The model name itself is
never sent in the request -- it's read back from the completion response
afterwards, so the report/saved results always reflect whatever model
actually answered (not whatever you assumed was running on that port).

Usage:
    uv run --project .. python run_eval.py --port 8012
    uv run --project .. python run_eval.py --port 8012 --judge-port 8013 --save
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import argcomplete

if TYPE_CHECKING:
    # deferred for real (see make_client) -- keeps `--port 80<TAB>` fast by not
    # eagerly importing langchain/openai (~200-300ms) just to generate completions
    from langchain_llama_server import ChatLlamaServer

REPO_ROOT = Path(__file__).resolve().parents[2]
FIM_DIR = Path(__file__).resolve().parent

PAXY_HOST = "paxy.lan"


@dataclass
class PartialAccepted:
    value: str
    reason: str


@dataclass
class Case:
    id: str
    source_trace: str
    language: str
    grader: str
    accepted: list[str] | None = None
    partial_accepted: list[PartialAccepted] | None = None
    rubric: str | None = None
    # answer-constraint tightness, for slicing reports separately from verdict:
    #   "sanity" = essentially one token/answer, near-zero domain knowledge needed
    #   "tight"  = one (or a small enumerable set of) correct answer(s), but real
    #              domain/context knowledge required to find it
    #   "open"   = genuine judgment call -- multiple stylistically different but
    #              valid answers (e.g. naming choices)
    constraint: str | None = None
    notes: str | None = None


@dataclass
class Result:
    id: str
    grader: str
    verdict: str
    reason: str
    expected: str
    candidate: str
    completion_tokens: int | None
    finish_reason: str | None
    model_name: str
    judge_model_name: str | None = None
    constraint: str | None = None


def make_client(port: int) -> ChatLlamaServer:
    from langchain_llama_server import ChatLlamaServer
    return ChatLlamaServer(base_url=f"http://{PAXY_HOST}:{port}/v1", api_key="none", timeout=120)


def load_cases(path: Path) -> list[Case]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            partial_accepted = data.pop("partial_accepted", None)
            case = Case(**data)
            if partial_accepted:
                case.partial_accepted = [PartialAccepted(**p) for p in partial_accepted]
            cases.append(case)
    return cases


def complete_case_ids(prefix: str, parsed_args: argparse.Namespace, **kwargs) -> list[str]:
    cases_path = Path(getattr(parsed_args, "cases", None) or (FIM_DIR / "cases.jsonl"))
    try:
        return [c.id for c in load_cases(cases_path)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def load_trace_prompt_and_expected(source_trace: str) -> tuple[list[dict], str]:
    trace_path = REPO_ROOT / source_trace
    trace = json.loads(trace_path.read_text())
    messages = trace["request_body"]["messages"]
    prompt_messages = messages[:-1]
    expected = messages[-1]["content"]
    return prompt_messages, expected


DEFAULT_CURSOR_MARKER = "<|fim_middle|>"


def swap_cursor_marker(prompt_messages: list[dict], marker: str) -> list[dict]:
    if marker == DEFAULT_CURSOR_MARKER:
        return prompt_messages
    # NOTE: naive global string-replace across every message's content -- this
    # correctly swaps both places the marker normally shows up (the "replace
    # <|fim_middle|>:" instruction line, and the actual cursor position in the
    # code block). BUT: if a trace's own surrounding code happens to contain a
    # literal, real occurrence of DEFAULT_CURSOR_MARKER as source text (e.g. a
    # FIM captured from within ask-openai.nvim's own codebase, which legitimately
    # references this marker string as data), this will also swap that
    # occurrence, subtly corrupting the "ground truth" code shown to the model.
    # Not handled yet -- revisit (e.g. flag/skip such cases for this sweep) if
    # it actually produces a surprising/broken result on one of those cases.
    return [{**m, "content": m["content"].replace(DEFAULT_CURSOR_MARKER, marker)} for m in prompt_messages]


def normalize(text: str) -> str:
    return text.strip()


def grade_exact_normalized(candidate: str, case: Case) -> tuple[str, str]:
    candidate_n = normalize(candidate)
    accepted = [normalize(a) for a in (case.accepted or [])]
    if candidate_n in accepted:
        return "correct", "exact match"
    for partial in case.partial_accepted or []:
        if candidate_n == normalize(partial.value):
            return "partial", partial.reason
    return "incorrect", f"expected one of {case.accepted!r}"


JUDGE_PROMPT_TEMPLATE = """You are grading a code-completion (fill-in-the-middle) suggestion.

Task given to the model under test (surrounding code, cursor marked by <|fim_middle|>):
---
{fim_task}
---

Reference (human-approved) completion for this exact spot -- exact text between
the markers, the markers themselves and any surrounding quotes are NOT part of it:
<<<REFERENCE>>>
{expected}
<<<END REFERENCE>>>

Rubric for this specific case:
{rubric}

Candidate completion to grade -- exact text between the markers, the markers
themselves and any surrounding quotes are NOT part of it:
<<<CANDIDATE>>>
{candidate}
<<<END CANDIDATE>>>

Respond with ONLY a JSON object, no markdown fences, no extra commentary:
{{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}
"""


def grade_llm_judge(candidate: str, case: Case, prompt_messages: list[dict], expected: str, judge_client: ChatLlamaServer) -> tuple[str, str, str]:
    fim_task = prompt_messages[-1]["content"]
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        fim_task=fim_task,
        expected=expected,
        rubric=case.rubric,
        candidate=candidate,
    )
    ai_message = judge_client.invoke([{"role": "user", "content": judge_prompt}], temperature=0)
    judge_model_name = ai_message.response_metadata.get("model_name") or "unknown"
    raw = (ai_message.content or "").strip()
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


def grade(candidate: str, case: Case, prompt_messages: list[dict], expected: str, judge_client: ChatLlamaServer | None) -> tuple[str, str, str | None]:
    if case.grader == "exact_normalized":
        verdict, reason = grade_exact_normalized(candidate, case)
        return verdict, reason, None
    if case.grader == "llm_judge":
        verdict, reason = grade_exact_normalized(candidate, case)
        if verdict == "correct":
            return verdict, f"{reason} (skipped judge -- exact match in accepted list)", None
        if verdict == "partial":
            return verdict, f"{reason} (skipped judge -- matched partial_accepted list)", None
        verdict, reason, judge_model_name = grade_llm_judge(candidate, case, prompt_messages, expected, judge_client)
        return verdict, reason, judge_model_name
    return "incorrect", f"unknown grader {case.grader!r}", None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help=f"port of the llama-server instance to test, on {PAXY_HOST}")
    parser.add_argument("--judge-port", type=int, default=None, help=f"port of the llama-server instance to use as judge, on {PAXY_HOST} (required if any case uses grader:llm_judge)")
    parser.add_argument("--cases", default=str(FIM_DIR / "cases.jsonl"), help="path to cases.jsonl").completer = argcomplete.completers.FilesCompleter(allowednames=[".jsonl"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save", action="store_true", help="write results JSON to results/<timestamp>-<model>.json")
    parser.add_argument("--only", default=None, help="only run the case with this id").completer = complete_case_ids
    parser.add_argument("--cursor-marker", default=DEFAULT_CURSOR_MARKER,
                         help=f"cursor marker to use instead of the trace's original {DEFAULT_CURSOR_MARKER!r} "
                              f"(default), to sweep prompt-format sensitivity, e.g. --cursor-marker '<|CURSOR|>'")

    argcomplete.autocomplete(parser)
    import rich
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        cases = [c for c in cases if c.id == args.only]
        if not cases:
            sys.exit(f"no case with id {args.only!r}")

    model_client = make_client(args.port)

    needs_judge = any(c.grader == "llm_judge" for c in cases)
    judge_client = None
    if needs_judge:
        if args.judge_port is None:
            sys.exit("one or more cases need grader:llm_judge -- pass --judge-port, "
                      "or rerun with --only on a non-judge case")
        judge_client = make_client(args.judge_port)

    results: list[Result] = []
    for case in cases:
        prompt_messages, expected = load_trace_prompt_and_expected(case.source_trace)
        prompt_messages = swap_cursor_marker(prompt_messages, args.cursor_marker)

        ai_message = model_client.invoke(prompt_messages, temperature=args.temperature, max_tokens=args.max_tokens)
        candidate = ai_message.content or ""
        finish_reason = ai_message.response_metadata.get("finish_reason")
        model_name = ai_message.response_metadata.get("model_name") or "unknown"
        usage = ai_message.usage_metadata or {}
        completion_tokens = usage.get("output_tokens")
        reasoning_content = ai_message.additional_kwargs.get("reasoning_content") or ""

        if not candidate.strip() and finish_reason == "length":
            verdict = "incorrect"
            reason = (f"truncated before emitting any content ({completion_tokens} tokens spent, "
                      f"{'reasoning: ' + reasoning_content[:80] + '...' if reasoning_content else 'likely on reasoning/thinking'}"
                      f") -- try a higher --max-tokens")
            judge_model_name = None
        else:
            verdict, reason, judge_model_name = grade(candidate, case, prompt_messages, expected, judge_client)

        if finish_reason == "length" and candidate.strip():
            reason += " [NOTE: hit max_tokens -- may be mid-completion]"

        results.append(Result(
            id=case.id,
            grader=case.grader,
            verdict=verdict,
            reason=reason,
            expected=expected,
            candidate=candidate,
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

    print_report(resolved_model, args.port, resolved_judge_model, args.judge_port, args.cursor_marker, results)

    if args.save:
        save_results(resolved_model, args.cursor_marker, results)


def print_report(model: str, port: int, judge_model: str | None, judge_port: int | None, cursor_marker: str, results: list[Result]):
    import rich
    icon = {"correct": "✅", "partial": "⚠️ ", "incorrect": "❌"}
    print()
    rich.print(f"FIM eval -- model: [bold black on bright_yellow] {model} [/]  (port {port})")
    if judge_model:
        rich.print(f"           judge: [bold black on bright_cyan] {judge_model} [/]  (port {judge_port})")
    if cursor_marker != DEFAULT_CURSOR_MARKER:
        rich.print(f"    cursor marker: [bold black on bright_magenta] {cursor_marker} [/]  (swept from default {DEFAULT_CURSOR_MARKER!r})")
    print("=" * 60)
    for r in results:
        constraint_tag = f", {r.constraint}" if r.constraint else ""
        print(f"\n{icon.get(r.verdict, '?')} [{r.verdict}] {r.id}  ({r.grader}{constraint_tag}, {r.completion_tokens} tokens, finish={r.finish_reason})")
        print(f"   expected : {r.expected!r}")
        print(f"   got      : {r.candidate!r}")
        if r.reason:
            print(f"   reason   : {r.reason}")

    total = len(results)
    correct = sum(1 for r in results if r.verdict == "correct")
    partial = sum(1 for r in results if r.verdict == "partial")
    incorrect = sum(1 for r in results if r.verdict == "incorrect")
    print("\n" + "-" * 60)
    print(f"{correct}/{total} correct, {partial}/{total} partial, {incorrect}/{total} incorrect")


def save_results(model: str, cursor_marker: str, results: list[Result]):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out_path = FIM_DIR / "results" / f"{ts}-{safe_model}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "cursor_marker": cursor_marker, "results": [asdict(r) for r in results]}, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
