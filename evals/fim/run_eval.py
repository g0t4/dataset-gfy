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
PAXY_HARDWARE = "2x NVIDIA RTX PRO 6000 Blackwell 96GB"


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
    attempt: int = 1


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


def grade_llm_judge(candidate: str, case: Case, prompt_messages: list[dict], expected: str, judge_client: ChatLlamaServer, dump_dir: Path, case_id: str) -> tuple[str, str, str]:
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
    dump_json(dump_dir, f"{case_id}.judge", {
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


def stream_completion(model_client: ChatLlamaServer, prompt_messages: list[dict], invoke_kwargs: dict, trace: bool, extra_body: dict | None = None):
    """Stream a completion, optionally echoing tokens to stderr as they arrive.

    Passes extra_body={"verbose": True, ...} so llama-server attaches its own
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
    body = {"verbose": True, **(extra_body or {})}
    try:
        for chunk in model_client.stream(prompt_messages, extra_body=body, **invoke_kwargs):
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


def grade(candidate: str, case: Case, prompt_messages: list[dict], expected: str, judge_client: ChatLlamaServer | None, dump_dir: Path, dump_id: str) -> tuple[str, str, str | None]:
    if case.grader == "exact_normalized":
        verdict, reason = grade_exact_normalized(candidate, case)
        return verdict, reason, None
    if case.grader == "llm_judge":
        verdict, reason = grade_exact_normalized(candidate, case)
        if verdict == "correct":
            return verdict, f"{reason} (skipped judge -- exact match in accepted list)", None
        if verdict == "partial":
            return verdict, f"{reason} (skipped judge -- matched partial_accepted list)", None
        verdict, reason, judge_model_name = grade_llm_judge(candidate, case, prompt_messages, expected, judge_client, dump_dir, dump_id)
        return verdict, reason, judge_model_name
    return "incorrect", f"unknown grader {case.grader!r}", None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help=f"port of the llama-server instance to test, on {PAXY_HOST}")
    parser.add_argument("--judge-port", type=int, default=None, help=f"port of the llama-server instance to use as judge, on {PAXY_HOST} (required if any case uses grader:llm_judge)")
    parser.add_argument("--cases", default=str(FIM_DIR / "cases.jsonl"), help="path to cases.jsonl").completer = argcomplete.completers.FilesCompleter(allowednames=[".jsonl"])
    parser.add_argument("--max-tokens", type=int, default=4096,
                         help="cap on generated tokens; 0 or -1 removes the cap entirely "
                              "(useful when troubleshooting -- no more guessing a limit, hitting it, and rerunning)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save", action="store_true", help="write results JSON to results/<timestamp>-<model>.json")
    parser.add_argument("--only", default=None, help="only run the case with this id").completer = complete_case_ids
    parser.add_argument("--cursor-marker", default=DEFAULT_CURSOR_MARKER,
                         help=f"cursor marker to use instead of the trace's original {DEFAULT_CURSOR_MARKER!r} "
                              f"(default), to sweep prompt-format sensitivity, e.g. --cursor-marker '<|CURSOR|>'")
    parser.add_argument("--verbose", "-v", action="store_true",
                         help="print each case's full result (verdict/expected/got/reason) as soon as it's "
                              "graded, instead of only at the end -- lets you catch something amiss or Ctrl-C "
                              "early without waiting for the whole run")
    parser.add_argument("--trace", action="store_true",
                         help="stream the completion and print tokens (content + reasoning_content) to stderr "
                              "as they arrive, instead of waiting for the full response -- also means a "
                              "client-side timeout still leaves you with whatever was streamed so far, dumped "
                              "to debug_dumps, instead of nothing")
    parser.add_argument("--reasoning", choices=["on", "off"], default="on",
                         help="toggle thinking/reasoning on the model under test (default: on). 'off' sends "
                              "chat_template_kwargs={enable_thinking: false}, which Qwen3-family templates honor "
                              "to skip the <think> block entirely -- lets you A/B the same cases with reasoning "
                              "on vs off. No effect on models/templates that don't support the toggle (llama-server "
                              "just ignores the unrecognized template kwarg). Only affects the model under test, "
                              "never the judge.")
    parser.add_argument("--repeat", type=int, default=1,
                         help="run each case N times and report a per-case stability breakdown (correct/partial/"
                              "incorrect counts, flagged FLAKY if the verdict isn't the same every time) -- use "
                              "this to tell whether a result (or an on/off comparison) is real or just sampling "
                              "noise before trusting it, even at --temperature 0 (llama-server batching/MTP/"
                              "speculative decoding can still make runs non-deterministic)")
    parser.add_argument("--hardware", default=PAXY_HARDWARE,
                         help=f"free-text note on what {PAXY_HOST} is running on, stamped into the report header "
                              f"and --save output -- GPU/backend numerics (and other processes sharing the same "
                              f"GPU concurrently) can affect run-to-run determinism, so it's worth recording "
                              f"alongside results (default: {PAXY_HARDWARE!r})")

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

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump_dir = FIM_DIR / "debug_dumps" / run_ts

    results: list[Result] = []
    for case in cases:
        prompt_messages, expected = load_trace_prompt_and_expected(case.source_trace)
        prompt_messages = swap_cursor_marker(prompt_messages, args.cursor_marker)

        for attempt in range(1, args.repeat + 1):
            dump_id = case.id if args.repeat == 1 else f"{case.id}.attempt{attempt}"
            if args.verbose or args.trace:
                attempt_tag = f" (attempt {attempt}/{args.repeat})" if args.repeat > 1 else ""
                print(f"running {case.id}{attempt_tag}...", file=sys.stderr)

            invoke_kwargs = {"temperature": args.temperature}
            if args.max_tokens not in (0, -1):
                invoke_kwargs["max_tokens"] = args.max_tokens
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.reasoning == "off" else None
            ai_message, timings, stream_error = stream_completion(model_client, prompt_messages, invoke_kwargs, args.trace, extra_body)
            candidate = (ai_message.content if ai_message else "") or ""
            finish_reason = ai_message.response_metadata.get("finish_reason") if ai_message else None
            model_name = (ai_message.response_metadata.get("model_name") if ai_message else None) or "unknown"
            completion_tokens = timings.get("predicted_n") if timings else None
            reasoning_content = (ai_message.additional_kwargs.get("reasoning_content") if ai_message else None) or ""

            dump_json(dump_dir, f"{dump_id}.model", {
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
                judge_model_name = None
            elif not candidate.strip() and finish_reason == "length":
                verdict = "incorrect"
                reason = (f"truncated before emitting any content ({completion_tokens} tokens spent, "
                          f"{'reasoning: ' + reasoning_content[:80] + '...' if reasoning_content else 'likely on reasoning/thinking'}"
                          f") -- try a higher --max-tokens, or --max-tokens -1 to remove the cap")
                judge_model_name = None
            else:
                verdict, reason, judge_model_name = grade(candidate, case, prompt_messages, expected, judge_client, dump_dir, dump_id)

            if finish_reason == "length" and candidate.strip():
                reason += " [NOTE: hit max_tokens -- may be mid-completion]"

            result = Result(
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
                attempt=attempt,
            )
            results.append(result)
            if args.verbose:
                print_result_block(result, show_attempt=args.repeat > 1)

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

    print_report(resolved_model, args.port, resolved_judge_model, args.judge_port, args.cursor_marker, args.reasoning, args.hardware, results, dump_dir)

    if args.repeat > 1:
        print_stability_report(results, args.repeat)

    if args.save:
        save_results(resolved_model, args.cursor_marker, args.reasoning, args.hardware, results)


RESULT_ICON = {"correct": "✅", "partial": "⚠️ ", "incorrect": "❌"}


def print_result_block(r: Result, show_attempt: bool = False) -> None:
    constraint_tag = f", {r.constraint}" if r.constraint else ""
    attempt_tag = f"  [attempt {r.attempt}]" if show_attempt else ""
    print(f"\n{RESULT_ICON.get(r.verdict, '?')} [{r.verdict}] {r.id}  ({r.grader}{constraint_tag}, {r.completion_tokens} tokens, finish={r.finish_reason}){attempt_tag}")
    print(f"   expected : {r.expected!r}")
    print(f"   got      : {r.candidate!r}")
    if r.reason:
        print(f"   reason   : {r.reason}")


def print_report(model: str, port: int, judge_model: str | None, judge_port: int | None, cursor_marker: str, reasoning: str, hardware: str, results: list[Result], dump_dir: Path):
    import rich
    print()
    rich.print(f"FIM eval -- model: [bold black on bright_yellow] {model} [/]  (port {port})")
    if judge_model:
        rich.print(f"           judge: [bold black on bright_cyan] {judge_model} [/]  (port {judge_port})")
    if cursor_marker != DEFAULT_CURSOR_MARKER:
        rich.print(f"    cursor marker: [bold black on bright_magenta] {cursor_marker} [/]  (swept from default {DEFAULT_CURSOR_MARKER!r})")
    if reasoning == "off":
        rich.print(f"        reasoning: [bold black on bright_red] off [/]  (chat_template_kwargs.enable_thinking=false)")
    rich.print(f"         hardware: [bold black on bright_green] {hardware} [/]")
    print(f"trace dumps: {dump_dir}")
    print("=" * 60)
    show_attempt = len({r.attempt for r in results}) > 1
    for r in results:
        print_result_block(r, show_attempt=show_attempt)

    total = len(results)
    correct = sum(1 for r in results if r.verdict == "correct")
    partial = sum(1 for r in results if r.verdict == "partial")
    incorrect = sum(1 for r in results if r.verdict == "incorrect")
    print("\n" + "-" * 60)
    print(f"{correct}/{total} correct, {partial}/{total} partial, {incorrect}/{total} incorrect"
          + (" (across all attempts)" if show_attempt else ""))


def print_stability_report(results: list[Result], repeat: int) -> None:
    from collections import defaultdict
    by_case: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_case[r.id].append(r)

    print(f"\nStability across {repeat} attempts/case:")
    print("=" * 60)
    flaky_count = 0
    for case_id, rs in by_case.items():
        n = len(rs)
        c = sum(1 for r in rs if r.verdict == "correct")
        p = sum(1 for r in rs if r.verdict == "partial")
        i = sum(1 for r in rs if r.verdict == "incorrect")
        flaky = len({r.verdict for r in rs}) > 1
        flag = "  ⚠️  FLAKY -- verdict changed across attempts" if flaky else ""
        if flaky:
            flaky_count += 1
        print(f"  {case_id:<45} {c}/{n} correct, {p}/{n} partial, {i}/{n} incorrect{flag}")
    print("-" * 60)
    if flaky_count:
        print(f"{flaky_count}/{len(by_case)} cases had a verdict that changed across attempts -- "
              f"treat single-run comparisons (e.g. --reasoning on vs off) on those cases with caution.")
    else:
        print(f"all {len(by_case)} cases gave the same verdict on every attempt -- single-run results look stable.")


def save_results(model: str, cursor_marker: str, reasoning: str, hardware: str, results: list[Result]):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    suffix = "-noreasoning" if reasoning == "off" else ""
    out_path = FIM_DIR / "results" / f"{ts}-{safe_model}{suffix}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "cursor_marker": cursor_marker, "reasoning": reasoning, "hardware": hardware, "results": [asdict(r) for r in results]}, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
