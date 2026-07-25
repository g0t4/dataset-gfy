#!/usr/bin/env python3
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
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_llama_server import ChatLlamaServer

REPO_ROOT = Path(__file__).resolve().parents[2]
FIM_DIR = Path(__file__).resolve().parent

PAXY_HOST = "paxy.lan"


def make_client(port: int) -> ChatLlamaServer:
    return ChatLlamaServer(base_url=f"http://{PAXY_HOST}:{port}/v1", api_key="none", timeout=120)


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def load_trace_prompt_and_expected(source_trace: str) -> tuple[list[dict], str]:
    trace_path = REPO_ROOT / source_trace
    trace = json.loads(trace_path.read_text())
    messages = trace["request_body"]["messages"]
    prompt_messages = messages[:-1]
    expected = messages[-1]["content"]
    return prompt_messages, expected


def normalize(text: str) -> str:
    return text.strip()


def grade_exact_normalized(candidate: str, case: dict) -> tuple[str, str]:
    candidate_n = normalize(candidate)
    accepted = [normalize(a) for a in case["accepted"]]
    if candidate_n in accepted:
        return "correct", "exact match"
    return "incorrect", f"expected one of {case['accepted']!r}"


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


def grade_llm_judge(candidate: str, case: dict, prompt_messages: list[dict], expected: str, judge_client: ChatLlamaServer) -> tuple[str, str]:
    fim_task = prompt_messages[-1]["content"]
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        fim_task=fim_task,
        expected=expected,
        rubric=case["rubric"],
        candidate=candidate,
    )
    ai_message = judge_client.invoke([{"role": "user", "content": judge_prompt}], temperature=0)
    raw = (ai_message.content or "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return "incorrect", f"judge did not return JSON: {raw!r}"
    try:
        parsed = json.loads(match.group(0))
        verdict = parsed.get("verdict", "incorrect")
        reason = parsed.get("reason", "")
        if verdict not in ("correct", "partial", "incorrect"):
            return "incorrect", f"judge returned unknown verdict {verdict!r}"
        return verdict, reason
    except json.JSONDecodeError:
        return "incorrect", f"judge returned unparseable JSON: {raw!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help=f"port of the llama-server instance to test, on {PAXY_HOST}")
    parser.add_argument("--judge-port", type=int, default=None, help=f"port of the llama-server instance to use as judge, on {PAXY_HOST} (required if any case uses grader:llm_judge)")
    parser.add_argument("--cases", default=str(FIM_DIR / "cases.jsonl"), help="path to cases.jsonl")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save", action="store_true", help="write results JSON to results/<timestamp>-<model>.json")
    parser.add_argument("--only", default=None, help="only run the case with this id")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            sys.exit(f"no case with id {args.only!r}")

    model_client = make_client(args.port)

    needs_judge = any(c["grader"] == "llm_judge" for c in cases)
    judge_client = None
    if needs_judge:
        if args.judge_port is None:
            sys.exit("one or more cases need grader:llm_judge -- pass --judge-port, "
                      "or rerun with --only on a non-judge case")
        judge_client = make_client(args.judge_port)

    results = []
    for case in cases:
        prompt_messages, expected = load_trace_prompt_and_expected(case["source_trace"])

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
        elif case["grader"] == "exact_normalized":
            verdict, reason = grade_exact_normalized(candidate, case)
        elif case["grader"] == "llm_judge":
            verdict, reason = grade_llm_judge(candidate, case, prompt_messages, expected, judge_client)
        else:
            verdict, reason = "incorrect", f"unknown grader {case['grader']!r}"

        if finish_reason == "length" and candidate.strip():
            reason += " [NOTE: hit max_tokens -- may be mid-completion]"

        results.append({
            "id": case["id"],
            "grader": case["grader"],
            "verdict": verdict,
            "reason": reason,
            "expected": expected,
            "candidate": candidate,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "model_name": model_name,
        })

    model_names = {r["model_name"] for r in results}
    if len(model_names) > 1:
        print(f"WARNING: port {args.port} answered with more than one model name across cases: {model_names} "
              f"-- was the server restarted with a different model mid-run?", file=sys.stderr)
    resolved_model = results[0]["model_name"] if results else "unknown"

    print_report(resolved_model, args.port, results)

    if args.save:
        save_results(resolved_model, results)


def print_report(model: str, port: int, results: list[dict]):
    icon = {"correct": "✅", "partial": "⚠️ ", "incorrect": "❌"}
    print(f"\nFIM eval -- model: {model} (port {port})\n" + "=" * 60)
    for r in results:
        print(f"\n{icon.get(r['verdict'], '?')} [{r['verdict']}] {r['id']}  ({r['grader']}, {r['completion_tokens']} tokens, finish={r['finish_reason']})")
        print(f"   expected : {r['expected']!r}")
        print(f"   got      : {r['candidate']!r}")
        if r["reason"]:
            print(f"   reason   : {r['reason']}")

    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partial")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    print("\n" + "-" * 60)
    print(f"{correct}/{total} correct, {partial}/{total} partial, {incorrect}/{total} incorrect")


def save_results(model: str, results: list[dict]):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out_path = FIM_DIR / "results" / f"{ts}-{safe_model}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "results": results}, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
