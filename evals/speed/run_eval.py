#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Raw speed testing: boot a llama-server instance (over SSH, on a remote GPU
box) with a specific set of server flags, replay precanned prompts against
it, and capture llama-server's own reported timings -- prompt (prefill) and
predicted (decode) tokens/sec, MTP/speculative-decoding draft-accept rate,
plus client-observed wall-clock time-to-first-token and total latency.

This is a different animal from fim/rewrite/agent: those grade whether an
answer is *correct*. This eval never grades correctness at all -- a case's
completion is captured and shown, but the only thing that matters is the
timings. The point is finding good speculative-decoding params (n-max/
n-min/p-min/p-split, MTP vs non-MTP, etc) for a given model/hardware combo,
which today means hand-tweaking one flag at a time and eyeballing
tokens/sec -- this automates that sweep.

Each case in cases.jsonl points at a captured trace (same underlying format
as fim/rewrite/agent: `request_body.messages`/`tools`). The prompt is
everything up to the first assistant message (or `prompt_end_idx` if a case
sets one) -- same convention as evals/agent. A case's `category` says which
axis it's meant to isolate:

  - "generation": short prompt, long reference completion -- prompt
    processing is over almost instantly, so total time is dominated by
    decode speed (and, if speculative decoding is on, draft-accept rate).
  - "prefill": huge prompt, short/near-empty reference completion -- isolates
    prompt-processing throughput. Cold KV cache matters here (a case
    replayed against an already-warm cache measures nothing -- see "Cases
    and what to think about" in README.md).

Server lifecycle is managed over plain `ssh`/`scp` (no paramiko/fabric dep --
this machine's ~/.ssh/config already has host aliases with
ControlMaster/ControlPersist, so repeated ssh calls in one run reuse a single
connection). By default this script only ever touches a server *it* started
-- it will not kill anyone else's already-running llama-server unless you
pass --kill-existing explicitly. That matters because the GPU box this was
built against (build21) is often mid-session with a real model loaded for
manual use.

Usage:
    # first run against a host: boot a model, sweep nothing, just sanity check
    uv run --project .. python run_eval.py --host build21 --port 8097 \\
        --model ggml-org/Qwen3.5-0.8B-GGUF:BF16

    # with MTP on, specific draft params
    uv run --project .. python run_eval.py --host build21 --port 8097 \\
        --model ggml-org/Qwen3.5-0.8B-GGUF:BF16 \\
        --llama-args "--spec-type draft-mtp --spec-draft-n-max 4" --save

    # hit a server you already started (e.g. with --keep-server from a prior run)
    uv run --project .. python run_eval.py --host build21 --port 8097 --reuse-server

    # let this run kill any other llama-server on the host first (DESTRUCTIVE)
    uv run --project .. python run_eval.py --host build21 --port 8097 \\
        --model ggml-org/Qwen3.5-0.8B-GGUF:BF16 --kill-existing
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import argcomplete

if TYPE_CHECKING:
    from langchain_llama_server import ChatLlamaServer

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEED_DIR = Path(__file__).resolve().parent

# discovered by inspecting `ps aux` on build21 -- override with
# --llama-server-bin for any other host/build layout.
DEFAULT_LLAMA_SERVER_BIN = "/home/wes/repos/github/ggml-org/llama.cpp/build/bin/llama-server"

STARTUP_POLL_INTERVAL_SECONDS = 2.0
REMOTE_LOG_DIR = "/tmp/speed-eval-logs"


@dataclass
class Case:
    id: str
    source_trace: str
    category: str  # "generation" | "prefill" -- which axis this case isolates
    prompt_end_idx: int | None = None
    notes: str | None = None


@dataclass
class Result:
    id: str
    category: str
    model_name: str
    prompt_n: int | None
    predicted_n: int | None
    prompt_ms: float | None
    predicted_ms: float | None
    prompt_per_second: float | None
    predicted_per_second: float | None
    draft_n: int | None
    draft_n_accepted: int | None
    cache_n: int | None
    client_ttft_ms: float | None
    client_total_ms: float | None
    finish_reason: str | None
    completion_preview: str
    error: str | None = None


def run_ssh(host: str, remote_cmd: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    # force bash remote-side regardless of the account's login shell (build21's is
    # fish, which has no `$!`/POSIX-job-control semantics -- needed for backgrounding
    # llama-server and grabbing its pid in one shot). ssh joins all trailing argv
    # elements with spaces and hands the result to the remote login shell as ONE
    # string -- passing "bash", "-c", remote_cmd as separate argv elements does NOT
    # protect remote_cmd's spaces/&/> from being re-split by that remote shell (fish)
    # before bash ever sees them. Quoting remote_cmd ourselves and collapsing
    # everything into a single argv element sidesteps that.
    wrapped = f"bash -c {shlex.quote(remote_cmd)}"
    return subprocess.run(["ssh", host, wrapped], capture_output=True, text=True, timeout=timeout)


def check_gpus(host: str) -> None:
    proc = run_ssh(host, "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
                          "--format=csv,noheader")
    if proc.returncode != 0:
        print(f"warning: nvidia-smi on {host} failed: {proc.stderr.strip()}", file=sys.stderr)
        return
    print(f"GPUs on {host}:")
    for line in proc.stdout.strip().splitlines():
        print(f"  {line}")


def find_llama_server_procs(host: str) -> list[tuple[str, str]]:
    """Returns [(pid, full_cmdline), ...] for every llama-server process on host."""
    proc = run_ssh(host, "pgrep -af llama-server || true")
    procs = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        pid, _, cmdline = line.partition(" ")
        if "pgrep" in cmdline:
            # pgrep -f matches its own argv (which contains the search pattern) --
            # exclude self-matches rather than reporting the pgrep invocation itself
            # as a running llama-server.
            continue
        procs.append((pid, cmdline))
    return procs


def kill_procs(host: str, pids: list[str]) -> None:
    if not pids:
        return
    print(f"killing existing llama-server pid(s) on {host}: {', '.join(pids)}")
    run_ssh(host, f"kill {' '.join(pids)}")
    time.sleep(3)
    still_alive = [pid for pid, _ in find_llama_server_procs(host) if pid in pids]
    if still_alive:
        print(f"  {', '.join(still_alive)} still alive after SIGTERM, sending SIGKILL")
        run_ssh(host, f"kill -9 {' '.join(still_alive)}")
        time.sleep(1)


def start_llama_server(host: str, port: int, model: str, llama_server_bin: str, extra_args: str) -> tuple[str, str]:
    """Starts llama-server in the background over ssh. Returns (pid, remote_log_path)."""
    run_ssh(host, f"mkdir -p {REMOTE_LOG_DIR}")
    remote_log_path = f"{REMOTE_LOG_DIR}/{port}-{int(time.time())}.log"
    extra = f" {extra_args}" if extra_args else ""
    cmd = (
        f"nohup {shlex.quote(llama_server_bin)} --host 0.0.0.0 --port {port} "
        f"--flash-attn on --jinja -hf {shlex.quote(model)}{extra} "
        f"> {shlex.quote(remote_log_path)} 2>&1 < /dev/null & echo $!"
    )
    proc = run_ssh(host, cmd, timeout=15)
    if proc.returncode != 0:
        sys.exit(f"failed to launch llama-server on {host}: {proc.stderr.strip()}")
    pid = proc.stdout.strip()
    if not pid.isdigit():
        sys.exit(f"unexpected output launching llama-server on {host}: {proc.stdout!r} / {proc.stderr!r}")
    print(f"started llama-server on {host}:{port} (pid {pid}), logging to {host}:{remote_log_path}")
    return pid, remote_log_path


def wait_for_ready(host: str, port: int, timeout: float, host_alive_check_pid: str | None,
                    remote_log_path: str | None) -> None:
    import httpx
    url = f"http://{host}:{port}/v1/models"
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                print(f"server on {host}:{port} is ready ({time.monotonic() - start:.0f}s)")
                return
        except httpx.HTTPError:
            pass
        if host_alive_check_pid is not None:
            check = run_ssh(host, f"kill -0 {host_alive_check_pid} 2>/dev/null && echo alive || echo dead")
            if check.stdout.strip() == "dead":
                tail = ""
                if remote_log_path:
                    tail_proc = run_ssh(host, f"tail -n 40 {shlex.quote(remote_log_path)}")
                    tail = tail_proc.stdout
                sys.exit(f"llama-server process on {host} died before becoming ready. Log tail:\n{tail}")
        time.sleep(STARTUP_POLL_INTERVAL_SECONDS)
    sys.exit(f"timed out after {timeout:.0f}s waiting for {host}:{port} to become ready")


def port_in_use(procs: list[tuple[str, str]], port: int) -> str | None:
    """Returns the pid already bound to `port`, if any (whole-token match on
    "--port N" -- a naive substring check would false-positive 8097 against 80970).
    """
    needle = f"--port {port}"
    for pid, cmdline in procs:
        if f" {needle} " in f" {cmdline} ":
            return pid
    return None


def stop_server(host: str, pid: str) -> None:
    print(f"stopping llama-server on {host} (pid {pid})")
    run_ssh(host, f"kill {pid}")


def download_log(host: str, remote_log_path: str, local_dir: Path) -> Path | None:
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(remote_log_path).name
    proc = subprocess.run(["scp", f"{host}:{remote_log_path}", str(local_path)],
                           capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        print(f"warning: failed to scp log back from {host}: {proc.stderr.strip()}", file=sys.stderr)
        return None
    return local_path


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
    cases_path = Path(getattr(parsed_args, "cases", None) or (SPEED_DIR / "cases.jsonl"))
    try:
        return [c.id for c in load_cases(cases_path)]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def load_trace_prompt_and_tools(case: Case) -> tuple[list[dict], list[dict]]:
    trace = json.loads((REPO_ROOT / case.source_trace).read_text())
    messages = trace["request_body"]["messages"]
    tools = trace["request_body"].get("tools") or []
    end_idx = case.prompt_end_idx if case.prompt_end_idx is not None else next(
        i for i, m in enumerate(messages) if m["role"] == "assistant")
    return messages[:end_idx], tools


def run_case(client: ChatLlamaServer, case: Case) -> Result:
    prompt_messages, tools = load_trace_prompt_and_tools(case)
    invoke_kwargs = {}
    if tools:
        invoke_kwargs["tools"] = tools
    body = {"verbose": True}

    timings: dict | None = None
    model_name = ""
    finish_reason = None
    error = None
    content_parts: list[str] = []
    start = time.monotonic()
    ttft_s: float | None = None
    try:
        for chunk in client.stream(prompt_messages, extra_body=body, **invoke_kwargs):
            if ttft_s is None:
                ttft_s = time.monotonic() - start
            debug_info = getattr(chunk, "debug", None)
            if debug_info is not None and getattr(debug_info, "timings", None):
                timings = debug_info.timings
            response_metadata = getattr(chunk, "response_metadata", None) or {}
            if response_metadata.get("model_name"):
                model_name = response_metadata["model_name"]
            if response_metadata.get("finish_reason"):
                finish_reason = response_metadata["finish_reason"]
            if chunk.content:
                content_parts.append(chunk.content)
    except Exception as e:  # noqa: BLE001 -- speed run should keep going past one bad case
        error = str(e)
    total_s = time.monotonic() - start

    timings = timings or {}
    preview = "".join(content_parts)[:200]
    return Result(
        id=case.id,
        category=case.category,
        model_name=model_name,
        prompt_n=timings.get("prompt_n"),
        predicted_n=timings.get("predicted_n"),
        prompt_ms=timings.get("prompt_ms"),
        predicted_ms=timings.get("predicted_ms"),
        prompt_per_second=timings.get("prompt_per_second"),
        predicted_per_second=timings.get("predicted_per_second"),
        draft_n=timings.get("draft_n"),
        draft_n_accepted=timings.get("draft_n_accepted"),
        cache_n=timings.get("cache_n"),
        client_ttft_ms=ttft_s * 1000 if ttft_s is not None else None,
        client_total_ms=total_s * 1000,
        finish_reason=finish_reason,
        completion_preview=preview,
        error=error,
    )


def print_report(host: str, port: int, model: str, llama_args: str, results: list[Result]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold]{host}:{port}[/bold]  model={model!r}  llama_args={llama_args!r}\n")
    table = Table(show_lines=False)
    for col in ("id", "cat", "prompt_n", "prompt_tok/s", "pred_n", "pred_tok/s",
                "draft_n", "draft_acc%", "ttft_ms", "total_ms"):
        table.add_column(col)
    for r in results:
        if r.error:
            table.add_row(r.id, r.category, "-", "-", "-", "-", "-", "-", "-", f"ERROR: {r.error}")
            continue
        accept_pct = (f"{100 * r.draft_n_accepted / r.draft_n:.0f}%"
                      if r.draft_n_accepted is not None and r.draft_n else "-")
        table.add_row(
            r.id, r.category,
            str(r.prompt_n) if r.prompt_n is not None else "-",
            f"{r.prompt_per_second:.0f}" if r.prompt_per_second is not None else "-",
            str(r.predicted_n) if r.predicted_n is not None else "-",
            f"{r.predicted_per_second:.0f}" if r.predicted_per_second is not None else "-",
            str(r.draft_n) if r.draft_n is not None else "-",
            accept_pct,
            f"{r.client_ttft_ms:.0f}" if r.client_ttft_ms is not None else "-",
            f"{r.client_total_ms:.0f}" if r.client_total_ms is not None else "-",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="ssh host alias (also used as the HTTP host) -- e.g. build21, paxy.lan")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", help="-hf spec passed straight to llama-server, e.g. ggml-org/Qwen3.5-0.8B-GGUF:BF16 "
                                        "(required unless --reuse-server)")
    parser.add_argument("--llama-args", default="", help="extra raw flags appended verbatim to the llama-server "
                                                           "invocation, e.g. \"--spec-type draft-mtp --spec-draft-n-max 4\"")
    parser.add_argument("--llama-server-bin", default=DEFAULT_LLAMA_SERVER_BIN)
    parser.add_argument("--kill-existing", action="store_true",
                         help="DESTRUCTIVE: kill any other llama-server process found on --host before starting. "
                              "Off by default -- this host may be mid-session with a real model loaded for manual use.")
    parser.add_argument("--reuse-server", action="store_true",
                         help="skip GPU-check/start entirely, assume a server is already listening at host:port "
                              "(e.g. left running by a prior --keep-server run)")
    parser.add_argument("--keep-server", action="store_true",
                         help="don't stop the server this run started once cases finish")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--only", default=None, help="only run the case with this id").completer = complete_case_ids
    parser.add_argument("--save", action="store_true", help="write a JSON result file under results/")
    parser.add_argument("--log-dir", type=Path, default=SPEED_DIR / "logs")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not args.reuse_server and not args.model:
        parser.error("--model is required unless --reuse-server is set")

    cases_path = args.cases or (SPEED_DIR / "cases.jsonl")
    cases = load_cases(cases_path)
    if args.only:
        cases = [c for c in cases if c.id == args.only]
        if not cases:
            sys.exit(f"no case matched --only {args.only}")

    from langchain_llama_server import ChatLlamaServer

    pid = None
    remote_log_path = None
    try:
        if not args.reuse_server:
            check_gpus(args.host)
            existing = find_llama_server_procs(args.host)
            if existing:
                print(f"found {len(existing)} existing llama-server process(es) on {args.host}:")
                for p, cmdline in existing:
                    print(f"  pid {p}: {cmdline}")
                if args.kill_existing:
                    kill_procs(args.host, [p for p, _ in existing])
                    existing = find_llama_server_procs(args.host)
                else:
                    print("  --kill-existing not set, leaving them running (they'll share the GPU with the new server)")
            # a leftover/orphaned server already on --port (e.g. from a prior run whose
            # ssh session got cut before its own cleanup ran -- nohup keeps it alive
            # independent of that ssh connection) would otherwise cause a silent race:
            # our new server fails to bind and dies, but wait_for_ready still gets a
            # 200 from the OLD process and happily reports its numbers as if they were
            # this run's. Fail loud instead.
            collision_pid = port_in_use(existing, args.port)
            if collision_pid is not None:
                sys.exit(f"port {args.port} on {args.host} is already bound by pid {collision_pid} -- "
                          f"pick a different --port, pass --reuse-server if that's the server you meant to hit, "
                          f"or --kill-existing to clear it first")
            pid, remote_log_path = start_llama_server(args.host, args.port, args.model, args.llama_server_bin, args.llama_args)
            wait_for_ready(args.host, args.port, args.startup_timeout, pid, remote_log_path)

        client = ChatLlamaServer(base_url=f"http://{args.host}:{args.port}/v1", api_key="none", timeout=300)
        results = [run_case(client, case) for case in cases]
        print_report(args.host, args.port, args.model or "(reused server)", args.llama_args, results)

        if remote_log_path:
            local_log = download_log(args.host, remote_log_path, args.log_dir)
            if local_log:
                print(f"downloaded server log to {local_log}")

        if args.save:
            out_dir = SPEED_DIR / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            model_slug = (args.model or "reused").replace("/", "_").replace(":", "_")
            out_path = out_dir / f"{ts}-{args.host}-{args.port}-{model_slug}.json"
            out_path.write_text(json.dumps({
                "host": args.host,
                "port": args.port,
                "model": args.model,
                "llama_args": args.llama_args,
                "results": [asdict(r) for r in results],
            }, indent=2))
            print(f"saved results to {out_path}")
    finally:
        if pid is not None and not args.keep_server and not args.reuse_server:
            stop_server(args.host, pid)


if __name__ == "__main__":
    main()
