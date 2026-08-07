from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from relflow.architecture.root import Model
from relflow.data.iterables import encode
from relflow.structs.enums import Strata
from relflow.structs.experiment import Hyperparameters

SERVER_SCRIPT = """
import sys

from relflow.inference.deployment import Deployment

checkpoint = sys.argv[1]
port = int(sys.argv[2])
max_batch_size = int(sys.argv[3])
batch_timeout = float(sys.argv[4])
workers = int(sys.argv[5])
monitor_queries = sys.argv[6].lower() == "true"
json_backend = sys.argv[7]

Deployment(
    checkpoint=checkpoint,
    accelerator="cpu",
    max_batch_size=max_batch_size,
    batch_timeout=batch_timeout,
    workers=workers,
    monitor_queries=monitor_queries,
    json_backend=json_backend,
    host="127.0.0.1",
    port=port,
    log_level="error",
).serve()
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: Any, timeout: float = 30.0) -> tuple[Any, float, int]:
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    elapsed = time.perf_counter() - started
    return json.loads(body.decode("utf-8")), elapsed, len(body)


def _tail(path: Path, size: int = 4000) -> str:
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8", errors="replace")[-size:]


def _wait_for_server(base_url: str, process: subprocess.Popen[str], log_path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup\n{_tail(log_path)}")

        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except OSError as exc:
            last_error = str(exc)

        time.sleep(0.05)

    raise RuntimeError(f"timed out waiting for server: {last_error}\n{_tail(log_path)}")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass

    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def _rss_bytes(process: subprocess.Popen[str]) -> int:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-g", str(process.pid)],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0

    return sum(int(line.strip()) for line in output.splitlines() if line.strip()) * 1024


def _hyperparameters(payload_kind: str) -> Hyperparameters:
    label: dict[str, Any] = {
        "name": "label",
        "type": "category",
        "query": "[*].label",
        "max_vocab_size": 64,
    }
    if payload_kind == "embedding":
        label["embed"] = True
    elif payload_kind == "topk":
        label["target"] = True
        label["topk"] = [5]
    else:
        label["target"] = True

    return Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "array",
                "dropout": 0.1,
                "max_length": 1,
                "fields": [
                    {
                        "name": "amount",
                        "type": "number",
                        "query": "[*].amount",
                    },
                    label,
                ],
            },
        }
    )


def _build_checkpoint(path: Path, payload_kind: str) -> Path:
    records = [{"amount": float(index), "label": f"label_{index}"} for index in range(16)]
    params = _hyperparameters(payload_kind)
    model = Model(hyperparameters=params, batch_size=len(records))
    inputs = encode(
        batch=[[record] for record in records],
        hyperparameters=params,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )
    model.forward(inputs, strata=Strata.train)

    checkpoint = path / "benchmark_model.ckpt"
    model.save(checkpoint)
    return checkpoint


def _payload(inputs_per_request: int, offset: int) -> list[dict[str, str | float]]:
    return [
        {
            "amount": float((offset + index) % 16),
            "label": f"label_{(offset + index) % 16}",
        }
        for index in range(inputs_per_request)
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _run_requests(
    base_url: str,
    *,
    requests: int,
    concurrency: int,
    inputs_per_request: int,
) -> dict[str, float]:
    url = f"{base_url}/predict"
    latencies: list[float] = []
    completed_inputs = 0
    response_bytes = 0

    def send(index: int) -> tuple[int, float, int]:
        response, latency, size = _post_json(url, _payload(inputs_per_request, index * inputs_per_request))
        if not isinstance(response, list) or len(response) != inputs_per_request:
            raise RuntimeError(f"unexpected response shape: {type(response).__name__}")
        return len(response), latency, size

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, index) for index in range(requests)]
        for future in concurrent.futures.as_completed(futures):
            inputs, latency, size = future.result()
            completed_inputs += inputs
            latencies.append(latency)
            response_bytes += size

    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "requests": float(requests),
        "inputs": float(completed_inputs),
        "requests_per_second": requests / elapsed,
        "inputs_per_second": completed_inputs / elapsed,
        "response_bytes_per_second": response_bytes / elapsed,
        "request_latency_p50_seconds": _percentile(latencies, 0.50),
        "request_latency_p95_seconds": _percentile(latencies, 0.95),
        "request_latency_p99_seconds": _percentile(latencies, 0.99),
        "input_latency_p50_seconds": _percentile(latencies, 0.50) / inputs_per_request,
        "input_latency_p95_seconds": _percentile(latencies, 0.95) / inputs_per_request,
        "input_latency_p99_seconds": _percentile(latencies, 0.99) / inputs_per_request,
    }


def _markdown(result: dict[str, Any]) -> str:
    measured = result["measured"]
    return "\n".join(
        [
            "| metric | value |",
            "| --- | ---: |",
            f"| HTTP requests/s | {measured['requests_per_second']:.2f} |",
            f"| input rows/s | {measured['inputs_per_second']:.2f} |",
            f"| request p50 ms | {measured['request_latency_p50_seconds'] * 1000:.2f} |",
            f"| request p95 ms | {measured['request_latency_p95_seconds'] * 1000:.2f} |",
            f"| request p99 ms | {measured['request_latency_p99_seconds'] * 1000:.2f} |",
            f"| response MiB/s | {measured['response_bytes_per_second'] / (1024 * 1024):.2f} |",
            f"| RSS MiB | {result['rss_bytes'] / (1024 * 1024):.2f} |",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark relflow realtime deployment throughput.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--warmup-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--inputs-per-request", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--batch-timeout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--payload-kind", choices=["small", "embedding", "topk"], default="embedding")
    parser.add_argument("--monitor-queries", action="store_true")
    parser.add_argument("--json", choices=["stdlib", "orjson"], default="orjson")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="relflow-rt-bench-") as directory:
        tmp = Path(directory)
        checkpoint = _build_checkpoint(tmp, args.payload_kind)
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = tmp / "server.log"

        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    SERVER_SCRIPT,
                    str(checkpoint),
                    str(port),
                    str(args.max_batch_size),
                    str(args.batch_timeout),
                    str(args.workers),
                    str(args.monitor_queries),
                    args.json,
                ],
                cwd=args.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            try:
                _wait_for_server(base_url, process, log_path)
                warmup = _run_requests(
                    base_url,
                    requests=args.warmup_requests,
                    concurrency=args.concurrency,
                    inputs_per_request=args.inputs_per_request,
                )
                measured = _run_requests(
                    base_url,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    inputs_per_request=args.inputs_per_request,
                )
                rss_bytes = _rss_bytes(process)
            finally:
                _stop_process(process)

    result = {
        "concurrency": args.concurrency,
        "inputs_per_request": args.inputs_per_request,
        "max_batch_size": args.max_batch_size,
        "batch_timeout": args.batch_timeout,
        "workers": args.workers,
        "payload_kind": args.payload_kind,
        "monitor_queries": args.monitor_queries,
        "json": args.json,
        "warmup": warmup,
        "measured": measured,
        "rss_bytes": rss_bytes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.markdown:
        print()
        print(_markdown(result))


if __name__ == "__main__":
    main()
