from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa

from relflow.architecture.root import Model
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema

SERVER_SCRIPT = """
import sys

from relflow.inference.deployment import Deployment

checkpoint = sys.argv[1]
port = int(sys.argv[2])

Deployment(
    checkpoint=checkpoint,
    accelerator="cpu",
    max_batch_size=1,
    batch_timeout=0.0,
    host="127.0.0.1",
    port=port,
    log_level="error",
).serve()
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def health_url(base_url: str) -> str:
    return f"{base_url}/health"


def tail_text(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def wait_for_server(base_url: str, process: subprocess.Popen[str], log_path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"deployment exited before readiness probe succeeded\nlog tail:\n{tail_text(log_path)}"
            )

        try:
            with urllib.request.urlopen(health_url(base_url), timeout=1.0) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except OSError as exc:
            last_error = str(exc)

        time.sleep(0.1)

    raise AssertionError(f"timed out waiting for deployment readiness: {last_error}\nlog tail:\n{tail_text(log_path)}")


def stop_process(process: subprocess.Popen[str], timeout: float = 10.0) -> None:
    try:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=timeout)
    finally:
        log_handle = getattr(process, "log_handle", None)
        if log_handle is not None:
            log_handle.close()


def post_json(url: str, payload: Any, timeout: float = 30.0) -> tuple[int, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"deployment returned HTTP {exc.code}: {body}") from exc


def schema() -> Schema:
    return Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "dropout": 0.1,
                "length": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "embed": True,
                        "size": 32,
                    }
                ],
            },
        }
    )


def write_fake_records(path: Path) -> list[dict[str, str]]:
    records = [{"label": "alpha"}, {"label": "beta"}]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return records


def build_checkpoint(tmp_path: Path) -> tuple[Path, Schema]:
    dataset_path = tmp_path / "fake_records.ndjson"
    records = write_fake_records(dataset_path)
    model_schema = schema()
    model = Model(schema=model_schema, batch_size=2)

    inputs = model.encode(
        pa.Table.from_pylist(records),
        strata=Strata.train,
    )
    model.forward(inputs, strata=Strata.train)

    checkpoint_path = tmp_path / "fake_model.ckpt"
    model.save(checkpoint_path)
    return checkpoint_path, model_schema


def launch_deployment(checkpoint: Path, port: int, log_path: Path) -> subprocess.Popen[str]:
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", SERVER_SCRIPT, str(checkpoint), str(port)],
        cwd=repo_root(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process.log_handle = log_handle
    return process


def test_deployment_serves_embeddings_from_temporary_checkpoint(tmp_path: Path) -> None:
    checkpoint_path, model_schema = build_checkpoint(tmp_path)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "deployment.log"
    process = launch_deployment(checkpoint=checkpoint_path, port=port, log_path=log_path)

    try:
        wait_for_server(base_url=base_url, process=process, log_path=log_path)
        status, payload = post_json(f"{base_url}/predict", {"label": "alpha"})
    finally:
        stop_process(process)

    assert status == 200
    assert "root/label" in payload["predictions"]

    embedding = payload["predictions"]["root/label"]["embedding"]
    assert len(embedding) == model_schema.d_model
    assert all(isinstance(value, float) for value in embedding)


def test_deployment_accepts_multiple_inputs_in_one_request(tmp_path: Path) -> None:
    checkpoint_path, model_schema = build_checkpoint(tmp_path)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "deployment.log"
    process = launch_deployment(checkpoint=checkpoint_path, port=port, log_path=log_path)

    try:
        wait_for_server(base_url=base_url, process=process, log_path=log_path)
        status, payload = post_json(f"{base_url}/predict", [{"label": "alpha"}, {"label": "beta"}])
    finally:
        stop_process(process)

    assert status == 200
    assert isinstance(payload, list)
    assert len(payload) == 2
    for item in payload:
        assert "root/label" in item["predictions"]
        embedding = item["predictions"]["root/label"]["embedding"]
        assert len(embedding) == model_schema.d_model


def test_deployment_accepts_unseen_category_values_at_runtime(tmp_path: Path) -> None:
    checkpoint_path, _ = build_checkpoint(tmp_path)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "deployment.log"
    process = launch_deployment(checkpoint=checkpoint_path, port=port, log_path=log_path)

    try:
        wait_for_server(base_url=base_url, process=process, log_path=log_path)
        _, alpha_payload = post_json(f"{base_url}/predict", {"label": "alpha"})
        status, gamma_payload = post_json(f"{base_url}/predict", {"label": "gamma"})
    finally:
        stop_process(process)

    assert status == 200
    assert "root/label" in alpha_payload["predictions"]
    assert "root/label" in gamma_payload["predictions"]
    assert (
        alpha_payload["predictions"]["root/label"]["embedding"]
        != gamma_payload["predictions"]["root/label"]["embedding"]
    )
