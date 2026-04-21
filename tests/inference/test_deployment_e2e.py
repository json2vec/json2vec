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

import torch

from json2vec.architecture.root import JSON2Vec
from json2vec.data.datasets import encode
from json2vec.structs.enums import Stage, Strata, Suffix
from json2vec.structs.experiment import Dataset, Session
from json2vec.structs.structure import Structure


SERVER_SCRIPT = """
import sys

import litserve as ls

from json2vec.inference.deployment import Deployment

checkpoint = sys.argv[1]
port = int(sys.argv[2])

server = ls.LitServer(
    lit_api=Deployment(
        checkpoint=checkpoint,
        max_batch_size=1,
        batch_timeout=0.0,
    ),
    accelerator="cpu",
    workers_per_device=1,
    track_requests=False,
)
server.run(
    host="127.0.0.1",
    port=port,
    log_level="error",
    generate_client_file=False,
)
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _health_url(base_url: str) -> str:
    return f"{base_url}/health"


def _tail_text(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _wait_for_server(base_url: str, process: subprocess.Popen[str], log_path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "deployment exited before readiness probe succeeded\n"
                f"log tail:\n{_tail_text(log_path)}"
            )

        try:
            with urllib.request.urlopen(_health_url(base_url), timeout=1.0) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except OSError as exc:
            last_error = str(exc)

        time.sleep(0.1)

    raise AssertionError(
        f"timed out waiting for deployment readiness: {last_error}\n"
        f"log tail:\n{_tail_text(log_path)}"
    )


def _stop_process(process: subprocess.Popen[str], timeout: float = 10.0) -> None:
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


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
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


def _structure() -> Structure:
    return Structure.model_validate(
        {
            "name": "deployment-e2e",
            "type": "structure",
            "batch_size": 2,
            "dropout": 0.1,
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "context",
                "context_size": 1,
                "n_outputs": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "query": "[*].label",
                        "max_vocab_size": 32,
                    }
                ],
            },
        }
    )


def _session(dataset_root: Path) -> Session:
    dataset = Dataset.model_validate(
        {
            "root": str(dataset_root),
            "sample_rate": 1.0,
            "file_buffer_size": 4,
            "observation_buffer_size": 4,
            "processor": "default",
            "kwargs": {},
            "suffix": Suffix.ndjson,
            "patterns": {strata: r".*\.ndjson$" for strata in Strata},
        }
    )

    return Session.model_validate(
        {
            "name": "deployment-e2e",
            "dataset": dataset,
            "structure": _structure(),
            "task": Stage.predict,
            "output": ["root/label"],
        }
    )


def _write_fake_records(path: Path) -> list[dict[str, str]]:
    records = [{"label": "alpha"}, {"label": "beta"}]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return records


def _build_checkpoint(tmp_path: Path) -> tuple[Path, Session]:
    dataset_path = tmp_path / "fake_records.ndjson"
    records = _write_fake_records(dataset_path)
    session = _session(dataset_root=dataset_path)
    model = JSON2Vec.get_or_create(session=session)

    inputs = encode(
        batch=[[record] for record in records],
        session=session,
        strata=Strata.train,
        state=model.state,
    )
    model.forward(inputs)

    checkpoint_path = tmp_path / "fake_model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "session": session.model_dump(mode="python"),
        },
        checkpoint_path,
    )
    return checkpoint_path, session


def _launch_deployment(checkpoint: Path, port: int, log_path: Path) -> subprocess.Popen[str]:
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", SERVER_SCRIPT, str(checkpoint), str(port)],
        cwd=_repo_root(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process.log_handle = log_handle
    return process


def test_deployment_serves_embeddings_from_temporary_checkpoint(tmp_path: Path) -> None:
    checkpoint_path, session = _build_checkpoint(tmp_path)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "deployment.log"
    process = _launch_deployment(checkpoint=checkpoint_path, port=port, log_path=log_path)

    try:
        _wait_for_server(base_url=base_url, process=process, log_path=log_path)
        status, payload = _post_json(f"{base_url}/predict", {"label": "alpha"})
    finally:
        _stop_process(process)

    assert status == 200
    assert payload["predictions"] == {}
    assert "embeddings" in payload
    assert "root/label" in payload["embeddings"]

    embedding = payload["embeddings"]["root/label"]["embedding"]
    assert len(embedding) == session.structure.d_model
    assert all(isinstance(value, float) for value in embedding)


def test_deployment_accepts_unseen_category_values_at_runtime(tmp_path: Path) -> None:
    checkpoint_path, _ = _build_checkpoint(tmp_path)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "deployment.log"
    process = _launch_deployment(checkpoint=checkpoint_path, port=port, log_path=log_path)

    try:
        _wait_for_server(base_url=base_url, process=process, log_path=log_path)
        _, alpha_payload = _post_json(f"{base_url}/predict", {"label": "alpha"})
        status, gamma_payload = _post_json(f"{base_url}/predict", {"label": "gamma"})
    finally:
        _stop_process(process)

    assert status == 200
    assert alpha_payload["predictions"] == {}
    assert gamma_payload["predictions"] == {}
    assert "root/label" in alpha_payload["embeddings"]
    assert "root/label" in gamma_payload["embeddings"]
    assert alpha_payload["embeddings"]["root/label"]["embedding"] != gamma_payload["embeddings"]["root/label"]["embedding"]
