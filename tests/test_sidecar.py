from __future__ import annotations

import os
import stat
import textwrap
import time
from pathlib import Path

from a2f_blender.protocol import PROTOCOL_VERSION
from a2f_blender.sidecar import (
    ClientDiagnostic,
    ControlMessage,
    Lifecycle,
    ProcessExited,
    SidecarClient,
)


def _make_fake_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "fake_a2f_worker.py"
    worker.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            import sys

            PROTOCOL = {PROTOCOL_VERSION!r}

            def emit(message):
                sys.stdout.write(json.dumps(message, separators=(\",\", \":\")) + \"\\n\")
                sys.stdout.flush()

            sys.stderr.write(\"fake worker diagnostic\\n\")
            sys.stderr.flush()

            for line in sys.stdin:
                request = json.loads(line)
                method = request[\"method\"]
                response = {{
                    \"protocol\": PROTOCOL,
                    \"type\": \"response\",
                    \"id\": request[\"id\"],
                    \"result\": {{\"method\": method}},
                }}
                emit(response)
                if method == \"shutdown\":
                    break
            """
        ),
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    return worker


def _make_invalid_utf8_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "invalid_utf8_worker.py"
    worker.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import sys

            sys.stdout.buffer.write(b'{"protocol":"a2f-blender/2","type":"response","id":"x","result":{"value":"\\xff"}}\\n')
            sys.stdout.buffer.flush()
            for _line in sys.stdin:
                pass
            """
        ),
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    return worker


def _collect_until(client: SidecarClient, predicate: object, timeout: float = 5.0) -> list[object]:
    deadline = time.monotonic() + timeout
    collected: list[object] = []
    while time.monotonic() < deadline:
        collected.extend(client.poll())
        if predicate(collected):  # type: ignore[operator]
            return collected
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for sidecar event; collected={collected!r}")


def test_sidecar_exchanges_jsonl_and_shuts_down_cleanly(tmp_path: Path) -> None:
    worker = _make_fake_worker(tmp_path)
    client = SidecarClient()

    try:
        client.start(worker, cwd=tmp_path, env=os.environ)
        assert client.state is Lifecycle.RUNNING
        assert client.pid is not None

        startup_events = _collect_until(
            client,
            lambda events: any(isinstance(event, ClientDiagnostic) for event in events),
        )
        assert any(
            isinstance(event, ClientDiagnostic)
            and event.message == "fake worker diagnostic"
            for event in startup_events
        )

        request_id = client.request("hello", {})
        response_events = _collect_until(
            client,
            lambda events: any(
                isinstance(event, ControlMessage)
                and event.envelope.get("type") == "response"
                and event.envelope.get("id") == request_id
                for event in events
            ),
        )
        response = next(
            event.envelope
            for event in response_events
            if isinstance(event, ControlMessage)
            and event.envelope.get("id") == request_id
        )
        assert response["result"] == {"method": "hello"}

        shutdown_id = client.begin_shutdown(timeout=1.0)
        assert isinstance(shutdown_id, str)
        assert client.state is Lifecycle.STOPPING
        shutdown_events = _collect_until(
            client,
            lambda events: any(isinstance(event, ProcessExited) for event in events),
        )
        assert any(
            isinstance(event, ControlMessage)
            and event.envelope.get("id") == shutdown_id
            and event.envelope.get("result") == {"method": "shutdown"}
            for event in shutdown_events
        )
        exit_event = next(
            event for event in shutdown_events if isinstance(event, ProcessExited)
        )
        assert exit_event.returncode == 0
        assert client.state is Lifecycle.STOPPED
        assert client.pid is None
    finally:
        client.close()


def test_sidecar_terminates_worker_that_emits_invalid_utf8(tmp_path: Path) -> None:
    worker = _make_invalid_utf8_worker(tmp_path)
    client = SidecarClient()

    try:
        client.start(worker, cwd=tmp_path, env=os.environ)
        events = _collect_until(
            client,
            lambda collected: any(
                isinstance(event, ProcessExited) for event in collected
            ),
        )
        assert any(
            isinstance(event, ClientDiagnostic)
            and "worker stdout failed" in event.message
            and "utf-8" in event.message
            for event in events
        )
        assert client.state is Lifecycle.FAILED
        assert client.pid is None
    finally:
        client.close()
