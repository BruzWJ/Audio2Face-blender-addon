"""Threaded, nonblocking JSONL client for the local Audio2Face worker.

Reader/writer threads only touch standard-library queues. Blender state is
consumed later by :mod:`audio2face.runtime` from a ``bpy.app.timers`` callback.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .protocol import ProtocolError, decode_message, encode_message, make_request


class SidecarError(RuntimeError):
    """Raised for local worker lifecycle failures."""


class Lifecycle(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ControlMessage:
    envelope: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClientDiagnostic:
    message: str


@dataclass(frozen=True, slots=True)
class ProcessExited:
    returncode: int


ClientEvent = ControlMessage | ClientDiagnostic | ProcessExited


class SidecarClient:
    """Own one worker process and exchange validated JSONL without blocking UI."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._state = Lifecycle.STOPPED
        self._state_lock = threading.RLock()
        self._incoming: queue.Queue[ClientEvent] = queue.Queue()
        self._outgoing: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._shutdown_deadline: float | None = None
        self._terminate_deadline: float | None = None

    @property
    def state(self) -> Lifecycle:
        with self._state_lock:
            return self._state

    def start(
        self,
        executable: Path,
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> None:
        """Spawn without a shell and start queue-only I/O threads."""

        with self._state_lock:
            previous = self._process
        if previous is not None:
            if previous.poll() is None:
                raise SidecarError("worker is already running")
            # Join the prior process's queue threads before reusing queues.
            self.close(timeout=0.05)

        with self._state_lock:
            self._shutdown_deadline = None
            self._terminate_deadline = None
            self._drain_queue(self._incoming)
            self._drain_queue(self._outgoing)

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW
            try:
                process = subprocess.Popen(
                    [str(executable)],
                    cwd=str(cwd),
                    env=dict(env),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="strict",
                    bufsize=1,
                    shell=False,
                    creationflags=creationflags,
                )
            except OSError as exc:
                self._state = Lifecycle.FAILED
                raise SidecarError(f"could not start worker: {exc}") from exc

            self._process = process
            self._state = Lifecycle.RUNNING

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.reconfigure(newline="\n")
        process.stdout.reconfigure(newline="")
        process.stderr.reconfigure(newline="")
        pipe_threads = (
            self._thread("a2f-worker-stdin", self._writer_loop, process),
            self._thread("a2f-worker-stdout", self._stdout_loop, process),
            self._thread("a2f-worker-stderr", self._stderr_loop, process),
        )
        self._threads = [
            *pipe_threads,
            self._thread(
                "a2f-worker-watch",
                self._watch_loop,
                process,
                pipe_threads,
            ),
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _thread(name: str, target: Any, *args: object) -> threading.Thread:
        return threading.Thread(name=name, target=target, args=args, daemon=True)

    @staticmethod
    def _drain_queue(target: queue.Queue[Any]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def _writer_loop(self, process: subprocess.Popen[str]) -> None:
        assert process.stdin is not None
        while True:
            line = self._outgoing.get()
            if line is None:
                return
            try:
                process.stdin.write(line)
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                self._incoming.put(ClientDiagnostic(f"worker stdin failed: {exc}"))
                return

    def _stdout_loop(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                try:
                    envelope = decode_message(line)
                    if envelope["type"] == "request":
                        raise ProtocolError("worker stdout may not contain requests")
                    self._incoming.put(ControlMessage(envelope))
                except ProtocolError as exc:
                    preview = line.rstrip()[:240]
                    self._incoming.put(
                        ClientDiagnostic(f"invalid worker JSONL ({exc}): {preview}")
                    )
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    return
        except (OSError, ValueError) as exc:
            self._incoming.put(ClientDiagnostic(f"worker stdout failed: {exc}"))
            try:
                process.terminate()
            except OSError:
                pass

    def _stderr_loop(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        try:
            for line in process.stderr:
                text = line.rstrip()
                if text:
                    self._incoming.put(ClientDiagnostic(text[:4000]))
        except (OSError, ValueError) as exc:
            self._incoming.put(ClientDiagnostic(f"worker stderr failed: {exc}"))

    def _watch_loop(
        self,
        process: subprocess.Popen[str],
        pipe_threads: tuple[threading.Thread, ...],
    ) -> None:
        returncode = process.wait()
        self._outgoing.put(None)
        for thread in pipe_threads:
            thread.join(timeout=0.1)
        with self._state_lock:
            if self._process is process:
                self._state = Lifecycle.STOPPED if returncode == 0 else Lifecycle.FAILED
                self._shutdown_deadline = None
                self._terminate_deadline = None
        self._incoming.put(ProcessExited(returncode))

    def request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> str:
        """Queue a request; no pipe I/O occurs on the caller's thread."""

        with self._state_lock:
            if self._process is None or self._process.poll() is not None:
                raise SidecarError("worker is not running")
            if self._state == Lifecycle.STOPPING and method != "shutdown":
                raise SidecarError("worker is shutting down")
        envelope = make_request(method, params)
        self._outgoing.put(encode_message(envelope))
        return envelope["id"]

    def begin_shutdown(self, *, timeout: float) -> str | None:
        """Queue graceful shutdown and let :meth:`tick` enforce its deadline."""

        with self._state_lock:
            process = self._process
            if process is None or process.poll() is not None:
                self._state = Lifecycle.STOPPED
                return None
            if self._state == Lifecycle.STOPPING:
                return None
        try:
            request_id = self.request("shutdown", {})
        except SidecarError:
            request_id = None
        with self._state_lock:
            self._state = Lifecycle.STOPPING
            self._shutdown_deadline = time.monotonic() + max(0.05, timeout)
            self._terminate_deadline = None
        return request_id

    def tick(self) -> None:
        """Perform nonblocking process/deadline maintenance from a UI timer."""

        with self._state_lock:
            process = self._process
            state = self._state
            shutdown_deadline = self._shutdown_deadline
            terminate_deadline = self._terminate_deadline
        if process is None or process.poll() is not None or state != Lifecycle.STOPPING:
            return

        now = time.monotonic()
        if shutdown_deadline is not None and now >= shutdown_deadline and terminate_deadline is None:
            try:
                process.terminate()
                self._incoming.put(ClientDiagnostic("worker exceeded shutdown deadline; terminated"))
            except OSError:
                pass
            with self._state_lock:
                self._terminate_deadline = now + 1.0
        elif terminate_deadline is not None and now >= terminate_deadline:
            try:
                process.kill()
                self._incoming.put(ClientDiagnostic("worker ignored termination; killed"))
            except OSError:
                pass
            with self._state_lock:
                self._terminate_deadline = None

    def poll(self) -> list[ClientEvent]:
        """Drain a bounded batch of events without blocking."""

        events: list[ClientEvent] = []
        for _ in range(256):
            try:
                events.append(self._incoming.get_nowait())
            except queue.Empty:
                break
        return events

    def close(self, *, timeout: float) -> None:
        """Bounded synchronous cleanup for add-on unregistration/interpreter exit."""

        with self._state_lock:
            process = self._process
        if process is None:
            return
        if process.poll() is None:
            self.begin_shutdown(timeout=max(0.05, timeout))
            try:
                process.wait(timeout=max(0.05, timeout))
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                    process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                        process.wait(timeout=0.5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass

        self._outgoing.put(None)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.1)
        with self._state_lock:
            self._process = None
            self._state = Lifecycle.STOPPED
            self._shutdown_deadline = None
            self._terminate_deadline = None
        self._threads.clear()
