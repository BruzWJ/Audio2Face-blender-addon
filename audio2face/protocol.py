"""The single JSONL control protocol used by Blender and its worker."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


PROTOCOL_VERSION = "audio2face/3"
MAX_CONTROL_LINE_BYTES = 1_048_576
_CONTROL_TYPES = frozenset({"request", "response", "error", "event"})
_REQUEST_METHODS = frozenset(
    {
        "hello",
        "load_model",
        "generate",
        "stream_start",
        "stream_chunk",
        "stream_end",
        "cancel",
        "shutdown",
    }
)
_EVENT_NAMES = frozenset(
    {"progress", "result", "canceled", "stream_frame", "stream_ended", "error"}
)
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


class ProtocolError(ValueError):
    """Raised when a control message violates the protocol schema."""


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must be a JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"control message contains duplicate field {key!r}")
        result[key] = value
    return result


def _require_exact_fields(
    value: dict[str, Any], expected: set[str], field: str
) -> None:
    keys = set(value)
    if keys == expected:
        return
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        details.append(f"unknown fields: {', '.join(unknown)}")
    raise ProtocolError(f"invalid {field} ({'; '.join(details)})")


def _require_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ProtocolError(f"{field} must be a non-empty protocol name")
    return value


def _require_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ProtocolError("id must be a non-empty string of at most 128 characters")
    return value


def _validate_message(message: dict[str, Any]) -> dict[str, Any]:
    envelope = _require_object(message, "message")
    if envelope.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol {envelope.get('protocol')!r}; expected {PROTOCOL_VERSION!r}"
        )

    message_type = envelope.get("type")
    if message_type not in _CONTROL_TYPES:
        raise ProtocolError(f"type must be one of {sorted(_CONTROL_TYPES)}")

    if message_type == "request":
        _require_exact_fields(
            envelope, {"protocol", "type", "id", "method", "params"}, "request"
        )
        envelope["id"] = _require_id(envelope["id"])
        method = _require_name(envelope["method"], "method")
        if method not in _REQUEST_METHODS:
            raise ProtocolError(f"unsupported request method {method!r}")
        envelope["method"] = method
        envelope["params"] = _require_object(envelope["params"], "params")
    elif message_type == "response":
        _require_exact_fields(
            envelope, {"protocol", "type", "id", "result"}, "response"
        )
        envelope["id"] = _require_id(envelope["id"])
        envelope["result"] = _require_object(envelope["result"], "result")
    elif message_type == "error":
        expected = {"protocol", "type", "error"}
        if "id" in envelope:
            expected.add("id")
            envelope["id"] = _require_id(envelope["id"])
        _require_exact_fields(envelope, expected, "error response")
        error = _require_object(envelope["error"], "error")
        _require_exact_fields(error, {"code", "message", "details"}, "error")
        error["code"] = _require_name(error["code"], "error.code")
        if not isinstance(error["message"], str) or not error["message"]:
            raise ProtocolError("error.message must be a non-empty string")
        error["details"] = _require_object(error["details"], "error.details")
        envelope["error"] = error
    else:
        _require_exact_fields(
            envelope,
            {"protocol", "type", "event", "job_id", "data"},
            "event",
        )
        event = _require_name(envelope["event"], "event")
        if event not in _EVENT_NAMES:
            raise ProtocolError(f"unsupported event {event!r}")
        envelope["event"] = event
        envelope["job_id"] = _require_id(envelope["job_id"])
        envelope["data"] = _require_object(envelope["data"], "data")

    return envelope


def make_request(
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "type": "request",
        "id": uuid.uuid4().hex,
        "method": method,
        "params": dict(params),
    }


def encode_message(message: dict[str, Any]) -> str:
    """Return one compact JSON line, rejecting non-JSON and oversized data."""

    envelope = _validate_message(message)
    try:
        line = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not valid JSON: {exc}") from exc
    try:
        payload_size = len(line.encode("utf-8"))
    except UnicodeError as exc:
        raise ProtocolError("control message is not valid UTF-8 text") from exc
    if payload_size > MAX_CONTROL_LINE_BYTES:
        raise ProtocolError("control message exceeds the 1 MiB JSONL limit")
    return line + "\n"


def decode_message(line: str | bytes) -> dict[str, Any]:
    """Decode and validate exactly one JSONL envelope."""

    if isinstance(line, bytes):
        if not line.endswith(b"\n"):
            raise ProtocolError("JSONL message must end with a newline")
        payload = line[:-2] if line.endswith(b"\r\n") else line[:-1]
        if b"\r" in payload or b"\n" in payload:
            raise ProtocolError("expected exactly one JSONL message")
        if len(payload) > MAX_CONTROL_LINE_BYTES:
            raise ProtocolError("control message exceeds the 1 MiB JSONL limit")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("control message is not UTF-8") from exc
    elif isinstance(line, str):
        if not line.endswith("\n"):
            raise ProtocolError("JSONL message must end with a newline")
        text = line[:-2] if line.endswith("\r\n") else line[:-1]
        if "\r" in text or "\n" in text:
            raise ProtocolError("expected exactly one JSONL message")
        try:
            payload_size = len(text.encode("utf-8"))
        except UnicodeError as exc:
            raise ProtocolError("control message is not valid UTF-8 text") from exc
        if payload_size > MAX_CONTROL_LINE_BYTES:
            raise ProtocolError("control message exceeds the 1 MiB JSONL limit")
    else:
        raise ProtocolError("control message must be text or bytes")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProtocolError(f"control message contains invalid number {token}")
            ),
        )
    except ProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON at column {exc.colno}: {exc.msg}") from exc
    return _validate_message(_require_object(value, "message"))
