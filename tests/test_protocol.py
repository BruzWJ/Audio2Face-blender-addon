from __future__ import annotations

import math
from pathlib import Path

import pytest

from audio2face.protocol import (
    MAX_CONTROL_LINE_BYTES,
    PROTOCOL_VERSION,
    WORKER_PROFILE,
    ProtocolError,
    decode_message,
    encode_message,
    make_request,
)


WORKER_PROTOCOL_SOURCE = (
    Path(__file__).resolve().parents[1] / "worker" / "src" / "protocol.cpp"
).read_text(encoding="utf-8")
WORKER_MAIN_SOURCE = (
    Path(__file__).resolve().parents[1] / "worker" / "src" / "main.cpp"
).read_text(encoding="utf-8")


def test_native_transport_requires_lf_and_rejects_cr() -> None:
    assert "if (std::cin.eof())" in WORKER_PROTOCOL_SOURCE
    assert '"JSONL request must end with LF"' in WORKER_PROTOCOL_SOURCE
    assert "line.find('\\r')" in WORKER_PROTOCOL_SOURCE
    assert '"JSONL request must not contain CR"' in WORKER_PROTOCOL_SOURCE


def test_windows_native_transport_uses_binary_stdio() -> None:
    assert "#ifdef _WIN32" in WORKER_MAIN_SOURCE
    assert "_setmode(_fileno(stdin), _O_BINARY)" in WORKER_MAIN_SOURCE
    assert "_setmode(_fileno(stdout), _O_BINARY)" in WORKER_MAIN_SOURCE


def test_native_worker_mirrors_the_python_wire_identity() -> None:
    assert WORKER_PROFILE.rpartition("/")[2] == "3"
    assert f'constexpr const char* kProtocol = "{PROTOCOL_VERSION}";' in (
        WORKER_PROTOCOL_SOURCE
    )
    assert f'{{"worker_profile", "{WORKER_PROFILE}"}}' in WORKER_PROTOCOL_SOURCE
    assert "constexpr std::size_t kMaximumRequestBytes = 1024U * 1024U;" in (
        WORKER_PROTOCOL_SOURCE
    )
    assert MAX_CONTROL_LINE_BYTES == 1024 * 1024


def test_request_round_trip_is_compact_utf8_and_one_record() -> None:
    message = make_request(
        "load_model",
        {
            "audio2face_model_path": "/models/脸/audio2face/model.json",
            "audio2emotion_model_path": "/models/脸/audio2emotion/model.json",
        },
    )

    encoded = encode_message(message)

    assert encoded.endswith("\n")
    assert encoded.count("\n") == 1
    assert "脸" in encoded
    assert " " not in encoded
    assert decode_message(encoded) == message


@pytest.mark.parametrize(
    "line",
    [
        "{}\n",
        '{"protocol":"audio2face/3","type":"response","id":"1","result":{}}',
        '{"protocol":"audio2face/999","type":"response","id":"1","result":{}}\n',
        '{"protocol":"audio2face/3","type":"response","id":"1","result":{}}\n{}\n',
        '{"protocol":"audio2face/3","type":"response","id":"1","result":{}}\n\n',
        b"\xff\n",
        '{"protocol":"audio2face/3","type":"response","id":"1","id":"2","result":{}}\n',
    ],
)
def test_decode_rejects_malformed_noncanonical_records(line: str | bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_message(line)


@pytest.mark.parametrize(
    "message",
    [
        {
            "protocol": PROTOCOL_VERSION,
            "type": "request",
            "id": 7,
            "method": "hello",
            "params": {},
        },
        {
            "protocol": PROTOCOL_VERSION,
            "type": "response",
            "id": 7,
            "result": {},
        },
        {
            "protocol": PROTOCOL_VERSION,
            "type": "event",
            "event": "progress",
            "operation_id": 7,
            "data": {},
        },
    ],
)
def test_every_control_identifier_must_be_a_string(message: dict[str, object]) -> None:
    expected_field = "operation_id" if message["type"] == "event" else "id"
    with pytest.raises(
        ProtocolError,
        match=rf"{expected_field} must be a non-empty string",
    ):
        encode_message(message)


def test_request_rejects_empty_id_unknown_method_and_extra_fields() -> None:
    empty_id = {
        "protocol": PROTOCOL_VERSION,
        "type": "request",
        "id": "",
        "method": "hello",
        "params": {},
    }
    with pytest.raises(ProtocolError, match="id must be a non-empty string"):
        encode_message(empty_id)
    with pytest.raises(ProtocolError, match="unsupported request method"):
        encode_message(make_request("unknown", {}))

    message = make_request("hello", {})
    message["unexpected"] = True
    with pytest.raises(ProtocolError, match="unknown fields: unexpected"):
        encode_message(message)


def test_canonical_result_event_requires_operation_id_and_exact_fields() -> None:
    event = {
        "protocol": PROTOCOL_VERSION,
        "type": "event",
        "event": "result",
        "operation_id": "operation-1",
        "data": {},
    }
    assert decode_message(encode_message(event))["data"] == {}

    without_operation = dict(event)
    without_operation.pop("operation_id")
    with pytest.raises(ProtocolError, match="missing fields: operation_id"):
        encode_message(without_operation)

    old_vocabulary = dict(event)
    old_vocabulary["job_id"] = old_vocabulary.pop("operation_id")
    with pytest.raises(
        ProtocolError,
        match=r"missing fields: operation_id; unknown fields: job_id",
    ):
        encode_message(old_vocabulary)


def test_stream_methods_and_events_are_canonical_protocol_members() -> None:
    for method in ("stream_start", "stream_chunk", "stream_end"):
        request = make_request(method, {})
        assert decode_message(encode_message(request)) == request

    for event_name in ("stream_frame", "stream_ended"):
        event = {
            "protocol": PROTOCOL_VERSION,
            "type": "event",
            "event": event_name,
            "operation_id": "stream-1",
            "data": {},
        }
        assert decode_message(encode_message(event)) == event


def test_error_requires_one_exact_shape() -> None:
    error = {
        "protocol": PROTOCOL_VERSION,
        "type": "error",
        "id": "request-1",
        "error": {
            "code": "invalid_params",
            "message": "invalid request",
            "details": {},
        },
    }
    assert decode_message(encode_message(error)) == error

    missing_details = {
        **error,
        "error": {"code": "invalid_params", "message": "invalid request"},
    }
    with pytest.raises(ProtocolError, match="missing fields: details"):
        encode_message(missing_details)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_encode_rejects_non_json_numbers(bad_value: float) -> None:
    message = make_request(
        "generate",
        {
            "settings": {
                "audio2emotion": {"emotion_strength": bad_value},
            }
        },
    )

    with pytest.raises(ProtocolError, match="not valid JSON"):
        encode_message(message)


def test_encode_rejects_a_control_record_over_the_jsonl_limit() -> None:
    message = make_request(
        "load_model",
        {"padding": "x" * MAX_CONTROL_LINE_BYTES},
    )

    with pytest.raises(ProtocolError, match="1 MiB"):
        encode_message(message)


def test_jsonl_limit_counts_payload_but_not_line_ending() -> None:
    message = make_request("load_model", {"padding": ""})
    empty_record = encode_message(message)
    padding_size = MAX_CONTROL_LINE_BYTES - len(empty_record[:-1].encode("utf-8"))
    message["params"]["padding"] = "x" * padding_size

    encoded = encode_message(message)

    assert len(encoded[:-1].encode("utf-8")) == MAX_CONTROL_LINE_BYTES
    assert decode_message(encoded) == message
    with pytest.raises(ProtocolError, match="LF-delimited"):
        decode_message(encoded[:-1] + "\r\n")
    with pytest.raises(ProtocolError, match="UTF-8 text"):
        decode_message(encoded.encode("utf-8"))  # type: ignore[arg-type]

    message["params"]["padding"] += "x"
    with pytest.raises(ProtocolError, match="1 MiB"):
        encode_message(message)


def test_protocol_normalizes_non_utf8_text_errors() -> None:
    message = make_request("load_model", {"audio2face_model_path": "\ud800"})
    with pytest.raises(ProtocolError, match="UTF-8"):
        encode_message(message)

    line = (
        '{"protocol":"audio2face/3","type":"response","id":"1",'
        '"result":{"value":"\ud800"}}\n'
    )
    with pytest.raises(ProtocolError, match="UTF-8"):
        decode_message(line)
