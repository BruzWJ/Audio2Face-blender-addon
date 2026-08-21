from __future__ import annotations

import math

import pytest

from audio2face.protocol import (
    MAX_CONTROL_LINE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
    make_request,
)


def test_request_round_trip_is_compact_utf8_and_one_record() -> None:
    message = make_request(
        "load_model",
        {
            "audio2face_model_path": "/models/脸/audio2face/model.json",
            "audio2emotion_model_path": "/models/脸/audio2emotion/model.json",
            "identity_index": 0,
        },
    )

    encoded = encode_message(message)

    assert encoded.endswith("\n")
    assert encoded.count("\n") == 1
    assert "脸" in encoded
    assert " " not in encoded
    assert decode_message(encoded.encode("utf-8")) == message


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
            "job_id": 7,
            "data": {},
        },
    ],
)
def test_every_control_correlation_id_must_be_a_string(message: dict[str, object]) -> None:
    with pytest.raises(ProtocolError, match="id must be a non-empty string"):
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


def test_canonical_result_event_requires_job_id_and_exact_fields() -> None:
    event = {
        "protocol": PROTOCOL_VERSION,
        "type": "event",
        "event": "result",
        "job_id": "job-1",
        "data": {},
    }
    assert decode_message(encode_message(event))["data"] == {}

    without_job = dict(event)
    without_job.pop("job_id")
    with pytest.raises(ProtocolError, match="missing fields: job_id"):
        encode_message(without_job)


def test_stream_methods_and_events_are_canonical_protocol_members() -> None:
    for method in ("stream_start", "stream_chunk", "stream_end"):
        request = make_request(method, {})
        assert decode_message(encode_message(request)) == request

    for event_name in ("stream_frame", "stream_ended"):
        event = {
            "protocol": PROTOCOL_VERSION,
            "type": "event",
            "event": event_name,
            "job_id": "stream-1",
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
        {"settings": {"input_strength": bad_value}},
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
    assert decode_message(encoded[:-1] + "\r\n") == message
    assert decode_message(encoded.encode("utf-8")) == message

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
