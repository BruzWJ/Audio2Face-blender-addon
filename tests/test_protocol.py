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
    assert WORKER_PROFILE.rpartition("/")[2] == "10"
    assert f'constexpr const char* kProtocol = "{PROTOCOL_VERSION}";' in (
        WORKER_PROTOCOL_SOURCE
    )
    assert f'{{"worker_profile", "{WORKER_PROFILE}"}}' in WORKER_PROTOCOL_SOURCE
    assert "constexpr std::size_t kMaximumRequestBytes = 1024U * 1024U;" in (
        WORKER_PROTOCOL_SOURCE
    )
    assert MAX_CONTROL_LINE_BYTES == 1024 * 1024
    assert '{"effective_emotions", frame.effective_emotions}' in WORKER_PROTOCOL_SOURCE


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
        f'{{"protocol":"{PROTOCOL_VERSION}","type":"response","id":"1","result":{{}}}}',
        '{"protocol":"audio2face/9","type":"response","id":"1","result":{}}\n',
        '{"protocol":"audio2face/999","type":"response","id":"1","result":{}}\n',
        f'{{"protocol":"{PROTOCOL_VERSION}","type":"response","id":"1","result":{{}}}}\n{{}}\n',
        f'{{"protocol":"{PROTOCOL_VERSION}","type":"response","id":"1","result":{{}}}}\n\n',
        b"\xff\n",
        f'{{"protocol":"{PROTOCOL_VERSION}","type":"response","id":"1","id":"2","result":{{}}}}\n',
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
            "event": "stream_frame",
            "operation_id": 7,
            "data": {
                "timestamp_sample": 0,
                "weights": [0.0],
                "effective_emotions": [0.0],
            },
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


def test_canonical_stream_event_requires_operation_id_and_exact_fields() -> None:
    data = {
        "timestamp_sample": 0,
        "weights": [0.0],
        "effective_emotions": [0.0],
    }
    event = {
        "protocol": PROTOCOL_VERSION,
        "type": "event",
        "event": "stream_frame",
        "operation_id": "operation-1",
        "data": data,
    }
    assert decode_message(encode_message(event))["data"] == data

    without_operation = dict(event)
    without_operation.pop("operation_id")
    with pytest.raises(ProtocolError, match="missing fields: operation_id"):
        encode_message(without_operation)


def test_stream_methods_and_events_are_canonical_protocol_members() -> None:
    for method in ("stream_start", "stream_chunk", "stream_settings", "stream_end"):
        request = make_request(method, {})
        assert decode_message(encode_message(request)) == request

    for event_name in (
        "stream_credit",
        "stream_frame",
        "stream_ended",
    ):
        data = (
            {
                "timestamp_sample": 0,
                "weights": [0.0],
                "effective_emotions": [0.0],
            }
            if event_name == "stream_frame"
            else {}
        )
        event = {
            "protocol": PROTOCOL_VERSION,
            "type": "event",
            "event": event_name,
            "operation_id": "stream-1",
            "data": data,
        }
        assert decode_message(encode_message(event)) == event


def test_bake_methods_and_events_are_canonical_protocol_members() -> None:
    for method in (
        "bake_start",
        "bake_chunk",
        "bake_prepare",
        "bake_frame",
        "bake_end",
    ):
        request = make_request(method, {})
        assert decode_message(encode_message(request)) == request

    for reason in ("completed", "canceled"):
        ended = {
            "protocol": PROTOCOL_VERSION,
            "type": "event",
            "event": "bake_ended",
            "operation_id": "bake-1",
            "data": {"reason": reason},
        }
        assert decode_message(encode_message(ended)) == ended

def test_native_stream_settings_is_one_ordered_queue_command() -> None:
    assert 'if (method == "stream_settings")' in WORKER_PROTOCOL_SOURCE
    assert "enqueue_stream_settings(id, params);" in WORKER_PROTOCOL_SOURCE
    assert 'require_exact_keys(params, {"operation_id", "settings"});' in (
        WORKER_PROTOCOL_SOURCE
    )
    enqueue_start = WORKER_PROTOCOL_SOURCE.index("  void enqueue_stream_settings(")
    enqueue_end = WORKER_PROTOCOL_SOURCE.index(
        "  void enqueue_stream_end(", enqueue_start
    )
    enqueue_source = WORKER_PROTOCOL_SOURCE[enqueue_start:enqueue_end]
    assert "stream_settings_pending_" in enqueue_source
    assert "Wait for the pending stream settings response" in enqueue_source
    assert "respond_and_release" not in enqueue_source

    loop_start = WORKER_PROTOCOL_SOURCE.index("  void stream_loop(")
    loop_end = WORKER_PROTOCOL_SOURCE.index(
        "  void emit_bake_ended(", loop_start
    )
    loop_source = WORKER_PROTOCOL_SOURCE[loop_start:loop_end]
    apply_settings = loop_source.index(
        "backend_.stream_settings(command.settings, canceled_);"
    )
    acknowledge = loop_source.index("respond_to_active_stream_settings(")
    assert apply_settings < acknowledge


def test_native_stream_chunk_emits_one_dequeue_credit() -> None:
    enqueue_start = WORKER_PROTOCOL_SOURCE.index("  void enqueue_stream_chunk(")
    enqueue_end = WORKER_PROTOCOL_SOURCE.index(
        "  void enqueue_stream_settings(", enqueue_start
    )
    enqueue_source = WORKER_PROTOCOL_SOURCE[enqueue_start:enqueue_end]
    assert "respond_and_release(request_id, response_gate);" in enqueue_source

    loop_start = WORKER_PROTOCOL_SOURCE.index("  void stream_loop(")
    loop_end = WORKER_PROTOCOL_SOURCE.index(
        "  void emit_stream_ended(", loop_start
    )
    loop_source = WORKER_PROTOCOL_SOURCE[loop_start:loop_end]
    pop = loop_source.index("stream_queue_.pop_front();")
    response_gate = loop_source.index("command.response_gate.get();")
    credit = loop_source.index(
        'emit_active_stream_event(operation_id, "stream_credit", json::object());'
    )
    inference = loop_source.index(
        "backend_.stream_chunk(command.audio, canceled_, emit_frame);"
    )
    assert pop < response_gate < credit < inference
    assert "StreamCommand::Kind::Settings" in WORKER_PROTOCOL_SOURCE
    assert "stream_queue_.push_back(std::move(command));" in WORKER_PROTOCOL_SOURCE
    assert "backend_.stream_settings(command.settings" in (
        WORKER_PROTOCOL_SOURCE
    )
def test_native_worker_exposes_the_bake_protocol_members() -> None:
    for method in (
        "bake_start",
        "bake_chunk",
        "bake_prepare",
        "bake_frame",
        "bake_end",
    ):
        assert f'if (method == "{method}")' in WORKER_PROTOCOL_SOURCE
    assert '"bake_ended"' in WORKER_PROTOCOL_SOURCE
    bake_loop = WORKER_PROTOCOL_SOURCE[
        WORKER_PROTOCOL_SOURCE.index("  void bake_loop(") :
        WORKER_PROTOCOL_SOURCE.index("  void emit_bake_ended(")
    ]
    assert '{{"weights", result.weights}}' in bake_loop


def test_native_bake_keeps_one_frame_in_flight_and_cancel_suppresses_responses() -> None:
    enqueue_start = WORKER_PROTOCOL_SOURCE.index("  void enqueue_bake_frame(")
    enqueue_end = WORKER_PROTOCOL_SOURCE.index("  void enqueue_bake_end(", enqueue_start)
    enqueue_source = WORKER_PROTOCOL_SOURCE[enqueue_start:enqueue_end]
    assert "if (bake_frame_pending_)" in enqueue_source
    assert "bake_frame_pending_ = true;" in enqueue_source

    response_start = WORKER_PROTOCOL_SOURCE.index("  void respond_to_active_bake(")
    response_end = WORKER_PROTOCOL_SOURCE.index(
        "  void emit_active_stream_event(", response_start
    )
    response_source = WORKER_PROTOCOL_SOURCE[response_start:response_end]
    cancel_check = response_source.index(
        "canceled_.load(std::memory_order_acquire)"
    )
    response = response_source.index("emitter_.response(")
    release = response_source.index("bake_frame_pending_ = false")
    assert cancel_check < response < release

    loop_start = WORKER_PROTOCOL_SOURCE.index("  void bake_loop(")
    loop_end = WORKER_PROTOCOL_SOURCE.index("  void emit_bake_ended(", loop_start)
    loop_source = WORKER_PROTOCOL_SOURCE[loop_start:loop_end]
    assert "backend_.bake_chunk(command.audio, canceled_)" in loop_source
    assert "canceled_);" in loop_source


def test_native_cancel_interrupts_interactive_compute_before_waiting() -> None:
    cancel_start = WORKER_PROTOCOL_SOURCE.index("  void cancel_operation(")
    cancel_end = WORKER_PROTOCOL_SOURCE.index(
        "  void respond_and_release(", cancel_start
    )
    cancel_source = WORKER_PROTOCOL_SOURCE[cancel_start:cancel_end]
    cancel_flag = cancel_source.index(
        "canceled_.store(true, std::memory_order_release);"
    )
    interrupt = cancel_source.index("backend_.interrupt_operation();")
    response = cancel_source.index("respond_and_release(request_id, response_gate);")
    assert cancel_flag < interrupt < response

    stop_start = WORKER_PROTOCOL_SOURCE.index("  void stop_operation(")
    stop_source = WORKER_PROTOCOL_SOURCE[stop_start:]
    assert "backend_.interrupt_operation();" in stop_source
    assert stop_source.index("backend_.interrupt_operation();") < stop_source.index(
        "operation_thread_.join();"
    )


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
        "stream_start",
        {
            "settings": {
                "emotion_driver": {
                    "emotion_strength": bad_value,
                    "generated": None,
                    "preferred": None,
                },
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
        f'{{"protocol":"{PROTOCOL_VERSION}","type":"response","id":"1",'
        '"result":{"value":"\ud800"}}\n'
    )
    with pytest.raises(ProtocolError, match="UTF-8"):
        decode_message(line)
