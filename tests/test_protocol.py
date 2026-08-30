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
    assert WORKER_PROFILE.rpartition("/")[2] == "13"
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
    for method in (
        "stream_start",
        "stream_chunk",
        "stream_settings",
        "stream_end",
    ):
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


def test_track_methods_and_events_are_canonical_protocol_members() -> None:
    for method in (
        "track_start",
        "track_chunk",
        "track_prepare",
        "track_render",
    ):
        request = make_request(method, {})
        assert decode_message(encode_message(request)) == request

    for event_name, data in (
        (
            "track_preview",
            {
                "revision": 1,
                "timestamp_sample": 0,
                "weights": [0.0],
                "effective_emotions": [0.0],
            },
        ),
        (
            "track_frame_batch",
            {
                "revision": 1,
                "offset": 0,
                "total_frames": 1,
                "timestamp_samples": [0],
                "weights": [[0.0]],
                "effective_emotions": [[0.0]],
            },
        ),
        ("track_ended", {"reason": "canceled"}),
    ):
        event = {
            "protocol": PROTOCOL_VERSION,
            "type": "event",
            "event": event_name,
            "operation_id": "track-1",
            "data": data,
        }
        assert decode_message(encode_message(event)) == event


@pytest.mark.parametrize(
    "method",
    ["track_frame", "track_end"],
)
def test_removed_operation_aliases_are_rejected(method: str) -> None:
    with pytest.raises(ProtocolError, match="unsupported request method"):
        encode_message(make_request(method, {}))
    assert f'if (method == "{method}")' not in WORKER_PROTOCOL_SOURCE


def test_native_stream_coalesces_settings_then_services_queued_pcm() -> None:
    enqueue_start = WORKER_PROTOCOL_SOURCE.index("  void enqueue_stream_chunk(")
    enqueue_end = WORKER_PROTOCOL_SOURCE.index(
        "  void enqueue_stream_end(", enqueue_start
    )
    enqueue_source = WORKER_PROTOCOL_SOURCE[enqueue_start:enqueue_end]
    assert "respond_and_release(request_id, response_gate);" in enqueue_source

    loop_start = WORKER_PROTOCOL_SOURCE.index("  void stream_loop(")
    loop_end = WORKER_PROTOCOL_SOURCE.index(
        "  void emit_stream_ended(", loop_start
    )
    loop_source = WORKER_PROTOCOL_SOURCE[loop_start:loop_end]
    collect_settings = loop_source.index(
        "settings_request_ids.push_back(std::move(it->request_id));"
    )
    latest_settings = loop_source.index(
        "latest_settings = std::move(it->settings);"
    )
    pop_pcm = loop_source.index("stream_queue_.pop_front();")
    apply_settings = loop_source.index(
        "backend_.stream_settings(latest_settings, canceled_);"
    )
    respond_settings = loop_source.index(
        "respond_to_active_stream_settings(operation_id,"
    )
    response_gate = loop_source.index("command->response_gate.get();")
    credit = loop_source.index(
        'emit_active_stream_event(operation_id, "stream_credit", json::object());'
    )
    inference = loop_source.index(
        "backend_.stream_chunk(command->audio, canceled_, emit_frame);"
    )
    assert (
        collect_settings
        < latest_settings
        < pop_pcm
        < apply_settings
        < respond_settings
        < response_gate
        < credit
        < inference
    )
    assert "for (const json& request_id : request_ids)" in WORKER_PROTOCOL_SOURCE
    assert "stream_queue_.push_back(std::move(command));" in WORKER_PROTOCOL_SOURCE


def test_native_worker_exposes_the_track_protocol_members() -> None:
    for method in (
        "track_start",
        "track_chunk",
        "track_prepare",
        "track_render",
    ):
        assert f'if (method == "{method}")' in WORKER_PROTOCOL_SOURCE
    assert '"track_ended"' in WORKER_PROTOCOL_SOURCE
    assert "bake_" not in WORKER_PROTOCOL_SOURCE
    track_loop = WORKER_PROTOCOL_SOURCE[
        WORKER_PROTOCOL_SOURCE.index("  void track_loop(") :
        WORKER_PROTOCOL_SOURCE.index("  void emit_track_ended(")
    ]
    assert '"track_preview"' in track_loop
    assert '"track_frame_batch"' in track_loop
    assert '{"timestamp_samples", std::move(timestamp_samples)}' in track_loop
    assert '{"weights", std::move(weights)}' in track_loop
    assert 'std::move(effective_emotions)' in track_loop


def test_native_track_supersedes_old_renders_and_batches_the_publish() -> None:
    enqueue_start = WORKER_PROTOCOL_SOURCE.index("  void enqueue_track_render(")
    enqueue_end = WORKER_PROTOCOL_SOURCE.index("  void cancel_operation(", enqueue_start)
    enqueue_source = WORKER_PROTOCOL_SOURCE[enqueue_start:enqueue_end]
    assert "if (revision <= latest)" in enqueue_source
    assert "latest_track_revision_.store(revision" in enqueue_source
    assert "track_queue_.erase(it)" in enqueue_source
    assert 'params.at("settings_timeline")' in enqueue_source
    assert "backend_.interrupt_operation();" not in enqueue_source

    loop_start = WORKER_PROTOCOL_SOURCE.index("  void track_loop(")
    loop_end = WORKER_PROTOCOL_SOURCE.index("  void emit_track_ended(", loop_start)
    loop_source = WORKER_PROTOCOL_SOURCE[loop_start:loop_end]
    assert "constexpr std::size_t kMaximumTrackFramesPerBatch = 64;" in (
        WORKER_PROTOCOL_SOURCE
    )
    batch = loop_source.index('"track_frame_batch"')
    render = loop_source.index("backend_.track_render(")
    response = loop_source.index("respond_to_track_render(", render)
    assert batch < render < response


def test_native_track_settings_timeline_expands_recursive_patches() -> None:
    start = WORKER_PROTOCOL_SOURCE.index(
        "std::vector<TrackSettingsEntry> parse_settings_timeline("
    )
    end = WORKER_PROTOCOL_SOURCE.index("\njson parse_request(", start)
    source = WORKER_PROTOCOL_SOURCE[start:end]
    for expression in (
        'require_exact_keys(entry, {"sample", "settings"});',
        '"settings_timeline must start at sample 0"',
        '"settings_timeline samples must be strictly increasing"',
        "apply_object_patch(settings, patch);",
        "timeline.push_back(TrackSettingsEntry{sample, settings});",
    ):
        assert expression in source


def test_native_cancel_cooperatively_stops_the_operation_before_waiting() -> None:
    cancel_start = WORKER_PROTOCOL_SOURCE.index("  void cancel_operation(")
    cancel_end = WORKER_PROTOCOL_SOURCE.index(
        "  void respond_and_release(", cancel_start
    )
    cancel_source = WORKER_PROTOCOL_SOURCE[cancel_start:cancel_end]
    cancel_flag = cancel_source.index(
        "canceled_.store(true, std::memory_order_release);"
    )
    notify = cancel_source.index("operation_condition_.notify_all();")
    response = cancel_source.index("respond_and_release(request_id, response_gate);")
    assert cancel_flag < notify < response
    assert "backend_.interrupt_operation();" not in cancel_source

    stop_start = WORKER_PROTOCOL_SOURCE.index("  void stop_operation(")
    stop_source = WORKER_PROTOCOL_SOURCE[stop_start:]
    cancel_flag = stop_source.index(
        "canceled_.store(true, std::memory_order_release);"
    )
    notify = stop_source.index("operation_condition_.notify_all();")
    join = stop_source.index("operation_thread_.join();")
    assert cancel_flag < notify < join
    assert "backend_.interrupt_operation();" not in stop_source


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
