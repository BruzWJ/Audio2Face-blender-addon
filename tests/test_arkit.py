from __future__ import annotations

import re
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "worker" / "src" / "a2f_backend.cpp"
).read_text(encoding="utf-8")

AUDIO2FACE_KEYS = {
    "input_strength",
    "lower_face_smoothing",
    "upper_face_smoothing",
    "lower_face_strength",
    "upper_face_strength",
    "face_mask_level",
    "face_mask_softness",
    "skin_strength",
    "blink_strength",
    "eyelid_open_offset",
    "lip_open_offset",
    "eyeballs_strength",
    "saccade_strength",
    "right_eye_rot_x_offset",
    "right_eye_rot_y_offset",
    "left_eye_rot_x_offset",
    "left_eye_rot_y_offset",
    "eye_saccade_seed",
}


def test_worker_schema_uses_model_reported_values() -> None:
    for expression in (
        "network.GetEmotionName(index)",
        "network.GetDefaultEmotion()",
        "params->data.poseNames[index]",
        '{"channels", std::move(output_channels)}',
    ):
        assert expression in SOURCE
    assert "kArkit52SdkNames" not in SOURCE
    assert '"identity_index"' not in SOURCE
    assert '{"identities",' not in SOURCE
    assert '{"parameters",' not in SOURCE
    assert "settings.parameters" not in SOURCE


def test_worker_reports_exact_model_owned_audio2face_defaults() -> None:
    for expression in (
        "nva2f::GetExecutorInputStrength(",
        "nva2f::GetExecutorSkinParameters(",
        "nva2f::GetExecutorEyesParameters(",
        '{"audio2face_defaults",',
    ):
        assert expression in SOURCE

    serializer = re.search(
        r"json audio2face_settings_json\(.*?\n\}", SOURCE, flags=re.DOTALL
    )
    assert serializer is not None
    serialized_keys = set(re.findall(r'\{"([a-z0-9_]+)"', serializer.group(0)))
    assert serialized_keys == AUDIO2FACE_KEYS
    assert "settings.skin.blinkOffset" not in serializer.group(0)
    assert "static_cast<std::size_t>(settings.eyes.saccadeSeed)" in (
        serializer.group(0)
    )

    defaults = SOURCE[SOURCE.index("Audio2FaceSettings audio2face_defaults;") :]
    defaults = defaults[: defaults.index("audio2face_defaults_ = audio2face_defaults;")]
    for getter in (
        "GetExecutorInputStrength",
        "GetExecutorSkinParameters",
        "GetExecutorEyesParameters",
    ):
        assert re.search(rf"{getter}\(\s*geometry_executor\(\),", defaults)


def test_stream_executors_validate_geometry_and_emotion_output_schema() -> None:
    setup = re.search(
        r"  void ensure_stream_executors\(\).*?(?=\n  void require_model_locked\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert setup is not None
    setup_source = setup.group(0)
    assert "GetNetworkInfo().GetEmotionsCount()" not in setup_source
    assert "emotion_executor_->GetEmotionsSize() !=" in setup_source
    assert "results.emotions.Size() != capture.emotion_count" in SOURCE
    set_option = setup_source.index("stream_geometry.SetExecutionOption(")
    get_option = setup_source.index("stream_geometry.GetExecutionOption()")
    model_invalid = setup_source.index('"model_invalid"', get_option)
    eyes_size = setup_source.index("stream_geometry.GetEyesRotationSize()")
    callback = setup_source.index("SetExecutorGeometryResultsCallback(")
    assert set_option < get_option < model_invalid < eyes_size < callback


def test_stream_snapshot_validates_and_applies_exact_skin_eyes_controls() -> None:
    parser = re.search(
        r"  Audio2FaceSettings parse_audio2face_settings\(.*?"
        r"(?=\n  EmotionDriver parse_emotion_driver\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert parser is not None
    parser_source = parser.group(0)
    exact_keys = re.search(
        r"require_exact_keys\(\s*value,\s*\{(.*?)\},\s*"
        r'"settings\.audio2face"\);',
        parser_source,
        flags=re.DOTALL,
    )
    assert exact_keys is not None
    assert set(re.findall(r'"([a-z0-9_]+)"', exact_keys.group(1))) == (
        AUDIO2FACE_KEYS
    )
    assert "Audio2FaceSettings parsed = audio2face_defaults_;" in parser_source
    assert "parsed.skin.blinkOffset" not in parser_source
    assert re.search(
        r'required_size_in_range\(\s*value, "eye_saccade_seed", '
        r'"settings\.audio2face\.", 0, 4999\)',
        parser_source,
    )

    expected_ranges = {
        "input_strength": (0.0, 3.0),
        "lower_face_smoothing": (0.0, 0.1),
        "upper_face_smoothing": (0.0, 0.1),
        "lower_face_strength": (0.0, 2.0),
        "upper_face_strength": (0.0, 2.0),
        "face_mask_level": (0.0, 1.0),
        "face_mask_softness": (0.001, 0.5),
        "skin_strength": (0.0, 2.0),
        "blink_strength": (0.0, 2.0),
        "blink_offset": (0.0, 1.0),
        "eyelid_open_offset": (-1.0, 1.0),
        "lip_open_offset": (-0.2, 0.2),
        "eyeballs_strength": (0.0, 2.0),
        "saccade_strength": (0.0, 2.0),
        "right_eye_rot_x_offset": (-10.0, 10.0),
        "right_eye_rot_y_offset": (-10.0, 10.0),
        "left_eye_rot_x_offset": (-10.0, 10.0),
        "left_eye_rot_y_offset": (-10.0, 10.0),
        "eye_saccade_seed": (0.0, 4999.0),
    }
    validator = re.search(
        r"void validate_audio2face_settings\(.*?(?=\n\})\n\}",
        SOURCE,
        flags=re.DOTALL,
    )
    assert validator is not None
    found_ranges = {
        name: (float(minimum), float(maximum))
        for name, minimum, maximum in re.findall(
            r'require\([^,]+,\s*"([a-z0-9_]+)",\s*'
            r"(-?[0-9.]+)F,\s*(-?[0-9.]+)F\)",
            validator.group(0),
        )
    }
    assert found_ranges == expected_ranges

    configure = re.search(
        r"  void configure_audio2face_input\(.*?"
        r"(?=\n  void configure_stream_emotion\()", SOURCE, flags=re.DOTALL
    )
    assert configure is not None
    configure_source = configure.group(0)
    for expression in (
        "IFaceExecutorAccessorInputStrength",
        "IFaceExecutorAccessorSkinParameters",
        "IFaceExecutorAccessorEyesParameters",
        "accessor.SetInputStrength(input_strength)",
        "skin.Set(0, settings.skin)",
        "eyes.Set(0, settings.eyes)",
        "configure_audio2face_input(settings.input_strength);",
        "configure_audio2face_postprocess(settings);",
    ):
        assert expression in configure_source
    assert "SetExecutorTongueParameters" not in SOURCE
    assert "SetExecutorTeethParameters" not in SOURCE


def test_output_keeps_model_order_and_named_eye_resolution() -> None:
    assert "stream_executor.GetWeightCount() != kArkit52ChannelCount" in SOURCE
    assert "copy_finite_values(\n        *pending.weights" in SOURCE
    assert not re.search(r"weights\[\d+\]\s*=", SOURCE)
    resolver = re.search(
        r"ArkitEyeLookIndices resolve_arkit_eye_look_indices\(.*?\n\}",
        SOURCE,
        flags=re.DOTALL,
    )
    assert resolver is not None
    assert len(set(re.findall(r'"(eyeLook[A-Za-z]+)"', resolver.group(0)))) == 8


def test_stream_frames_carry_the_model_channel_order() -> None:
    assert '{"channels", std::move(output_channels)}' in SOURCE
    assert "capture.weight_count = executor().GetWeightCount()" in SOURCE
    assert "capture.emotion_count = emotion_channels_.size()" in SOURCE
    assert "copy_finite_values(\n        *pending.weights" in SOURCE
    assert "frame_callback(make_stream_frame(" in SOURCE


def test_worker_uses_one_compositional_emotion_driver() -> None:
    assert re.search(
        r"struct EmotionDriver\s*\{\s*float emotion_strength.*?"
        r"optional<GeneratedEmotionSettings> generated.*?"
        r"optional<PreferredEmotionSettings> preferred.*?\};",
        SOURCE,
        flags=re.DOTALL,
    )
    assert "#include <variant>" not in SOURCE
    assert not re.search(r"(?:Manual|Automatic)EmotionDriver", SOURCE)

    configure = re.search(
        r"  void configure_stream_emotion\(.*?"
        r"(?=\n  static std::size_t settings_index_at\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert configure is not None
    configure_source = configure.group(0)
    for expression in (
        "emotion_postprocess_accessor()",
        "settings.generated",
        "parameters.emotionStrength",
        "parameters.emotionContrast",
        "parameters.maxEmotions",
        "parameters.liveBlendCoef",
        "parameters.liveTransitionTime",
        "parameters.enablePreferredEmotion",
        "settings.preferred.has_value()",
        "parameters.preferredEmotionStrength",
        "parameters.preferredEmotion =",
        "*settings.preferred",
        "preferred.values.data()",
        "accessor.Get(0, parameters)",
        "accessor.Set(0, parameters)",
    ):
        assert expression in configure_source
    assert "value * preferred.strength" in configure_source

    assert '{"audio2face", "emotion_driver"}' in SOURCE
    assert '{"emotion_strength", "generated", "preferred"}' in SOURCE
    assert '{"emotion_contrast", "max_emotions", "live_blend_coef",' in SOURCE
    assert '{"values", "strength"}' in SOURCE
    assert re.search(
        r'"emotion_strength", "settings\.emotion_driver\.",\s*0\.0F, 2\.0F\)',
        SOURCE,
    )
    assert "if (!generated.is_null())" in SOURCE
    assert "if (!preferred.is_null())" in SOURCE
    assert "settings.emotion_driver.generated." in SOURCE
    assert "settings.emotion_driver.preferred.values" in SOURCE
    assert "value.size() != emotion_channels_.size()" in SOURCE
    assert "amount < 0.0F || amount > 1.0F" in SOURCE


def test_stream_uses_incremental_regular_executors() -> None:
    start = re.search(
        r"  json stream_start\(.*?(?=\n  void stream_chunk\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert start is not None
    assert '{"prebuffer_samples", prebuffer_samples_}' in start.group(0)

    begin = re.search(
        r"  void begin_operation\(.*?(?=\n  void finish_operation\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert begin is not None
    begin_source = begin.group(0)
    assert "ensure_stream_executors();" in begin_source
    assert "reset_stream_inference(parse_settings(settings));" in begin_source

    chunk = re.search(
        r"  void stream_chunk\(.*?(?=\n  void stream_settings\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert chunk is not None
    chunk_source = chunk.group(0)
    assert "accumulate_audio(audio.data(), audio.size());" in chunk_source
    assert "drain_ready(canceled, frame);" in chunk_source
    assert "interactive" not in chunk_source
    drain = re.search(
        r"  void drain_interleaved_ready\(.*?"
        r"(?=\n  std::size_t execute_generated_emotion_once\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert drain is not None
    assert "GetNbReadyTracks(executor())" in drain.group(0)
    assert "GetNbReadyTracks(*emotion_executor_)" in drain.group(0)
    assert "evaluate_interactive_stream" not in SOURCE


def test_stream_settings_update_executors_without_reset_or_audio_replay() -> None:
    update = re.search(
        r"  void stream_settings\(.*?(?=\n  void stream_end\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert update is not None
    update_source = update.group(0)
    assert "InferenceSettings parsed = parse_settings(settings);" in update_source
    assert "configure_audio2face(parsed.audio2face);" in update_source
    assert "configure_stream_emotion(parsed.emotion_driver);" in update_source
    for removed in ("Reset(", "reset_stream_inference", "accumulate_audio", "retain_audio"):
        assert removed not in update_source


def test_track_reuses_regular_executors_and_retains_uploaded_audio() -> None:
    track_start = SOURCE[SOURCE.index("  void track_start(") :]
    track_start = track_start[: track_start.index("  void track_chunk(")]
    assert "ensure_stream_executors();" in track_start

    track_chunk = SOURCE[SOURCE.index("  void track_chunk(") :]
    track_chunk = track_chunk[: track_chunk.index("  void track_prepare(")]
    assert "track_audio_.insert(" in track_chunk

    track_prepare = SOURCE[SOURCE.index("  void track_prepare(") :]
    track_prepare = track_prepare[: track_prepare.index("  std::size_t track_render(")]
    assert "track_audio_samples_ = track_audio_.size();" in track_prepare
    for obsolete in (
        "clear_stream_executors",
        "interactive",
        "accumulate_audio",
        "ComputeAllFrames",
    ):
        assert obsolete not in track_start + track_prepare


def test_track_render_replays_audio_with_sample_scheduled_settings() -> None:
    render = re.search(
        r"  std::size_t compute_track_render\(.*?(?=\n  void accumulate_audio\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert render is not None
    source = render.group(0)
    for expression in (
        "for (const TrackSettingsEntry& entry : request.settings_timeline)",
        "reset_stream_inference(settings_timeline.front().settings);",
        "accumulate_audio(track_audio_.data(), track_audio_.size());",
        "bundle_->GetAudioAccumulator(0).Close()",
        "advance_face_input_settings(settings_schedule, timestamp);",
        "advance_face_postprocess_settings(settings_schedule, timestamp);",
        "advance_emotion_postprocess_settings(settings_schedule, timestamp);",
        "nva2x::GetNbReadyTracks(executor())",
        "nva2x::GetNbReadyTracks(*emotion_executor_)",
        "bool preview_emitted = false;",
        "frame.timestamp_sample >= *request.preview_sample",
        "preview(sample_track_frames(candidate, *request.preview_sample));",
        "preview_emitted = true;",
        "if (request.preview_sample && !preview_emitted && !superseded())",
        "cache(candidate);",
        "return superseded() ? 0 : candidate.size();",
    ):
        assert expression in source
    assert source.count(
        "preview(sample_track_frames(candidate, *request.preview_sample));"
    ) == 2
    tail = source[source.rindex("if (request.preview_sample && !preview_emitted") :]
    guarded_preview = tail.index("!preview_emitted && !superseded()")
    preview = tail.index(
        "preview(sample_track_frames(candidate, *request.preview_sample));"
    )
    cache_guard = tail.index("if (superseded()) return 0;", preview)
    cache = tail.index("cache(candidate);", cache_guard)
    result = tail.index("return superseded() ? 0 : candidate.size();", cache)
    assert guarded_preview < preview < cache_guard < cache < result
    for obsolete in ("interactive", "ComputeAllFrames", "Interrupt("):
        assert obsolete not in source


def test_animated_postprocessing_advances_at_sdk_frame_callbacks() -> None:
    geometry = re.search(
        r"  static bool geometry_callback\(.*?(?=\n  static bool weights_callback\()",
        SOURCE,
        flags=re.DOTALL,
    )
    emotion = re.search(
        r"  static bool generated_emotion_callback\(.*?"
        r"(?=\n  static void effective_emotions_callback\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert geometry is not None
    assert emotion is not None
    assert (
        "advance_face_postprocess_settings(\n"
        "            *capture.settings_schedule, results.timeStampNextFrame);"
        in geometry.group(0)
    )
    assert (
        "advance_emotion_postprocess_settings(\n"
        "            *capture.settings_schedule, results.timeStampNextFrame);"
        in emotion.group(0)
    )
    assert "capture.callback_error = std::current_exception();" in geometry.group(0)
    assert "capture.callback_error = std::current_exception();" in emotion.group(0)
