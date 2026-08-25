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
        assert re.search(rf"{getter}\(\s*geometry,", defaults)


def test_audio2emotion_compatibility_uses_postprocessed_output() -> None:
    setup = SOURCE[SOURCE.index("const auto emotion_model_info") :]
    setup = setup[: setup.index("Installing Audio2Emotion callback")]
    assert "GetNetworkInfo().GetEmotionsCount()" not in setup
    assert "emotion_executor_->GetEmotionsSize() != emotion_channels_.size()" in setup
    assert "results.emotions.Size() != capture.emotion_count" in SOURCE


def test_stream_snapshot_validates_and_applies_exact_skin_eyes_controls() -> None:
    parser = re.search(
        r"  Audio2FaceSettings parse_audio2face_settings\(.*?"
        r"(?=\n  void configure_audio2face\()",
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
        r"  void configure_audio2face\(.*?(?=\n  void apply_settings\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert configure is not None
    for setter in (
        "nva2f::SetExecutorInputStrength(",
        "nva2f::SetExecutorSkinParameters(",
        "nva2f::SetExecutorEyesParameters(",
    ):
        assert setter in configure.group(0)
    assert "auto& geometry = geometry_executor();" in configure.group(0)
    assert not re.search(r"SetExecutor\w+Parameters\(executor\(\)", configure.group(0))
    assert "SetExecutorTongueParameters" not in SOURCE
    assert "SetExecutorTeethParameters" not in SOURCE

    reset = re.search(
        r"  void reset_inference\(.*?(?=\n  void begin_operation\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert reset is not None
    assert reset.group(0).index("executor().Reset(0)") < reset.group(0).index(
        "apply_settings(settings)"
    )


def test_output_keeps_model_order_and_named_eye_resolution() -> None:
    assert "executor.GetWeightCount() != output_channels.size()" in SOURCE
    assert "pending.weights->Data()[channel]" in SOURCE
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
    assert "pending.weights->Data()[channel]" in SOURCE
    assert "pending.emotions->Data()[index]" in SOURCE
    assert (
        "frame_callback(StreamFrame{stream_timestamp, std::move(arkit)," in SOURCE
    )
    assert (
        "std::move(emotions)})" in SOURCE
    )


def test_arkit_solve_uses_the_model_owned_default_identity() -> None:
    assert "constexpr std::size_t kDefaultIdentityIndex = 0" in SOURCE
    assert "ReadDiffusionBlendshapeSolveModelInfo(" in SOURCE
    assert "blendshape_parameters.initializationSkinParams" in SOURCE
    assert "ReadDiffusionBlendshapeSolveExecutorBundle(" in SOURCE
    assert "kDefaultIdentityIndex, true" in SOURCE


def test_callbacks_follow_the_sdk_result_stream_contract() -> None:
    restore = SOURCE.index("geometry.SetExecutionOption(execution_option)")
    geometry_callback = SOURCE.index("SetExecutorGeometryResultsCallback(")
    emotions_callback = SOURCE.index("executor.SetEmotionsCallback(")
    weights_callback = SOURCE.index("executor.SetResultsCallback(")
    assert restore < geometry_callback < emotions_callback < weights_callback
    assert "results.eyesRotation, results.eyesCudaStream" in SOURCE
    assert "results.emotions, results.cudaStream" in SOURCE
    assert "results.weights, results.cudaStream" in SOURCE
    assert "results.eyesCudaStream !=" not in SOURCE
    assert "results.emotions.Size() != capture.emotion_count" in SOURCE
    assert "results.cudaStream !=" not in SOURCE


def test_worker_configures_the_sdk_from_one_optional_preferred_snapshot() -> None:
    reset_inference = re.search(
        r"  void reset_inference\(.*?(?=\n  void begin_operation\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert reset_inference is not None
    reset_source = reset_inference.group(0)
    assert reset_source.index("emotion_executor_->Reset(0)") < reset_source.index(
        "configure_automatic_emotion();"
    )

    configure = re.search(
        r"  void configure_automatic_emotion\(\).*?(?=\n  json load_emotion_channels\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert configure is not None
    configure_source = configure.group(0)
    for expression in (
        "nva2e::GetExecutorPostProcessParameters",
        "parameters.emotionStrength",
        "parameters.emotionContrast",
        "parameters.maxEmotions",
        "parameters.liveBlendCoef",
        "parameters.liveTransitionTime",
        "parameters.enablePreferredEmotion",
        "automatic_emotion_settings_.preferred_emotion.has_value()",
        "parameters.preferredEmotionStrength",
        "parameters.preferredEmotion =",
        "*automatic_emotion_settings_.preferred_emotion",
        "nva2e::SetExecutorPostProcessParameters",
    ):
        assert expression in configure_source

    assert '"preferred_emotion", "preferred_emotion_strength"' in SOURCE
    assert re.search(
        r'"emotion_strength", "settings\.audio2emotion\.", 0\.0F,\s*2\.0F\)',
        SOURCE,
    )
    assert "if (!preferred_value->is_null())" in SOURCE
    assert "parse_emotion_snapshot(" in SOURCE
    assert "value.size() != emotion_channels_.size()" in SOURCE
    assert "amount < 0.0F || amount > 1.0F" in SOURCE


def test_active_settings_replay_bounded_pcm_on_one_absolute_timeline() -> None:
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
    assert "retained_audio_capacity_ = prebuffer_samples_ + sample_rate_;" in begin_source

    update = re.search(
        r"  void stream_settings\(.*?(?=\n  void stream_end\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert update is not None
    update_source = update.group(0)
    for earlier, later in (
        ("reset_inference(settings);", "previous_timestamp_.reset();"),
        ("previous_timestamp_.reset();", "reset();"),
        ("reset();", "const std::vector<float> replay"),
        ("accumulate_audio(replay.data(), replay.size());", "drain_ready"),
    ):
        assert update_source.index(earlier) < update_source.index(later)
    assert re.search(
        r"timestamp_offset_ =\s*total_audio_samples_ -\s*"
        r"static_cast<std::int64_t>\(retained_audio_\.size\(\)\);",
        update_source,
    )

    assert "retain_audio(audio);" in SOURCE
    assert "while (retained_audio_.size() > retained_audio_capacity_)" in SOURCE
    assert "return timestamp_offset_ + local_timestamp;" in SOURCE
    assert "previous_timestamp_ = stream_timestamp;" in SOURCE
