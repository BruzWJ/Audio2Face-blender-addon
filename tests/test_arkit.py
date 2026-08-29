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


def test_audio2emotion_output_matches_the_effective_emotion_schema() -> None:
    setup = SOURCE[SOURCE.index("emotion_model_info_ = require_sdk_ptr") :]
    setup = setup[: setup.index("Installing interactive Audio2Emotion callback")]
    assert "GetNetworkInfo().GetEmotionsCount()" not in setup
    assert "interactive_emotion_executor_->GetEmotionsSize() !=" in setup
    assert "results.emotions.Size() != capture.emotion_count" in SOURCE


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
        r"  void configure_interactive_audio2face\(.*?"
        r"(?=\n  void configure_interactive_generated_emotion\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert configure is not None
    for setter in (
        "nva2f::SetInteractiveExecutorInputStrength(",
        "nva2f::SetInteractiveExecutorSkinParameters(",
        "nva2f::SetInteractiveExecutorEyesParameters(",
    ):
        assert setter in configure.group(0)
    assert "SetExecutorTongueParameters" not in SOURCE
    assert "SetExecutorTeethParameters" not in SOURCE


def test_output_keeps_model_order_and_named_eye_resolution() -> None:
    assert (
        "interactive_executor_->GetWeightCount() != kArkit52ChannelCount"
        in SOURCE
    )
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
    assert "capture.weight_count = interactive_executor_->GetWeightCount()" in SOURCE
    assert "capture.weight_count = executor().GetWeightCount()" in SOURCE
    assert "capture.emotion_count = emotion_channels_.size()" in SOURCE
    assert "copy_finite_values(\n        *pending.weights" in SOURCE
    assert "interactive_effective_emotions_" in SOURCE
    assert "frame_callback(make_stream_frame(" in SOURCE


def test_interactive_arkit_solve_initializes_all_sdk_postprocessors() -> None:
    assert "constexpr std::size_t kDefaultIdentityIndex = 0" in SOURCE
    assert "ReadDiffusionBlendshapeSolveExecutorBundle(" in SOURCE
    assert "&geometry_model_info" in SOURCE
    assert "&blendshape_model_info" in SOURCE
    assert "ExecutionOption::All" in SOURCE
    assert "ExecutionOption::Skin |" in SOURCE
    interactive_setup = SOURCE[SOURCE.index("  void ensure_interactive_executors()") :]
    interactive_setup = interactive_setup[: interactive_setup.index("  void require_model_locked()")]
    assert "ExecutionOption::All" in interactive_setup
    assert "ExecutionOption::Skin," in interactive_setup
    assert "blendshape_parameters.initializationSkinParams" in SOURCE
    assert "kDefaultIdentityIndex, true" in SOURCE
    assert "CreateDeviceBlendshapeSolveInteractiveExecutor(" in SOURCE


def test_callbacks_follow_the_sdk_result_stream_contract() -> None:
    geometry_callback = SOURCE.index("SetInteractiveExecutorGeometryResultsCallback(")
    weights_callback = SOURCE.index("interactive_executor_->SetResultsCallback(")
    emotions_callback = SOURCE.index(
        "interactive_emotion_executor_->SetResultsCallback("
    )
    assert geometry_callback < weights_callback < emotions_callback
    assert "results.eyesRotation, results.eyesCudaStream" in SOURCE
    assert "results.emotions, results.cudaStream" in SOURCE
    assert "results.weights, results.cudaStream" in SOURCE
    assert "results.eyesCudaStream !=" not in SOURCE
    assert "results.emotions.Size() != capture.emotion_count" in SOURCE
    assert "results.cudaStream !=" not in SOURCE


def test_interactive_audio2emotion_matches_the_one_track_engine_profile() -> None:
    creation = re.search(
        r"CreateClassifierEmotionInteractiveExecutor\(\s*"
        r"emotion_parameters, classifier_parameters, 1\)",
        SOURCE,
    )
    assert creation is not None


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

    prepare = SOURCE[SOURCE.index("  bool prepare_interactive_settings(") :]
    prepare = prepare[: prepare.index("  static std::vector<float> copy_finite_values(")]
    assert "const bool generated = parsed.emotion_driver.generated.has_value();" in prepare
    assert "if (generated)" in prepare
    assert "install_interactive_driver_emotions();" in prepare
    driver = SOURCE[SOURCE.index("  void install_interactive_driver_emotions(") :]
    driver = driver[: driver.index("  void configure_interactive_generated_emotion(")]
    assert "emotion(emotion_channels_.size(), 0.0F)" in driver
    assert "interactive_emotion_driver_.emotion_strength *" in driver

    configure = re.search(
        r"  void configure_interactive_generated_emotion\(.*?"
        r"(?=\n  static bool render_is_superseded\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert configure is not None
    configure_source = configure.group(0)
    for expression in (
        "nva2e::GetInteractiveExecutorPostProcessParameters",
        "interactive_emotion_driver_.generated.value()",
        "parameters.emotionStrength",
        "parameters.emotionContrast",
        "parameters.maxEmotions",
        "parameters.liveBlendCoef",
        "parameters.liveTransitionTime",
        "parameters.enablePreferredEmotion",
        "interactive_emotion_driver_.preferred.has_value()",
        "parameters.preferredEmotionStrength",
        "parameters.preferredEmotion =",
        "*interactive_emotion_driver_.preferred",
        "preferred.values.data()",
        "nva2e::SetInteractiveExecutorPostProcessParameters",
    ):
        assert expression in configure_source

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
    assert "reset_stream_inference(settings);" in begin_source

    chunk = re.search(
        r"  void stream_chunk\(.*?(?=\n  void stream_end\()",
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
        r"(?=\n  void execute_generated_emotion_once\()",
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


def test_track_start_swaps_to_the_interactive_executor_family() -> None:
    track_start = SOURCE[SOURCE.index("  void track_start(") :]
    track_start = track_start[: track_start.index("  void track_chunk(")]
    assert track_start.index("clear_stream_executors();") < track_start.index(
        "ensure_interactive_executors();"
    )

def test_track_path_uses_supported_interactive_setters_and_closed_inputs() -> None:
    for expression in (
        "CreateDiffusionGeometryInteractiveExecutor(",
        "CreateDeviceBlendshapeSolveInteractiveExecutor(",
        "CreateClassifierEmotionInteractiveExecutor(",
        "SetInteractiveExecutorInputStrength(",
        "SetInteractiveExecutorSkinParameters(",
        "SetInteractiveExecutorEyesParameters(",
        "SetInteractiveExecutorPostProcessParameters(",
        "interactive_audio_accumulator_->Close()",
        "interactive_emotion_accumulator_->Close()",
        "executor.ComputeAllFrames()",
        "generate_interactive_emotions(revision, canceled",
        "active_interactive_compute_->Interrupt()",
        '"Synchronizing streaming audio upload"',
        '"Synchronizing interactive audio upload"',
        '"Synchronizing constant interactive emotions"',
    ):
        assert expression in SOURCE
    assert "ComputeFrame" not in SOURCE


def test_track_render_builds_one_continuous_timestamped_cache() -> None:
    prepare = re.search(
        r"  void track_prepare\(.*?(?=\n  std::size_t track_render\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert prepare is not None
    prepare_source = prepare.group(0)
    assert "refill_interactive_audio(track_audio_);" in prepare_source
    assert "install_constant_interactive_emotions(" not in prepare_source
    assert "ComputeFrame" not in prepare_source

    track = re.search(
        r"  std::size_t track_render\(.*?(?=\n  void interrupt_operation\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert track is not None
    assert "compute_track_render(request, canceled, latest_revision" in track.group(0)

    render = re.search(
        r"  std::size_t compute_track_render\(.*?(?=\n  void accumulate_audio\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert render is not None
    source = render.group(0)
    assert "prepare_interactive_settings(request.settings, request.revision" in source
    assert "compute_interactive(*interactive_executor_, canceled," in source
    assert "candidate.reserve(capture.frames.size());" in source
    assert "preview(sample_track_frames(candidate, *request.preview_sample));" in source
    assert "cache(candidate);" in source
    assert "return superseded() ? 0 : candidate.size();" in source

    assert "emotion_settings == interactive_emotion_settings_" in SOURCE
    assert "interactive_emotions_valid_ = true;" in SOURCE


def test_generated_emotion_is_a_continuous_timestamped_pass() -> None:
    emotion = re.search(
        r"  bool generate_interactive_emotions\(.*?"
        r"(?=\n  static std::vector<float> interpolate_values\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert emotion is not None
    source = emotion.group(0)
    assert "compute_interactive(*interactive_emotion_executor_" in source
    assert "interactive_emotion_accumulator_->Close()" in source
    assert "for (auto& [timestamp, frame] : emotion_frames)" in source
    assert "interactive_effective_emotions_.swap(effective_emotions);" in source
