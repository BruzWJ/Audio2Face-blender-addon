from __future__ import annotations

import re
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "worker" / "src" / "a2f_backend.cpp"
).read_text(encoding="utf-8")


def test_worker_schema_uses_model_reported_values() -> None:
    for expression in (
        "network.GetEmotionName(index)",
        "network.GetDefaultEmotion()",
        "params->data.poseNames[index]",
        '{"channels", output_channels_}',
    ):
        assert expression in SOURCE
    assert "kArkit52SdkNames" not in SOURCE
    assert '"identity_index"' not in SOURCE
    assert '{"identities",' not in SOURCE
    assert '{"parameters",' not in SOURCE
    assert "settings.parameters" not in SOURCE


def test_output_keeps_model_order_and_named_eye_resolution() -> None:
    assert "executor.GetWeightCount() != output_channels_.size()" in SOURCE
    assert "pending.weights->Data()[channel]" in SOURCE
    assert not re.search(r"weights\[\d+\]\s*=", SOURCE)
    resolver = re.search(
        r"ArkitEyeLookIndices resolve_arkit_eye_look_indices\(.*?\n\}",
        SOURCE,
        flags=re.DOTALL,
    )
    assert resolver is not None
    assert len(set(re.findall(r'"(eyeLook[A-Za-z]+)"', resolver.group(0)))) == 8


def test_results_carry_the_model_channel_order() -> None:
    assert '{{"schema", "a2f-animation/2"}' in SOURCE
    assert '{"channels", output_channels_}' in SOURCE
    assert "kMaximumResultScalars / output_channels_.size()" in SOURCE


def test_worker_configures_the_sdk_from_one_optional_preferred_snapshot() -> None:
    begin_operation = re.search(
        r"  void begin_operation\(.*?(?=\n  void finish_operation\()",
        SOURCE,
        flags=re.DOTALL,
    )
    assert begin_operation is not None
    begin_source = begin_operation.group(0)
    assert begin_source.index("emotion_executor_->Reset(0)") < begin_source.index(
        "configure_automatic_emotion();"
    )

    configure = re.search(
        r"  void configure_automatic_emotion\(\).*?(?=\n  void copy_emotion_channels\()",
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
    assert "if (!preferred_value->is_null())" in SOURCE
    assert "parse_emotion_snapshot(" in SOURCE
    assert "value.size() != emotion_channels_.size()" in SOURCE
    assert "amount < 0.0F || amount > 1.0F" in SOURCE
    assert "manual_emotion_.data()" not in configure_source
    assert "fully replaces the manual vector" not in SOURCE
