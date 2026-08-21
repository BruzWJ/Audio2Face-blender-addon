from __future__ import annotations

import re
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "worker" / "src" / "a2f_backend.cpp"
).read_text(encoding="utf-8")


def test_worker_schema_uses_model_reported_values() -> None:
    for expression in (
        "network.GetIdentityName(index)",
        "network.GetEmotionName(index)",
        "network.GetDefaultEmotion()",
        "params->data.poseNames[index]",
        '{"channels", output_channels_}',
        '{"parameters", parameter_schema(parameter_values)}',
    ):
        assert expression in SOURCE
    assert "kArkit52SdkNames" not in SOURCE
    assert 'identities.push_back(name)' in SOURCE


def test_one_typed_adapter_reads_and_writes_sdk_parameters() -> None:
    assert SOURCE.count("constexpr ParameterBinding kParameterBindings[]") == 1
    assert SOURCE.count(
        "for (const ParameterBinding& binding : kParameterBindings)"
    ) >= 3
    for operation in ("InputStrength", "SkinParameters", "PostProcessParameters"):
        assert f"GetExecutor{operation}" in SOURCE
        assert f"SetExecutor{operation}" in SOURCE
    assert 'parameters.contains(binding.path)' in SOURCE
    assert 'parameter_value->at(binding.path)' in SOURCE


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
