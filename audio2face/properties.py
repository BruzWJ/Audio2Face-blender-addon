"""Blender RNA state and model-tunable Audio2Face parameters."""

from __future__ import annotations

import math
from typing import Any

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .result_io import ResultValidationError, validate_output_channels


STATUS_ITEMS = (
    ("IDLE", "Idle", "No worker operation is active"),
    ("STARTING", "Starting", "Starting and handshaking with the worker"),
    ("LOADING_MODEL", "Loading Models", "Loading Audio2Face and Audio2Emotion"),
    ("MODEL_READY", "Model Ready", "Model is ready for generation"),
    ("GENERATING", "Generating", "Generating timestamped model channel values"),
    ("CANCELLING", "Cancelling", "Cancellation has been requested"),
    ("COMPLETED", "Completed", "A generated model channel stream is available"),
    ("STREAM_STARTING", "Starting Stream", "Preparing incremental PCM inference"),
    ("STREAMING", "Streaming", "Incremental PCM is driving model channel values"),
    ("STREAM_ENDING", "Ending Stream", "Draining the stream's final model frames"),
    ("STOPPING", "Stopping", "Worker is shutting down"),
    ("ERROR", "Error", "The last operation failed"),
)


def _mesh_object(_self: Any, obj: bpy.types.Object) -> bool:
    return obj.type == "MESH"


class A2FTargetMeshItem(bpy.types.PropertyGroup):
    """One mesh subscriber to the model-provided frame bus."""

    enabled: BoolProperty(name="Enabled", default=True)
    object: PointerProperty(
        name="Face Mesh",
        description="Mesh whose Shape Keys receive model output values when available",
        type=bpy.types.Object,
        poll=_mesh_object,
    )


class A2FEmotionValueItem(bpy.types.PropertyGroup):
    """One model-defined manual emotion driver channel."""

    name: StringProperty(
        name="Emotion",
        description="Model-defined Audio2Face emotion channel",
        default="",
        options={"HIDDEN"},
    )
    value: FloatProperty(
        name="Value",
        description="Manual value supplied to this model emotion channel",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )


class A2FModelParameterItem(bpy.types.PropertyGroup):
    """One worker-advertised, model-tunable numeric parameter."""

    path: StringProperty(
        name="Path",
        description="Opaque worker parameter identifier",
        default="",
        options={"HIDDEN"},
    )
    kind: StringProperty(
        name="Kind",
        description="Worker parameter type: float or integer",
        default="",
        options={"HIDDEN"},
    )
    float_value: FloatProperty(
        name="Value",
        description="Floating-point model parameter value",
        default=0.0,
    )
    int_value: IntProperty(
        name="Value",
        description="Integer model parameter value",
        default=0,
    )


class A2FModelIdentityItem(bpy.types.PropertyGroup):
    """One identity reported by the loaded Audio2Face model."""

    name: StringProperty(name="Identity", default="", options={"HIDDEN"})


class A2FSceneSettings(bpy.types.PropertyGroup):
    input_mode: EnumProperty(
        name="Input Mode",
        description="Process a complete selected WAV or infer from incremental PCM",
        items=(
            ("SELECTED", "Selected WAV", "Generate from a complete selected WAV file"),
            ("STREAM", "Stream", "Drive Shape Keys from incremental mono float PCM"),
        ),
        default="SELECTED",
    )
    audio_path: StringProperty(
        name="Speech WAV",
        description="WAV used by Selected mode or by the built-in streamed-WAV source",
        subtype="FILE_PATH",
    )
    identity_index: IntProperty(
        name="Identity Index",
        description="Identity selected from the loaded model",
        default=0,
        min=0,
    )
    model_identities: CollectionProperty(type=A2FModelIdentityItem)

    auto_audio2emotion: BoolProperty(
        name="Auto Audio2Emotion",
        description=(
            "Override the manual emotion driver with emotion values inferred "
            "from the input audio"
        ),
        default=False,
    )
    manual_emotions: CollectionProperty(type=A2FEmotionValueItem)
    model_parameters: CollectionProperty(type=A2FModelParameterItem)

    target_meshes: CollectionProperty(type=A2FTargetMeshItem)
    target_mesh_index: IntProperty(default=0, min=0)

    preview_loop: BoolProperty(
        name="Loop",
        description="Loop selected audio and its model-provided channel stream",
        default=False,
    )
    preview_reset_on_stop: BoolProperty(
        name="Reset on Stop",
        description="Set driven shape keys to zero when preview playback stops",
        default=True,
    )
    preview_volume: FloatProperty(
        name="Volume",
        description="Selected or streamed WAV playback volume",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    preview_state: EnumProperty(
        items=(
            ("IDLE", "Idle", "Preview is stopped"),
            ("PLAYING", "Playing", "Preview audio and Shape Keys are playing"),
            ("PAUSED", "Paused", "Preview is paused on its current values"),
        ),
        default="IDLE",
        options={"HIDDEN", "SKIP_SAVE"},
    )
    preview_time: FloatProperty(
        name="Preview Time", default=0.0, min=0.0, unit="TIME", options={"SKIP_SAVE"}
    )
    preview_duration: FloatProperty(
        name="Preview Duration", default=0.0, min=0.0, unit="TIME", options={"SKIP_SAVE"}
    )
    stream_reset_on_stop: BoolProperty(
        name="Reset on Stop",
        description="Set driven Shape Keys to zero when a live stream stops",
        default=True,
    )
    stream_id: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    stream_sample_rate: IntProperty(
        name="Stream Sample Rate", default=0, min=0, options={"HIDDEN", "SKIP_SAVE"}
    )
    stream_prebuffer_samples: IntProperty(
        name="Stream Prebuffer Samples",
        default=0,
        min=0,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    stream_time: FloatProperty(
        name="Stream Time", default=0.0, min=0.0, unit="TIME", options={"SKIP_SAVE"}
    )

    status: EnumProperty(
        name="Status", items=STATUS_ITEMS, default="IDLE", options={"SKIP_SAVE"}
    )
    status_message: StringProperty(
        name="Message", default="Worker is stopped", options={"SKIP_SAVE"}
    )
    current_job_id: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    result_path: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    result_audio_path: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    progress: FloatProperty(
        name="Progress",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        options={"SKIP_SAVE"},
    )
_MODEL_SCHEMA_FIELDS = {"identities", "channels", "parameters", "emotion_channels"}
_EMOTION_DESCRIPTOR_FIELDS = {"name", "default"}
_BLENDER_INT_MIN = -(1 << 31)
_BLENDER_INT_MAX = (1 << 31) - 1


def _finite_float_in_range(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    result = _finite_float(value, label=label)
    if result < minimum or result > maximum:
        raise ValueError(f"{label} must be in [{minimum:g}, {maximum:g}]")
    return result


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < _BLENDER_INT_MIN or value > _BLENDER_INT_MAX:
        raise ValueError(f"{label} is outside Blender's integer range")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validated_parameters(
    defaults: object,
) -> list[tuple[str, str, float | int]]:
    if not isinstance(defaults, dict):
        raise ValueError("model_schema.parameters must be an object")
    result: list[tuple[str, str, float | int]] = []
    for raw_path, default in defaults.items():
        path = _nonempty_string(raw_path, label="model_schema parameter path")
        label = f"model_schema.parameters[{path!r}]"
        if isinstance(default, int) and not isinstance(default, bool):
            result.append((path, "integer", _integer(default, label=label)))
        else:
            result.append((path, "float", _finite_float(default, label=label)))
    return result


def _validated_emotion_descriptors(
    descriptors: object,
) -> list[tuple[str, float]]:
    if not isinstance(descriptors, list):
        raise ValueError("model_schema.emotion_channels must be an array")
    result: list[tuple[str, float]] = []
    seen_names: set[str] = set()
    for index, descriptor in enumerate(descriptors):
        location = f"model_schema.emotion_channels[{index}]"
        if not isinstance(descriptor, dict) or set(descriptor) != _EMOTION_DESCRIPTOR_FIELDS:
            raise ValueError(f"{location} has unexpected or missing fields")
        name = _nonempty_string(descriptor["name"], label=f"{location}.name")
        if name in seen_names:
            raise ValueError(f"model_schema contains duplicate emotion name {name!r}")
        seen_names.add(name)
        default = _finite_float_in_range(
            descriptor["default"],
            label=f"{location}.default",
            minimum=0.0,
            maximum=1.0,
        )
        result.append((name, default))
    return result


def _validated_identities(identities: object) -> list[str]:
    if not isinstance(identities, list) or not identities:
        raise ValueError("model_schema.identities must be a non-empty array")
    result = []
    for position, identity in enumerate(identities):
        location = f"model_schema.identities[{position}]"
        result.append(_nonempty_string(identity, label=location))
    return result


def _manual_emotion_values(settings: A2FSceneSettings) -> dict[str, float]:
    return {item.name: float(item.value) for item in settings.manual_emotions}


def _model_parameter_value(parameter: A2FModelParameterItem) -> float | int:
    if parameter.kind == "float":
        return float(parameter.float_value)
    if parameter.kind == "integer":
        return int(parameter.int_value)
    raise ValueError(f"unsupported model parameter kind {parameter.kind!r}")


def _model_parameter_values(settings: A2FSceneSettings) -> dict[str, float | int]:
    return {
        item.path: _model_parameter_value(item)
        for item in settings.model_parameters
    }


def tuning_parameters(settings: A2FSceneSettings) -> dict[str, object]:
    """Return values for exactly the schema advertised by the loaded worker."""

    return {
        "auto_audio2emotion": bool(settings.auto_audio2emotion),
        "manual_emotions": _manual_emotion_values(settings),
        "parameters": _model_parameter_values(settings),
    }


def apply_model_schema(
    settings: A2FSceneSettings,
    model_schema: object,
) -> tuple[str, ...]:
    """Validate and materialize one self-describing worker model schema."""

    if not isinstance(model_schema, dict) or set(model_schema) != _MODEL_SCHEMA_FIELDS:
        raise ValueError("worker returned a noncanonical model_schema object")
    try:
        channels = validate_output_channels(model_schema["channels"])
    except ResultValidationError as exc:
        raise ValueError(f"worker returned invalid output channels: {exc}") from exc
    identities = _validated_identities(model_schema["identities"])
    parameters = _validated_parameters(model_schema["parameters"])
    emotions = _validated_emotion_descriptors(model_schema["emotion_channels"])

    preserved_parameters = {
        (item.path, item.kind): _model_parameter_value(item)
        for item in settings.model_parameters
    }
    preserved_manual = _manual_emotion_values(settings)

    settings.model_parameters.clear()
    for path, kind, default in parameters:
        item = settings.model_parameters.add()
        item.path = path
        item.kind = kind
        value = preserved_parameters.get((path, kind), default)
        if kind == "float":
            item.float_value = float(value)
        else:
            item.int_value = int(value)

    settings.manual_emotions.clear()
    for name, default in emotions:
        item = settings.manual_emotions.add()
        item.name = name
        item.value = preserved_manual.get(name, default)

    selected_identity = settings.identity_index
    settings.model_identities.clear()
    for name in identities:
        item = settings.model_identities.add()
        item.name = name
    if (
        isinstance(selected_identity, bool)
        or not isinstance(selected_identity, int)
        or not 0 <= selected_identity < len(identities)
    ):
        settings.identity_index = 0
    return channels


CLASSES = (
    A2FTargetMeshItem,
    A2FEmotionValueItem,
    A2FModelParameterItem,
    A2FModelIdentityItem,
    A2FSceneSettings,
)
