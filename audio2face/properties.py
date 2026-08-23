"""Blender RNA state for Audio2Face playback and model-derived channels."""

from __future__ import annotations

import hashlib
import json
import math

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

from .shape_keys import ShapeKeyStreamError, validate_output_channels


STATUS_ITEMS = (
    ("IDLE", "Idle", "No worker operation is active"),
    ("STARTING", "Starting", "Starting and handshaking with the worker"),
    ("LOADING_MODEL", "Loading Models", "Loading Audio2Face and Audio2Emotion"),
    ("MODEL_READY", "Model Ready", "Model is ready for audio-driven inference"),
    ("STREAM_STARTING", "Starting Stream", "Preparing incremental PCM inference"),
    ("STREAMING", "Streaming", "Incremental PCM is driving model channel values"),
    ("STREAM_ENDING", "Ending Stream", "Draining the stream's final model frames"),
    ("STOPPING", "Stopping", "Worker is shutting down"),
    ("ERROR", "Error", "The last operation failed"),
)


class A2FTargetMeshItem(bpy.types.PropertyGroup):
    """One mesh subscriber to the model-provided frame bus."""

    object: PointerProperty(
        name="Face Mesh",
        description="Mesh whose Shape Keys receive model output values when available",
        type=bpy.types.Object,
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


class A2FSceneSettings(bpy.types.PropertyGroup):
    input_mode: EnumProperty(
        name="Input Mode",
        description="Play a selected WAV or receive incremental PCM",
        items=(
            ("SELECTED", "Selected WAV", "Play and infer from a selected WAV file"),
            ("STREAM", "Stream", "Drive Shape Keys from incremental mono float PCM"),
        ),
        default="SELECTED",
    )
    audio_path: StringProperty(
        name="Speech WAV",
        description="WAV played and inferred in Selected mode",
        subtype="FILE_PATH",
    )
    auto_audio2emotion: BoolProperty(
        name="Auto Audio2Emotion",
        description="Infer emotion values from the input audio for this operation",
        default=False,
    )
    manual_emotions: CollectionProperty(type=A2FEmotionValueItem)
    preferred_emotions: CollectionProperty(
        type=A2FEmotionValueItem,
        options={"HIDDEN"},
    )
    a2e_emotion_strength: FloatProperty(
        name="Emotion Strength",
        description="Strength of automatic emotion relative to neutral emotion",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    a2e_emotion_contrast: FloatProperty(
        name="Emotion Contrast",
        description="Increase or reduce the spread between generated emotions",
        default=1.0,
        min=0.1,
        max=3.0,
    )
    a2e_max_emotions: IntProperty(
        name="Max Emotions",
        description="Maximum generated emotions retained at each inference frame",
        default=6,
        min=1,
        max=6,
    )
    a2e_live_blend_coef: FloatProperty(
        name="Smoothing",
        description="Influence of the preceding generated emotion on the next frame",
        default=0.7,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    a2e_transition_smoothing: FloatProperty(
        name="Transition Time",
        description="Time used to transition between automatic emotion states",
        default=0.5,
        min=0.1,
        max=1.0,
        unit="TIME",
    )
    a2e_preferred_emotion_strength: FloatProperty(
        name="Strength",
        description=(
            "Strength of the loaded preferred emotion relative to generated emotion"
        ),
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    model_schema_signature: StringProperty(options={"HIDDEN"})

    target_meshes: CollectionProperty(type=A2FTargetMeshItem)
    target_mesh_index: IntProperty(default=0, min=0)

    playback_loop: BoolProperty(
        name="Loop",
        description="Loop selected audio and its model-provided channel stream",
        default=False,
    )
    prediction_delay: FloatProperty(
        name="Prediction Delay",
        description="Adjust synchronization of mouth motion to audio in seconds",
        default=0.0,
        min=-1.0,
        max=1.0,
        unit="TIME",
    )
    playback_state: EnumProperty(
        items=(
            ("IDLE", "Idle", "Selected audio is stopped"),
            ("PLAYING", "Playing", "Selected audio and Shape Keys are playing"),
            ("PAUSED", "Paused", "Selected audio is paused on its current values"),
        ),
        default="IDLE",
        options={"HIDDEN", "SKIP_SAVE"},
    )
    playback_time: FloatProperty(
        name="Playback Time", default=0.0, min=0.0, unit="TIME", options={"SKIP_SAVE"}
    )
    playback_duration: FloatProperty(
        name="Playback Duration", default=0.0, min=0.0, unit="TIME", options={"SKIP_SAVE"}
    )
    playback_progress: FloatProperty(
        name="Playback Position",
        description="Seek within the selected audio playback",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        options={"SKIP_SAVE"},
    )
    stream_operation_id: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
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


_MODEL_SCHEMA_FIELDS = {"channels", "emotion_channels"}
_EMOTION_DESCRIPTOR_FIELDS = {"name", "default"}


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
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validated_emotion_descriptors(
    descriptors: object,
) -> list[tuple[str, float]]:
    if not isinstance(descriptors, list):
        raise ValueError("model_schema.emotion_channels must be an array")
    result: list[tuple[str, float]] = []
    seen_names: set[str] = set()
    for index, descriptor in enumerate(descriptors):
        location = f"model_schema.emotion_channels[{index}]"
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != _EMOTION_DESCRIPTOR_FIELDS
        ):
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


def _schema_signature(
    model_signature: tuple[str, str],
    model_schema: dict[str, object],
) -> str:
    if (
        not isinstance(model_signature, tuple)
        or len(model_signature) != 2
        or not all(isinstance(value, str) and value for value in model_signature)
    ):
        raise ValueError("model signature is invalid")
    payload = json.dumps(
        {"model": model_signature, "schema": model_schema},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _emotion_values(items: object, *, label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for position, item in enumerate(items):
        name = _nonempty_string(
            item.name,
            label=f"{label} {position} name",
        )
        if name in values:
            raise ValueError(f"{label} collection repeats {name!r}")
        values[name] = _finite_float_in_range(
            item.value,
            label=f"{label} {name!r} value",
            minimum=0.0,
            maximum=1.0,
        )
    return values


def _manual_emotion_values(settings: A2FSceneSettings) -> dict[str, float]:
    return _emotion_values(settings.manual_emotions, label="manual emotion")


def _preferred_emotion_values(settings: A2FSceneSettings) -> dict[str, float]:
    return _emotion_values(settings.preferred_emotions, label="preferred emotion")


def load_preferred_emotion(settings: A2FSceneSettings) -> None:
    """Snapshot the current model-defined manual emotion values."""

    values = _manual_emotion_values(settings)
    if not values:
        raise ValueError("load the Audio2Face model before loading preferred emotion")
    settings.preferred_emotions.clear()
    for name, value in values.items():
        item = settings.preferred_emotions.add()
        item.name = name
        item.value = value


def clear_preferred_emotion(settings: A2FSceneSettings) -> None:
    """Unset the preferred emotion snapshot."""

    settings.preferred_emotions.clear()


def emotion_settings(settings: A2FSceneSettings) -> dict[str, object]:
    """Return the exact manual/automatic emotion mixer settings."""

    manual_emotions = _manual_emotion_values(settings)
    preferred_emotions = _preferred_emotion_values(settings)
    if preferred_emotions and set(preferred_emotions) != set(manual_emotions):
        raise ValueError(
            "preferred emotion does not match the loaded model emotion channels"
        )
    return {
        "auto_audio2emotion": bool(settings.auto_audio2emotion),
        "manual_emotions": manual_emotions,
        "audio2emotion": {
            "emotion_strength": float(settings.a2e_emotion_strength),
            "emotion_contrast": float(settings.a2e_emotion_contrast),
            "max_emotions": int(settings.a2e_max_emotions),
            "live_blend_coef": float(settings.a2e_live_blend_coef),
            "transition_smoothing": float(settings.a2e_transition_smoothing),
            "preferred_emotion": preferred_emotions or None,
            "preferred_emotion_strength": float(
                settings.a2e_preferred_emotion_strength
            ),
        },
    }


def apply_model_schema(
    settings: A2FSceneSettings,
    model_schema: object,
    model_signature: tuple[str, str],
) -> tuple[str, ...]:
    """Validate and materialize one self-describing worker model schema."""

    if not isinstance(model_schema, dict) or set(model_schema) != _MODEL_SCHEMA_FIELDS:
        raise ValueError("worker returned a noncanonical model_schema object")
    try:
        channels = validate_output_channels(model_schema["channels"])
    except ShapeKeyStreamError as exc:
        raise ValueError(f"worker returned invalid output channels: {exc}") from exc
    emotions = _validated_emotion_descriptors(model_schema["emotion_channels"])
    signature = _schema_signature(model_signature, model_schema)

    same_schema = settings.model_schema_signature == signature
    preserved_manual: dict[str, float] = {}
    expected_emotion_names = {name for name, _default in emotions}
    if same_schema:
        preserved_manual = _manual_emotion_values(settings)
        if set(preserved_manual) != expected_emotion_names:
            raise ValueError(
                "saved manual emotions do not match the exact loaded model schema"
            )
        preserved_preferred = _preferred_emotion_values(settings)
        if (
            preserved_preferred
            and set(preserved_preferred) != expected_emotion_names
        ):
            raise ValueError(
                "saved preferred emotion does not match the exact loaded model schema"
            )

    settings.manual_emotions.clear()
    if not same_schema:
        settings.preferred_emotions.clear()
    for name, default in emotions:
        item = settings.manual_emotions.add()
        item.name = name
        item.value = preserved_manual[name] if same_schema else default

    settings.model_schema_signature = signature
    return channels


CLASSES = (
    A2FTargetMeshItem,
    A2FEmotionValueItem,
    A2FSceneSettings,
)
