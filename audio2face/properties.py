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


AUDIO2FACE_SETTING_GROUPS = (
    (
        "Input",
        (
            ("input_strength", True),
            ("eye_saccade_seed", False),
        ),
    ),
    (
        "Face",
        (
            ("skin_strength", True),
            ("upper_face_strength", True),
            ("lower_face_strength", True),
            ("eyelid_open_offset", True),
            ("blink_strength", True),
            ("lip_open_offset", True),
            ("upper_face_smoothing", True),
            ("lower_face_smoothing", True),
            ("face_mask_level", True),
            ("face_mask_softness", True),
        ),
    ),
    (
        "Eyes",
        (
            ("eyeballs_strength", True),
            ("saccade_strength", True),
            ("right_eye_rot_x_offset", True),
            ("right_eye_rot_y_offset", True),
            ("left_eye_rot_x_offset", True),
            ("left_eye_rot_y_offset", True),
        ),
    ),
)
AUDIO2FACE_SETTING_FIELDS = tuple(
    name
    for _group, fields in AUDIO2FACE_SETTING_GROUPS
    for name, _slider in fields
)

_AUDIO2FACE_FLOAT_RANGES = {
    "input_strength": (0.0, 3.0),
    "lower_face_smoothing": (0.0, 0.1),
    "upper_face_smoothing": (0.0, 0.1),
    "lower_face_strength": (0.0, 2.0),
    "upper_face_strength": (0.0, 2.0),
    "face_mask_level": (0.0, 1.0),
    "face_mask_softness": (0.001, 0.5),
    "skin_strength": (0.0, 2.0),
    "blink_strength": (0.0, 2.0),
    "eyelid_open_offset": (-1.0, 1.0),
    "lip_open_offset": (-0.2, 0.2),
    "eyeballs_strength": (0.0, 2.0),
    "saccade_strength": (0.0, 2.0),
    "right_eye_rot_x_offset": (-10.0, 10.0),
    "right_eye_rot_y_offset": (-10.0, 10.0),
    "left_eye_rot_x_offset": (-10.0, 10.0),
    "left_eye_rot_y_offset": (-10.0, 10.0),
}


def _inference_setting_updated(
    _settings: bpy.types.PropertyGroup,
    context: bpy.types.Context,
) -> None:
    """Refresh inference from one shared RNA update callback."""

    scene = getattr(context, "scene", None)
    if scene is None or not hasattr(scene, "audio2face"):
        return
    from .runtime import get_controller

    get_controller().refresh_inference_settings(scene)


class A2FTargetMeshItem(bpy.types.PropertyGroup):
    """One mesh driven by the model-provided frame stream."""

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
        update=_inference_setting_updated,
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
    input_strength: FloatProperty(
        name="Input Strength",
        description="Scale the audio signal supplied to Audio2Face",
        default=1.0,
        min=0.0,
        max=3.0,
        update=_inference_setting_updated,
    )
    lower_face_smoothing: FloatProperty(
        name="Lower Face Smoothing",
        description="Apply temporal smoothing to lower-face motion",
        default=0.006,
        min=0.0,
        max=0.1,
        update=_inference_setting_updated,
    )
    upper_face_smoothing: FloatProperty(
        name="Upper Face Smoothing",
        description="Apply temporal smoothing to upper-face motion",
        default=0.001,
        min=0.0,
        max=0.1,
        update=_inference_setting_updated,
    )
    lower_face_strength: FloatProperty(
        name="Lower Face Strength",
        description="Control the range of motion in the lower face",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_inference_setting_updated,
    )
    upper_face_strength: FloatProperty(
        name="Upper Face Strength",
        description="Control the range of motion in the upper face",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_inference_setting_updated,
    )
    face_mask_level: FloatProperty(
        name="Face Mask Level",
        description="Set the boundary between upper- and lower-face regions",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_inference_setting_updated,
    )
    face_mask_softness: FloatProperty(
        name="Face Mask Softness",
        description="Blend upper- and lower-face motion across their boundary",
        default=0.0085,
        min=0.001,
        max=0.5,
        update=_inference_setting_updated,
    )
    skin_strength: FloatProperty(
        name="Skin Strength",
        description="Control the overall range of skin motion",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_inference_setting_updated,
    )
    blink_strength: FloatProperty(
        name="Blink Strength",
        description="Control the range of eyelid blink motion",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_inference_setting_updated,
    )
    eyelid_open_offset: FloatProperty(
        name="Eyelid Offset",
        description="Adjust the resting eyelid open-close pose",
        default=0.0,
        min=-1.0,
        max=1.0,
        update=_inference_setting_updated,
    )
    lip_open_offset: FloatProperty(
        name="Lip Open Offset",
        description="Adjust the resting lip close-open pose",
        default=0.0,
        min=-0.2,
        max=0.2,
        update=_inference_setting_updated,
    )
    eyeballs_strength: FloatProperty(
        name="Offset Strength",
        description="Control the range of eye offset motion per emotion",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_inference_setting_updated,
    )
    saccade_strength: FloatProperty(
        name="Saccade Strength",
        description="Control the range of procedural eye saccades",
        default=0.6,
        min=0.0,
        max=2.0,
        update=_inference_setting_updated,
    )
    right_eye_rot_x_offset: FloatProperty(
        name="Right Eye Rotate X",
        description="Offset the right eye's vertical orientation in degrees",
        default=0.0,
        min=-10.0,
        max=10.0,
        update=_inference_setting_updated,
    )
    right_eye_rot_y_offset: FloatProperty(
        name="Right Eye Rotate Y",
        description="Offset the right eye's horizontal orientation in degrees",
        default=0.0,
        min=-10.0,
        max=10.0,
        update=_inference_setting_updated,
    )
    left_eye_rot_x_offset: FloatProperty(
        name="Left Eye Rotate X",
        description="Offset the left eye's vertical orientation in degrees",
        default=0.0,
        min=-10.0,
        max=10.0,
        update=_inference_setting_updated,
    )
    left_eye_rot_y_offset: FloatProperty(
        name="Left Eye Rotate Y",
        description="Offset the left eye's horizontal orientation in degrees",
        default=0.0,
        min=-10.0,
        max=10.0,
        update=_inference_setting_updated,
    )
    eye_saccade_seed: IntProperty(
        name="Eye Saccade Data",
        description="Control which deterministic eye dart motion is applied",
        default=0,
        min=0,
        max=4999,
        update=_inference_setting_updated,
    )
    auto_audio2emotion: BoolProperty(
        name="Auto Audio2Emotion",
        description="Infer emotion values from the input audio for this operation",
        default=False,
        update=_inference_setting_updated,
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
        update=_inference_setting_updated,
    )
    a2e_emotion_contrast: FloatProperty(
        name="Emotion Contrast",
        description="Increase or reduce the spread between generated emotions",
        default=1.0,
        min=0.1,
        max=3.0,
        update=_inference_setting_updated,
    )
    a2e_max_emotions: IntProperty(
        name="Max Emotions",
        description="Maximum generated emotions retained at each inference frame",
        default=6,
        min=1,
        max=6,
        update=_inference_setting_updated,
    )
    a2e_live_blend_coef: FloatProperty(
        name="Smoothing",
        description="Influence of the preceding generated emotion on the next frame",
        default=0.7,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_inference_setting_updated,
    )
    a2e_transition_smoothing: FloatProperty(
        name="Transition Time",
        description="Time used to transition between automatic emotion states",
        default=0.5,
        min=0.1,
        max=1.0,
        unit="TIME",
        update=_inference_setting_updated,
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
        update=_inference_setting_updated,
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
    stream_time: FloatProperty(
        name="Stream Time", default=0.0, min=0.0, unit="TIME", options={"SKIP_SAVE"}
    )

    status: EnumProperty(
        name="Status", items=STATUS_ITEMS, default="IDLE", options={"SKIP_SAVE"}
    )
    status_message: StringProperty(
        name="Message", default="Worker is stopped", options={"SKIP_SAVE"}
    )


_MODEL_SCHEMA_FIELDS = {
    "channels",
    "emotion_channels",
    "audio2face_defaults",
}
_EMOTION_DESCRIPTOR_FIELDS = {"name", "default"}


def _finite_float_in_range(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    result = value
    if result < minimum or result > maximum:
        raise ValueError(f"{label} must be in [{minimum:g}, {maximum:g}]")
    return result


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


def _validated_audio2face_values(
    values: object,
    *,
    label: str,
) -> dict[str, float | int]:
    if not isinstance(values, dict) or set(values) != set(
        AUDIO2FACE_SETTING_FIELDS
    ):
        raise ValueError(f"{label} has unexpected or missing fields")
    result: dict[str, float | int] = {}
    for name in AUDIO2FACE_SETTING_FIELDS:
        value = values[name]
        if name == "eye_saccade_seed":
            if type(value) is not int or value < 0 or value > 4999:
                raise ValueError(
                    f"{label}.{name} must be an integer in [0, 4999]"
                )
            result[name] = value
            continue
        minimum, maximum = _AUDIO2FACE_FLOAT_RANGES[name]
        result[name] = _finite_float_in_range(
            value,
            label=f"{label}.{name}",
            minimum=minimum,
            maximum=maximum,
        )
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


def inference_settings(settings: A2FSceneSettings) -> dict[str, object]:
    """Return the exact Audio2Face and emotion settings for one stream."""

    manual_emotions = _manual_emotion_values(settings)
    preferred_emotions = _preferred_emotion_values(settings)
    if preferred_emotions and set(preferred_emotions) != set(manual_emotions):
        raise ValueError(
            "preferred emotion does not match the loaded model emotion channels"
        )
    return {
        "audio2face": _validated_audio2face_values(
            {
                name: getattr(settings, name)
                for name in AUDIO2FACE_SETTING_FIELDS
            },
            label="Audio2Face settings",
        ),
        "auto_audio2emotion": settings.auto_audio2emotion,
        "manual_emotions": manual_emotions,
        "audio2emotion": {
            "emotion_strength": settings.a2e_emotion_strength,
            "emotion_contrast": settings.a2e_emotion_contrast,
            "max_emotions": settings.a2e_max_emotions,
            "live_blend_coef": settings.a2e_live_blend_coef,
            "transition_smoothing": settings.a2e_transition_smoothing,
            "preferred_emotion": preferred_emotions or None,
            "preferred_emotion_strength": settings.a2e_preferred_emotion_strength,
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
    audio2face_defaults = _validated_audio2face_values(
        model_schema["audio2face_defaults"],
        label="model_schema.audio2face_defaults",
    )
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

    if not same_schema:
        for name, default in audio2face_defaults.items():
            setattr(settings, name, default)
        settings.preferred_emotions.clear()
    settings.manual_emotions.clear()
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
