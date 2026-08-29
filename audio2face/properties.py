"""Blender RNA state for Audio2Face playback and model-derived channels."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterator
from contextlib import contextmanager

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

from .shape_keys import (
    ShapeKeyStreamError,
    supports_shape_keys,
    validate_output_channels,
)


STATUS_ITEMS = (
    ("IDLE", "Idle", "No worker operation is active"),
    ("STARTING", "Starting", "Starting and handshaking with the worker"),
    ("LOADING_MODEL", "Loading Models", "Loading Audio2Face and Audio2Emotion"),
    ("MODEL_READY", "Model Ready", "Model is ready for audio-driven inference"),
    ("STREAM_STARTING", "Starting Stream", "Preparing incremental PCM inference"),
    ("STREAMING", "Streaming", "Incremental PCM is driving model channel values"),
    ("STREAM_ENDING", "Ending Stream", "Draining the stream's final model frames"),
    ("BAKE_UPLOADING", "Uploading Audio", "Uploading selected audio for baking"),
    ("BAKE_PREPARING", "Preparing Bake", "Preparing frame-based inference"),
    ("BAKING", "Baking", "Generating Blender-timeline Shape Key frames"),
    ("BAKE_ENDING", "Finishing Bake", "Writing native Shape Key animation"),
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
EMOTION_SETTING_FIELDS = (
    "auto_audio2emotion",
    "a2e_emotion_strength",
    "a2e_max_emotions",
    "a2e_emotion_contrast",
    "a2e_live_blend_coef",
    "a2e_transition_smoothing",
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

_internal_emotion_write_depth = 0


@contextmanager
def _internal_emotion_write() -> Iterator[None]:
    global _internal_emotion_write_depth
    _internal_emotion_write_depth += 1
    try:
        yield
    finally:
        _internal_emotion_write_depth -= 1


def _update_scene(context: bpy.types.Context) -> bpy.types.Scene | None:
    scene = getattr(context, "scene", None)
    if scene is None or not hasattr(scene, "audio2face"):
        return None
    return scene


def _inference_setting_updated(
    _settings: bpy.types.PropertyGroup,
    context: bpy.types.Context,
) -> None:
    """Refresh inference from one shared RNA update callback."""

    scene = _update_scene(context)
    if scene is None:
        return
    _refresh_inference(scene)


def _refresh_inference(scene: bpy.types.Scene) -> None:
    from .runtime import get_controller

    get_controller().refresh_inference_settings(scene)


def _preferred_emotion_updated(
    _emotion: bpy.types.PropertyGroup,
    context: bpy.types.Context,
) -> None:
    """Refresh inference immediately after an authored value changes."""

    if _internal_emotion_write_depth:
        return
    scene = _update_scene(context)
    if scene is None:
        return
    _refresh_inference(scene)


def _configure_selected_audio_timeline(
    settings: bpy.types.PropertyGroup,
    scene: bpy.types.Scene,
) -> int | None:
    from .selected_audio_timeline import (
        configure_selected_audio,
        remove_selected_audio_strips,
    )

    if not settings.audio_path:
        return None
    audio_path = bpy.path.abspath(settings.audio_path)
    try:
        frame_end = configure_selected_audio(
            scene,
            audio_path,
            first_frame=settings.audio_first_frame,
        )
    except (OSError, ValueError) as exc:
        remove_selected_audio_strips(scene)
        settings.status = "ERROR"
        settings.status_message = str(exc)
        return None
    return frame_end


def _audio_path_updated(
    settings: bpy.types.PropertyGroup,
    context: bpy.types.Context,
) -> None:
    """Replace the Selected WAV source and its native sound strip."""

    from .runtime import get_controller
    from .selected_audio_timeline import remove_selected_audio_strips

    scene = _update_scene(context)
    if scene is None:
        return
    get_controller().discard_selected_audio(scene)
    if not settings.audio_path:
        remove_selected_audio_strips(scene)
        return
    _configure_selected_audio_timeline(settings, scene)


def _audio_first_frame_updated(
    settings: bpy.types.PropertyGroup,
    context: bpy.types.Context,
) -> None:
    """Move the Selected WAV and cached samples to a new First Frame."""

    from .live_stream import get_live_stream_controller

    scene = _update_scene(context)
    if scene is None:
        return
    frame_end = _configure_selected_audio_timeline(settings, scene)
    if frame_end is None:
        return
    get_live_stream_controller().remap_timeline(
        scene,
        int(settings.audio_first_frame),
        frame_end,
    )


def _input_mode_updated(
    settings: bpy.types.PropertyGroup,
    context: bpy.types.Context,
) -> None:
    """Enter or leave Selected WAV ownership explicitly."""

    from .runtime import get_controller
    from .selected_audio_timeline import remove_selected_audio_strips

    scene = _update_scene(context)
    if scene is None:
        return
    get_controller().input_mode_changed(scene)
    if settings.input_mode == "SELECTED":
        _audio_first_frame_updated(settings, context)
        return
    remove_selected_audio_strips(scene)


def _target_object_poll(
    _item: bpy.types.PropertyGroup,
    target: bpy.types.Object,
) -> bool:
    return supports_shape_keys(target)


class A2FTargetObjectItem(bpy.types.PropertyGroup):
    """One Shape Key-capable object driven by model output."""

    object: PointerProperty(
        name="Target Object",
        description="Object whose Shape Keys receive matching model output values",
        type=bpy.types.Object,
        poll=_target_object_poll,
    )


class A2FPreferredEmotionItem(bpy.types.PropertyGroup):
    """One editable model-defined preferred-emotion channel."""

    name: StringProperty(
        name="Emotion",
        description="Model-defined Audio2Face emotion channel",
        default="",
        options={"HIDDEN"},
    )
    value: FloatProperty(
        name="Value",
        description="Authored preference for this model-defined emotion channel",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_preferred_emotion_updated,
    )


class A2FMixedEmotionItem(bpy.types.PropertyGroup):
    """One callback-free model-defined mixed-emotion display channel."""

    name: StringProperty(
        name="Emotion",
        options={"HIDDEN"},
    )
    value: FloatProperty(
        name="Value",
        description="Effective post-processed value returned by Audio2Face",
        default=0.0,
        soft_min=0.0,
        soft_max=2.0,
    )


class A2FSceneSettings(bpy.types.PropertyGroup):
    input_mode: EnumProperty(
        name="Input Mode",
        description="Play and bake a selected WAV or receive incremental PCM",
        items=(
            ("SELECTED", "Selected WAV", "Play or bake a selected WAV file"),
            ("STREAM", "Stream", "Drive Shape Keys from incremental mono float PCM"),
        ),
        default="SELECTED",
        update=_input_mode_updated,
    )
    audio_path: StringProperty(
        name="Speech WAV",
        description="WAV played or baked in Selected mode",
        subtype="FILE_PATH",
        update=_audio_path_updated,
    )
    audio_first_frame: IntProperty(
        name="First Frame",
        description="Blender timeline frame where the selected WAV begins",
        default=1,
        update=_audio_first_frame_updated,
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
    preferred_emotions: CollectionProperty(
        type=A2FPreferredEmotionItem,
        options={"HIDDEN"},
    )
    mixed_emotions: CollectionProperty(
        type=A2FMixedEmotionItem,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    a2e_emotion_strength: FloatProperty(
        name="Emotion Strength",
        description=(
            "Overall automatic-emotion multiplier applied after preferred "
            "emotion mixing; values above 1 amplify emotion without changing "
            "Skin Strength"
        ),
        default=0.6,
        min=0.0,
        max=2.0,
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
        name="Preferred Emotion Strength",
        description=(
            "Strength of nonzero preferred emotion values relative to generated emotion"
        ),
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_inference_setting_updated,
    )
    model_schema_signature: StringProperty(options={"HIDDEN"})

    target_objects: CollectionProperty(type=A2FTargetObjectItem)
    target_object_index: IntProperty(default=0, min=0)

    prediction_delay: FloatProperty(
        name="Prediction Delay",
        description="Adjust synchronization of mouth motion to audio in seconds",
        default=0.0,
        min=-1.0,
        max=1.0,
        unit="TIME",
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
    # Blender RNA and NVIDIA's SDK store these values as IEEE-754 binary32.
    storage_minimum = struct.unpack("=f", struct.pack("=f", minimum))[0]
    storage_maximum = struct.unpack("=f", struct.pack("=f", maximum))[0]
    if result == storage_minimum:
        result = minimum
    elif result == storage_maximum:
        result = maximum
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


def _replace_emotion_values(items: object, values: dict[str, float]) -> None:
    with _internal_emotion_write():
        items.clear()
        for name, value in values.items():
            item = items.add()
            item.name = name
            item.value = value


def apply_mixed_emotions(
    settings: A2FSceneSettings,
    emotion_channels: tuple[str, ...],
    values: tuple[float, ...],
) -> None:
    """Publish worker output into the read-only mixed-emotion controls."""

    if type(emotion_channels) is not tuple:
        raise ValueError("mixed emotion channels must be a tuple")
    current_channels = tuple(item.name for item in settings.mixed_emotions)
    if emotion_channels != current_channels:
        raise ValueError(
            "mixed emotion channels do not match the loaded model schema"
        )
    if type(values) is not tuple or len(values) != len(emotion_channels):
        raise ValueError(
            "mixed emotion values do not match the loaded model schema"
        )
    for name, value in zip(emotion_channels, values):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"mixed emotion {name!r} must be a finite float")
    for item, value in zip(settings.mixed_emotions, values):
        item.value = value


def reset_mixed_emotions(settings: A2FSceneSettings) -> None:
    """Clear transient worker output without touching authored Preferred values."""

    for item in settings.mixed_emotions:
        item.value = 0.0


def reset_model_tuning(settings: A2FSceneSettings) -> None:
    """Restore Model Tuning controls to their RNA defaults."""

    for name in AUDIO2FACE_SETTING_FIELDS:
        settings.property_unset(name)


def reset_emotion_settings(settings: A2FSceneSettings) -> None:
    """Restore generated-emotion controls without changing other sources."""

    for name in EMOTION_SETTING_FIELDS:
        settings.property_unset(name)


def inference_settings(settings: A2FSceneSettings) -> dict[str, object]:
    """Freeze generated emotion and the current value-driven Preferred source."""

    preferred_emotions = _emotion_values(
        settings.preferred_emotions,
        label="preferred emotion",
    )
    preferred_active = any(value != 0.0 for value in preferred_emotions.values())
    emotion_driver: dict[str, object] = {
        "emotion_strength": settings.a2e_emotion_strength,
        "generated": (
            {
                "emotion_contrast": settings.a2e_emotion_contrast,
                "max_emotions": settings.a2e_max_emotions,
                "live_blend_coef": settings.a2e_live_blend_coef,
                "transition_smoothing": settings.a2e_transition_smoothing,
            }
            if settings.auto_audio2emotion
            else None
        ),
        "preferred": (
            {
                "values": preferred_emotions,
                "strength": settings.a2e_preferred_emotion_strength,
            }
            if preferred_active
            else None
        ),
    }
    return {
        "audio2face": _validated_audio2face_values(
            {
                name: getattr(settings, name)
                for name in AUDIO2FACE_SETTING_FIELDS
            },
            label="Audio2Face settings",
        ),
        "emotion_driver": emotion_driver,
    }


def apply_model_schema(
    settings: A2FSceneSettings,
    model_schema: object,
    model_signature: tuple[str, str],
) -> None:
    """Validate and materialize one self-describing worker model schema."""

    if not isinstance(model_schema, dict) or set(model_schema) != _MODEL_SCHEMA_FIELDS:
        raise ValueError("worker returned a noncanonical model_schema object")
    try:
        validate_output_channels(model_schema["channels"])
    except ShapeKeyStreamError as exc:
        raise ValueError(f"worker returned invalid output channels: {exc}") from exc
    emotions = _validated_emotion_descriptors(model_schema["emotion_channels"])
    audio2face_defaults = _validated_audio2face_values(
        model_schema["audio2face_defaults"],
        label="model_schema.audio2face_defaults",
    )
    signature = _schema_signature(model_signature, model_schema)

    ordered_emotion_names = tuple(name for name, _default in emotions)
    default_emotions = {name: default for name, default in emotions}
    empty_mixed_emotions = dict.fromkeys(ordered_emotion_names, 0.0)
    same_schema = settings.model_schema_signature == signature
    preserve_preferred = same_schema and tuple(
        item.name for item in settings.preferred_emotions
    ) == ordered_emotion_names
    if not same_schema:
        for name, default in audio2face_defaults.items():
            setattr(settings, name, default)
    if not preserve_preferred:
        _replace_emotion_values(settings.preferred_emotions, default_emotions)
    _replace_emotion_values(settings.mixed_emotions, empty_mixed_emotions)

    settings.model_schema_signature = signature


CLASSES = (
    A2FTargetObjectItem,
    A2FPreferredEmotionItem,
    A2FMixedEmotionItem,
    A2FSceneSettings,
)
