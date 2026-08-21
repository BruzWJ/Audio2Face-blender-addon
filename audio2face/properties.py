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


STATUS_ITEMS = (
    ("IDLE", "Idle", "No worker operation is active"),
    ("INSTALLING_RUNTIME", "Installing Runtime", "Downloading and preparing the managed GPU runtime"),
    ("STARTING", "Starting", "Starting and handshaking with the worker"),
    ("LOADING_MODEL", "Loading Models", "Loading Audio2Face and Audio2Emotion"),
    ("MODEL_READY", "Model Ready", "Model is ready for generation"),
    ("GENERATING", "Generating", "Generating timestamped ARKit-52 values"),
    ("CANCELLING", "Cancelling", "Cancellation has been requested"),
    ("COMPLETED", "Completed", "A generated ARKit-52 stream is available"),
    ("STREAM_STARTING", "Starting Stream", "Preparing incremental PCM inference"),
    ("STREAMING", "Streaming", "Incremental PCM is driving ARKit-52 values"),
    ("STREAM_ENDING", "Ending Stream", "Draining the stream's final model frames"),
    ("STOPPING", "Stopping", "Worker is shutting down"),
    ("ERROR", "Error", "The last operation failed"),
)


def _mesh_object(_self: Any, obj: bpy.types.Object) -> bool:
    return obj.type == "MESH"


class A2FTargetMeshItem(bpy.types.PropertyGroup):
    """One mesh subscriber to the canonical ARKit-52 frame bus."""

    enabled: BoolProperty(name="Enabled", default=True)
    object: PointerProperty(
        name="Face Mesh",
        description="Mesh whose exact-name ARKit shape keys receive streamed values",
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
        description="Worker/model identity index used consistently for generation",
        default=0,
        min=0,
    )

    auto_audio2emotion: BoolProperty(
        name="Auto Audio2Emotion",
        description=(
            "Override the manual emotion driver with emotion values inferred "
            "from the input audio"
        ),
        default=False,
    )
    manual_emotions: CollectionProperty(type=A2FEmotionValueItem)
    emotion_strength: FloatProperty(
        name="Strength",
        description="Overall strength of automatically inferred emotions",
        default=0.6,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    emotion_contrast: FloatProperty(
        name="Contrast",
        description="Contrast applied to automatically inferred emotion probabilities",
        default=1.0,
        min=0.1,
        max=3.0,
    )
    emotion_smoothing: FloatProperty(
        name="Smoothing",
        description="Temporal smoothing applied to automatically inferred emotions",
        default=0.7,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    emotion_transition_time: FloatProperty(
        name="Transition Time",
        description="Seconds used to transition between inferred emotion states",
        default=0.5,
        min=0.1,
        max=1.0,
        unit="TIME",
    )
    max_emotions: IntProperty(
        name="Max Emotions",
        description="Maximum number of simultaneous inferred emotions",
        default=6,
        min=1,
        max=6,
    )

    target_meshes: CollectionProperty(type=A2FTargetMeshItem)
    target_mesh_index: IntProperty(default=0, min=0)

    preview_loop: BoolProperty(
        name="Loop",
        description="Loop selected audio and its canonical ARKit-52 frame stream",
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

    # Managed-model defaults are applied on the main thread after worker startup.
    input_strength: FloatProperty(name="Input Strength", default=1.0, min=0.0, soft_max=2.0)
    lower_face_smoothing: FloatProperty(name="Lower Face Smoothing", default=0.0, min=0.0, soft_max=1.0)
    upper_face_smoothing: FloatProperty(name="Upper Face Smoothing", default=0.0, min=0.0, soft_max=1.0)
    lower_face_strength: FloatProperty(name="Lower Face Strength", default=1.0, min=0.0, soft_max=2.0)
    upper_face_strength: FloatProperty(name="Upper Face Strength", default=1.0, min=0.0, soft_max=2.0)
    face_mask_level: FloatProperty(name="Face Mask Level", default=0.5, min=0.0, max=1.0)
    face_mask_softness: FloatProperty(name="Face Mask Softness", default=0.1, min=0.0, max=1.0)
    skin_strength: FloatProperty(name="Skin Strength", default=1.0, min=0.0, soft_max=2.0)
    blink_strength: FloatProperty(name="Blink Strength", default=1.0, min=0.0, soft_max=2.0)
    blink_offset: FloatProperty(name="Blink Offset", default=0.0, soft_min=-1.0, soft_max=1.0)
    eyelid_open_offset: FloatProperty(name="Eyelid Open Offset", default=0.0, soft_min=-1.0, soft_max=1.0)
    lip_open_offset: FloatProperty(name="Lip Open Offset", default=0.0, soft_min=-1.0, soft_max=1.0)

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
    runtime_install_progress: FloatProperty(
        name="Install Progress",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        options={"HIDDEN", "SKIP_SAVE"},
    )
    show_tuning: BoolProperty(name="Model Tuning", default=False)


PARAMETER_GROUPS = (
    (
        "skin",
        (
            "lower_face_smoothing",
            "upper_face_smoothing",
            "lower_face_strength",
            "upper_face_strength",
            "face_mask_level",
            "face_mask_softness",
            "skin_strength",
            "blink_strength",
            "blink_offset",
            "eyelid_open_offset",
            "lip_open_offset",
        ),
    ),
)

EMOTION_AUTO_PROPERTIES = (
    ("strength", "emotion_strength"),
    ("contrast", "emotion_contrast"),
    ("smoothing", "emotion_smoothing"),
    ("transition_time", "emotion_transition_time"),
    ("max_emotions", "max_emotions"),
)


def _finite_float_in_range(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{label} must be in [{minimum:g}, {maximum:g}]")
    return result


def _manual_emotion_values(settings: A2FSceneSettings) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in settings.manual_emotions:
        name = item.name
        if not isinstance(name, str) or not name:
            raise ValueError("manual emotion names must be non-empty strings")
        if name in values:
            raise ValueError(f"manual emotion name {name!r} is duplicated")
        values[name] = _finite_float_in_range(
            item.value,
            label=f"manual emotion {name!r}",
            minimum=0.0,
            maximum=1.0,
        )
    return values


def tuning_parameters(settings: A2FSceneSettings) -> dict[str, object]:
    """Return the worker's nested get-mutate-set parameter shape."""

    return {
        "input_strength": float(settings.input_strength),
        **{
            group: {name: float(getattr(settings, name)) for name in names}
            for group, names in PARAMETER_GROUPS
        },
        "emotion": {
            "auto_audio2emotion": bool(settings.auto_audio2emotion),
            "manual_values": _manual_emotion_values(settings),
            "auto": {
                worker_name: (
                    int(getattr(settings, property_name))
                    if worker_name == "max_emotions"
                    else float(getattr(settings, property_name))
                )
                for worker_name, property_name in EMOTION_AUTO_PROPERTIES
            },
        },
    }


def apply_model_defaults(
    settings: A2FSceneSettings,
    defaults: object,
    emotion_names: object,
) -> None:
    """Apply the worker's exact finite managed-model parameter document."""

    expected_groups = {"input_strength", "emotion"} | {
        group for group, _names in PARAMETER_GROUPS
    }
    if not isinstance(defaults, dict) or set(defaults) != expected_groups:
        raise ValueError("worker returned a noncanonical parameter_defaults object")

    if (
        not isinstance(emotion_names, list)
        or any(not isinstance(name, str) or not name for name in emotion_names)
        or len(set(emotion_names)) != len(emotion_names)
    ):
        raise ValueError("worker returned invalid or duplicate emotion_names")

    current_emotion_names = tuple(item.name for item in settings.manual_emotions)
    schema_matches = current_emotion_names == tuple(emotion_names)

    values: dict[str, object] = {"input_strength": defaults["input_strength"]}
    for group, names in PARAMETER_GROUPS:
        group_values = defaults[group]
        if not isinstance(group_values, dict) or set(group_values) != set(names):
            raise ValueError(f"worker returned noncanonical {group} parameter defaults")
        values.update(group_values)

    emotion = defaults["emotion"]
    if not isinstance(emotion, dict) or set(emotion) != {"manual_values", "auto"}:
        raise ValueError("worker returned noncanonical emotion parameter defaults")
    manual_values = emotion["manual_values"]
    if not isinstance(manual_values, dict) or set(manual_values) != set(emotion_names):
        raise ValueError("worker emotion defaults do not match emotion_names")
    validated_manual = {
        name: _finite_float_in_range(
            manual_values[name],
            label=f"model default emotion {name!r}",
            minimum=0.0,
            maximum=1.0,
        )
        for name in emotion_names
    }

    auto = emotion["auto"]
    expected_auto = {name for name, _property_name in EMOTION_AUTO_PROPERTIES}
    if not isinstance(auto, dict) or set(auto) != expected_auto:
        raise ValueError("worker returned noncanonical Audio2Emotion defaults")
    validated_auto: dict[str, float | int] = {
        "strength": _finite_float_in_range(
            auto["strength"],
            label="model default Audio2Emotion strength",
            minimum=0.0,
            maximum=1.0,
        ),
        "contrast": _finite_float_in_range(
            auto["contrast"],
            label="model default Audio2Emotion contrast",
            minimum=0.1,
            maximum=3.0,
        ),
        "smoothing": _finite_float_in_range(
            auto["smoothing"],
            label="model default Audio2Emotion smoothing",
            minimum=0.0,
            maximum=1.0,
        ),
        "transition_time": _finite_float_in_range(
            auto["transition_time"],
            label="model default Audio2Emotion transition time",
            minimum=0.1,
            maximum=1.0,
        ),
    }
    max_emotions = auto["max_emotions"]
    if (
        isinstance(max_emotions, bool)
        or not isinstance(max_emotions, int)
        or not 1 <= max_emotions <= 6
    ):
        raise ValueError("model default Audio2Emotion max_emotions must be in [1, 6]")
    validated_auto["max_emotions"] = max_emotions

    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"worker returned invalid model default {name!r}")
        if not schema_matches:
            try:
                setattr(settings, name, float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model default {name!r} is outside Blender property limits"
                ) from exc

    if not schema_matches:
        for worker_name, property_name in EMOTION_AUTO_PROPERTIES:
            try:
                setattr(settings, property_name, validated_auto[worker_name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Audio2Emotion default {worker_name!r} is outside Blender property limits"
                ) from exc

    preserved_manual = _manual_emotion_values(settings)
    settings.manual_emotions.clear()
    for name in emotion_names:
        item = settings.manual_emotions.add()
        item.name = name
        item.value = preserved_manual.get(name, validated_manual[name])


CLASSES = (A2FTargetMeshItem, A2FEmotionValueItem, A2FSceneSettings)


__all__ = [
    "A2FSceneSettings",
    "A2FEmotionValueItem",
    "A2FTargetMeshItem",
    "CLASSES",
    "EMOTION_AUTO_PROPERTIES",
    "PARAMETER_GROUPS",
    "STATUS_ITEMS",
    "apply_model_defaults",
    "tuning_parameters",
]
