# Worker protocol `audio2face/4`

## Transport and envelopes

Blender starts one local child process and exchanges UTF-8 JSON Lines over its
stdin and stdout. The worker is silent until `hello {}`. stdout is
protocol-only and diagnostics use stderr. No socket or separately hosted
service is involved.

Each record is exactly one JSON object followed by LF. The JSON payload before
that LF is limited to 1 MiB. Duplicate keys, non-finite JSON numbers, malformed
UTF-8, blank records, carriage returns, and multiple records on one line are
rejected.

A request contains exactly:

```json
{"protocol":"audio2face/4","type":"request","id":"1","method":"hello","params":{}}
```

`id` is a non-empty string of at most 128 characters. A successful response
repeats it and contains an object `result`:

```json
{"protocol":"audio2face/4","type":"response","id":"1","result":{}}
```

A request error contains exact `code`, `message`, and `details` fields. `id` is
included only when it could be recovered safely:

```json
{"protocol":"audio2face/4","type":"error","id":"1","error":{"code":"invalid_params","message":"invalid request","details":{}}}
```

An asynchronous event contains exact `event`, `operation_id`, and object `data`
fields in addition to `protocol` and `type`:

```json
{"protocol":"audio2face/4","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

The only request methods are `hello`, `load_model`, `stream_start`,
`stream_chunk`, `stream_end`, `cancel`, and `shutdown`. The only events are
`stream_frame`, `stream_ended`, and `error`. The worker accepts one active
stream. A request response is emitted before any event unlocked by that
request.

Selected WAV playback and external PCM use this same stream contract. Play and
first-chunk auto-start are Blender controller behavior, not additional worker
methods. There is no `generate` method or result-file protocol.

## `hello`

Parameters are exactly `{}`. The result is exactly:

```json
{"worker_profile":"nvidia-a2f3-a2e3-gpu-arkit52/4","worker_version":"0.1.0"}
```

Blender requires that exact profile and a non-empty worker version. `hello`
must complete before model or inference methods. It does not allocate CUDA or
model resources.

## `load_model`

Parameters contain exactly:

```json
{
  "audio2face_model_path": "/absolute/user-selected/audio2face/model.json",
  "audio2emotion_model_path": "/absolute/user-selected/audio2emotion/model.json"
}
```

Blender stores two repository roots and derives exactly `<root>/model.json`
from each. Setup has already validated the repositories and their optimized
`network.trt` engines. The worker selects the default Audio2Face identity at
SDK index `0`; identity is not a protocol input or schema field.

Loading creates the device-0 Audio2Face diffusion/geometry executor, the
model-owned GPU blendshape solver, the Audio2Emotion classifier executor, and
shared accumulators. The selected default v3 model's solver data contains its
internal 24,002-vertex neutral basis and 52 pose bases. That data is used only
inside the worker to translate raw model geometry into scalar weights; Blender
target geometry is never a protocol input. Loading allocates resources but
does not execute audio inference.

The response contains exactly `sample_rate` and `model_schema`:

```json
{
  "sample_rate": 16000,
  "model_schema": {
    "channels": ["<52 exact model-provided names in model order>"],
    "emotion_channels": [{"name": "<model emotion>", "default": 0.0}]
  }
}
```

`channels` contains 52 unique non-empty strings in the model's order.
`emotion_channels` is an ordered array of exact `{name, default}` objects with
unique names and finite defaults in `[0.0, 1.0]`. Internal graph nodes, tensors,
geometry, identities, and parameter structures are outside the schema.

## Settings document

Every `stream_start` request includes a `settings` object with exactly:

```json
{
  "auto_audio2emotion": false,
  "manual_emotions": {
    "<every model_schema emotion name>": 0.0
  },
  "audio2emotion": {
    "emotion_strength": 0.6,
    "emotion_contrast": 1.0,
    "max_emotions": 6,
    "live_blend_coef": 0.7,
    "transition_smoothing": 0.5,
    "preferred_emotion": null,
    "preferred_emotion_strength": 0.5
  }
}
```

`manual_emotions` contains every advertised emotion name exactly once and no
other key. Values are finite in `[0.0, 1.0]`. The `audio2emotion` object has
exactly the seven keys shown:

- `emotion_strength`: finite float in `[0.0, 1.0]`;
- `emotion_contrast`: finite float in `[0.1, 3.0]`;
- `max_emotions`: integer from `1` through the classifier's emotion count;
- `live_blend_coef`: finite float in `[0.0, 1.0]`;
- `transition_smoothing`: finite seconds in `[0.1, 1.0]`;
- `preferred_emotion`: `null` or an exact complete emotion snapshot; and
- `preferred_emotion_strength`: finite float in `[0.0, 1.0]`.

Partial documents and unknown keys are rejected. With
`auto_audio2emotion=false`, the manual snapshot is the direct constant emotion
driver. With it true, Audio2Emotion analyzes the same stream and NVIDIA's
post-processor applies the nested controls. A non-null preferred snapshot is
mixed as `p * preferred + (1 - p) * generated`, followed by overall emotion
strength. The settings remain frozen until the stream ends or is canceled.

## `stream_start`

Parameters contain exactly:

```json
{
  "operation_id": "stream-1",
  "sample_rate": 16000,
  "settings": {
    "auto_audio2emotion": true,
    "manual_emotions": {"<model emotion>": 0.0},
    "audio2emotion": {
      "emotion_strength": 0.6,
      "emotion_contrast": 1.0,
      "max_emotions": 6,
      "live_blend_coef": 0.7,
      "transition_smoothing": 0.5,
      "preferred_emotion": null,
      "preferred_emotion_strength": 0.5
    }
  }
}
```

`operation_id` is non-empty and at most 128 characters. `sample_rate` must
equal the rate returned by `load_model`; Blender's source adapters own WAV
conversion and external integrations own their resampling. The worker freezes
settings, resets incremental executors and accumulators, then returns exactly:

```json
{"sample_rate":16000,"prebuffer_samples":60000}
```

`prebuffer_samples` is a non-negative integer at the model rate. It is the
Audio2Face input lead required before a coefficient frame becomes available.
When automatic emotion is active, it is the greater of the Audio2Face lead and
Audio2Emotion readiness window.

## `stream_chunk`

Parameters contain exactly:

```json
{"operation_id":"stream-1","audio_f32le_base64":"AAAAAA=="}
```

The base64 text must be canonical and decode to a non-empty mono block of
little-endian IEEE-754 float32 samples. Every sample is finite. One chunk
covers at most one model-rate second. The ID must name the active stream, no
chunks are accepted after `stream_end` is queued, and the worker bounds queued
PCM to four seconds.

The worker replies `{}` before publishing any frame unlocked by that chunk. A
frame event has exact data fields `timestamp_sample` and `weights`:

```json
{
  "protocol": "audio2face/4",
  "type": "event",
  "event": "stream_frame",
  "operation_id": "stream-1",
  "data": {"timestamp_sample": 0, "weights": [0.0]}
}
```

The abbreviated `weights` array contains exactly 52 finite values in
`[0.0, 1.0]`, ordered by `model_schema.channels`. `timestamp_sample` is a
strictly increasing signed 64-bit position at the model sample rate. Events do
not repeat channel names.

The worker obtains these values by running the raw Audio2Face geometry through
the default v3 model's internal GPU blendshape basis and resolving eye
rotations into model-named eye-look slots. Raw geometry and solver bases are
never serialized.

## `stream_end`

Parameters are exactly `{"operation_id":"stream-1"}`. The worker replies `{}`,
closes input, drains padded tail frames, waits for scheduled GPU work, and then
emits:

```json
{"protocol":"audio2face/4","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

Every final `stream_frame` precedes `stream_ended`. Both models remain loaded,
ready for another stream. No animation file is created.

## `cancel`

Parameters are exactly `{"operation_id":"<active-id>"}`. A matching active
stream receives an immediate `{}` response. Queued input and execution stop
without draining, followed by `stream_ended {}`. An unknown, inactive, or
already terminal ID returns `operation_not_found`.

Blender uses cancellation internally for seek, rewind, loop restart, playback
cleanup, and failure recovery. It is not a separate result-management path.

## `shutdown`

Parameters and result are both `{}`. Shutdown stops any active stream, joins
its thread, responds, and exits the protocol loop. Backend destruction
synchronizes CUDA and releases both executors, model metadata, accumulators,
and CUDA resources. Blender applies bounded graceful, terminate, and kill
deadlines.

## Terminal stream errors

An asynchronous inference failure uses an `error` event with exact `code` and
`message` data:

```json
{"protocol":"audio2face/4","type":"event","event":"error","operation_id":"stream-1","data":{"code":"inference_failed","message":"operation failed"}}
```

Request validation failures use the request error envelope. Worker error codes
used by the implementation include:

- `invalid_json`, `invalid_request`, `request_too_large`,
  `protocol_mismatch`, `method_not_found`, `invalid_params`, `invalid_state`,
  and `busy`;
- `model_invalid` and `model_not_loaded`;
- `operation_not_found`, `sample_rate_mismatch`, `stream_backpressure`, and
  `inference_failed`; and
- `sdk_error`, `gpu_error`, and `internal_error`.
