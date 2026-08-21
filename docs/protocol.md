# Worker protocol `audio2face/3`

## Transport and envelopes

Blender starts one local child process and exchanges UTF-8 JSON Lines over its
stdin and stdout. The worker is silent until `hello {}`. stdout is
protocol-only and diagnostics use stderr. No socket or separately hosted
service is involved.

Each record is exactly one JSON object followed by LF. The JSON payload
excluding that ending is limited to 1 MiB. Duplicate keys, non-finite JSON
numbers, malformed UTF-8, blank records, and additional records on the same
line are rejected.

A request contains exactly:

```json
{"protocol":"audio2face/3","type":"request","id":"1","method":"hello","params":{}}
```

`id` is a non-empty string of at most 128 characters. A successful response
repeats it and contains an object `result`:

```json
{"protocol":"audio2face/3","type":"response","id":"1","result":{}}
```

A request error contains exact `code`, `message`, and `details` fields. `id` is
included only when it could be recovered safely:

```json
{"protocol":"audio2face/3","type":"error","id":"1","error":{"code":"invalid_params","message":"invalid request","details":{}}}
```

An asynchronous event contains exact `event`, `operation_id`, and object `data`
fields in addition to `protocol` and `type`:

```json
{"protocol":"audio2face/3","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

The request methods are `hello`, `load_model`, `generate`, `stream_start`,
`stream_chunk`, `stream_end`, `cancel`, and `shutdown`. The event names are
`progress`, `result`, `canceled`, `stream_frame`, `stream_ended`, and `error`.
The worker accepts one active generation or stream operation. Its immediate
response is always written before an event caused by that request.

## `hello`

Parameters are exactly `{}`. The result is exactly:

```json
{"worker_profile":"nvidia-a2f3-a2e3-gpu-arkit52/2","worker_version":"0.1.0"}
```

Blender requires that exact profile and a non-empty worker version. `hello`
must complete before model or inference methods. It does not allocate CUDA or
model resources.

## `load_model`

Parameters contain exactly:

```json
{
  "audio2face_model_path": "/absolute/user-selected/audio2face/model.json",
  "audio2emotion_model_path": "/absolute/user-selected/audio2emotion/model.json",
  "identity_index": 0
}
```

Blender persistently stores two user-selected repository roots, then derives
these protocol paths as exactly `<root>/model.json`; no other descriptor path
is considered. Before the worker starts, setup has validated each root's
non-empty `model.json`, `network.onnx`, `trt_info.json`, every file referenced
by the descriptor, and locally optimized `network.trt`; unresolved Git LFS
pointers are rejected. The model repositories remain external to the add-on's
add-on-owned storage. `identity_index` is a non-negative integer within the
Audio2Face identity range. Loading creates the device-0 Audio2Face
diffusion/blendshape executor, Audio2Emotion classifier executor, their shared
CUDA stream, and shared audio and emotion accumulators. The models must agree
on sample rate and emotion-vector width. Loading does not execute inference.

The response contains exactly `sample_rate` and `model_schema`:

```json
{
  "sample_rate": 16000,
  "model_schema": {
    "identities": ["<model identity>"],
    "channels": ["<52 exact model-provided names in model order>"],
    "parameters": {"/advertised/path": 0.0},
    "emotion_channels": [{"name": "<model emotion>", "default": 0.0}]
  }
}
```

Angle-bracket values in this document are explanatory metavariables. The
`channels` example abbreviates an array that contains exactly 52 strings.

`model_schema` has exactly these four fields:

- `identities` is a non-empty ordered array of non-empty model names. Its array
  position is the identity index.
- `channels` contains 52 unique non-empty strings in the model's order.
- `parameters` maps each unique opaque worker path to its finite numeric
  default. JSON integer and float types define the control type.
- `emotion_channels` is an ordered array of exact `{name, default}` objects.
  Names are unique model-provided strings and defaults are finite values in
  `[0.0, 1.0]`.

Output channels, identity names, emotion names, and their ordering come from
the loaded Audio2Face model. Numeric defaults are read through the SDK. The
worker's typed adapter defines the paths it can apply through the SDK's public
parameter structures. SDK 1.0 has no parameter reflection; graph nodes and
tensors are outside this contract.

## Settings document

Every `generate` and `stream_start` request includes a `settings` object with
exactly:

```json
{
  "auto_audio2emotion": false,
  "manual_emotions": {
    "<every model_schema emotion name>": 0.0
  },
  "parameters": {
    "/every/advertised/parameter/path": 0.0
  }
}
```

`manual_emotions` contains every advertised emotion name exactly once and no
other key. Values are finite in `[0.0, 1.0]`. `parameters` contains every
advertised parameter path exactly once and no other key. Each value retains
the advertised JSON numeric type and must be accepted by the applicable SDK
setter. Partial documents and unknown keys are rejected.

When `auto_audio2emotion` is false, the ordered manual values are accumulated
at timestamp zero and form a constant emotion driver. When it is true,
Audio2Emotion analyzes the operation's audio and its timestamped output fully
replaces the manual vector. Manual values remain present in the document but
are ignored. Selected WAV and Stream use this identical rule and parameter
shape.

## `generate`

Parameters contain exactly:

```json
{
  "operation_id": "operation-1",
  "audio_path": "/absolute/path/speech.wav",
  "result_path": "/absolute/audio2face-results/operation-1.a2f.json",
  "settings": {
    "auto_audio2emotion": false,
    "manual_emotions": {"<model emotion>": 0.0},
    "parameters": {"/advertised/path": 0.0}
  }
}
```

`operation_id` follows the 128-character ID bound. Both paths are absolute. Blender
supplies `result_path` inside its add-on-owned results directory. The WAV is limited
to 512 MiB and may contain 8-, 16-, 24-, or 32-bit integer PCM or float32. The
worker downmixes channels and linearly resamples to the loaded model rate. It
also bounds the decoded and resampled audio to 512 MiB of float samples.

The immediate response is `{}`. Progress events contain exact finite
`progress` in `[0.0, 1.0]` and a non-empty `stage`:

```json
{"protocol":"audio2face/3","type":"event","event":"progress","operation_id":"operation-1","data":{"progress":0.5,"stage":"generating"}}
```

After all frames are generated, the worker atomically commits the result and
emits an empty event:

```json
{"protocol":"audio2face/3","type":"event","event":"result","operation_id":"operation-1","data":{}}
```

Blender derives the add-on-owned path it submitted and requires the file to exist.
The worker never replaces an existing result or exposes a partial document.

## `stream_start`

Parameters contain exactly:

```json
{
  "operation_id": "stream-1",
  "sample_rate": 16000,
  "settings": {
    "auto_audio2emotion": true,
    "manual_emotions": {"<model emotion>": 0.0},
    "parameters": {"/advertised/path": 0.0}
  }
}
```

`sample_rate` must equal the rate returned by `load_model`; source adapters own
resampling. The worker freezes the complete settings, resets both incremental
executors and accumulators, and selects the manual or automatic emotion
driver. The response contains exactly:

```json
{"sample_rate":16000,"prebuffer_samples":60000}
```

`prebuffer_samples` is a non-negative integer at the returned sample rate. It
is the Audio2Face input lead required before a face frame can become ready. In
automatic emotion mode it is the greater of that lead and Audio2Emotion's
readiness window. Sources must use the returned value before starting
synchronized audible monitoring.

## `stream_chunk`

Parameters contain exactly:

```json
{"operation_id":"stream-1","audio_f32le_base64":"AAAAAA=="}
```

The base64 text must be canonical and decode to a non-empty mono block of
little-endian IEEE-754 float32 samples. Every sample is finite. A chunk covers
at most one model-rate second; Blender additionally caps its decoded size at
256 KiB. The ID must name the active stream. The worker replies `{}` before any
frames unlocked by that chunk.

A frame event has exact data fields `timestamp_sample` and `weights`:

```json
{
  "protocol": "audio2face/3",
  "type": "event",
  "event": "stream_frame",
  "operation_id": "stream-1",
  "data": {"timestamp_sample": 0, "weights": [0.0]}
}
```

The weights example abbreviates exactly 52 finite values in `[0.0, 1.0]`, in
the `model_schema.channels` order. `timestamp_sample` is a strictly
increasing signed 64-bit model-sample position. The event does not repeat the
channels. Audio samples no longer required by either inference
stage are dropped from the shared accumulator.

## `stream_end`

Parameters are exactly `{"operation_id":"stream-1"}`. The worker replies `{}`,
closes input, drains padded tail frames, waits for scheduled GPU work, and then
emits:

```json
{"protocol":"audio2face/3","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

Every final `stream_frame` precedes `stream_ended`. Both models remain loaded
and no result file is written.

## `cancel`

Parameters are exactly `{"operation_id":"<active-id>"}`. The ID may address either
operation type. A matching operation receives an immediate `{}` response.

For generation, execution stops, no partial result is committed, and the
terminal event is `canceled {}`. For a stream, queued input and execution stop
without draining, and the terminal event is `stream_ended {}`. An unknown or
inactive ID returns `operation_not_found`. An atomic result commit that completed
before cancellation remains a successful result.

## `shutdown`

Parameters and result are both `{}`. Shutdown stops the active operation,
joins its thread, responds, and exits the protocol loop. Backend destruction
synchronizes CUDA and releases both executors, both model metadata objects,
the shared accumulators, and the CUDA stream. Blender applies bounded graceful,
terminate, and kill deadlines.

## Result schema `a2f-animation/2`

The Selected WAV result has exactly six fields:

```json
{
  "schema": "a2f-animation/2",
  "operation_id": "operation-1",
  "sample_rate": 16000,
  "channels": ["<52 exact model-provided names in model order>"],
  "timestamps_samples": [0],
  "weights": [[0.0]]
}
```

The channel and weight arrays are abbreviated above. Validation requires:

- a non-empty `operation_id` of at most 128 characters;
- a positive uint32 `sample_rate`;
- exactly 52 unique non-empty channel strings;
- non-empty, strictly increasing signed 64-bit sample timestamps;
- one weight row per timestamp;
- exactly one finite `[0.0, 1.0]` value per channel in every row; and
- the serialized channel list and row order to remain identical to the loaded
  model's output description.

The worker preserves the model's skin channel order. It locates the eight
standard ARKit eye-look semantics in that list and resolves six SDK eye-rotation
components into those slots. No separate Python channel list, reorder table,
raw geometry, jaw transform, or eye-rotation payload is part of the result.

## Terminal operation errors

An asynchronous operation failure uses an `error` event with exact `code` and
`message` data:

```json
{"protocol":"audio2face/3","type":"event","event":"error","operation_id":"operation-1","data":{"code":"generation_failed","message":"operation failed"}}
```

Request validation failures use the request error envelope. Worker error codes
used by the implementation include:

- `invalid_json`, `invalid_request`, `request_too_large`,
  `protocol_mismatch`, `method_not_found`, `invalid_params`, `invalid_state`,
  and `busy`;
- `model_not_found`, `model_invalid`, `identity_invalid`, and
  `model_not_loaded`;
- `audio_open_failed`, `invalid_audio`,
  `unsupported_audio`, and `audio_too_large`;
- `operation_not_found`, `generation_failed`, `sample_rate_mismatch`, and
  `stream_backpressure`;
- `sdk_error`, `gpu_error`, and `internal_error`; and
- `invalid_result_path`, `result_exists`,
  `result_too_large`, `result_write_failed`, and `result_commit_failed`.
