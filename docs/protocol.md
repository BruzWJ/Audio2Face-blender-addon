# Worker protocol `audio2face/10`

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
{"protocol":"audio2face/10","type":"request","id":"1","method":"hello","params":{}}
```

`id` is a non-empty string of at most 128 characters. A successful response
repeats it and contains an object `result`:

```json
{"protocol":"audio2face/10","type":"response","id":"1","result":{}}
```

A request error contains exact `code`, `message`, and `details` fields. `id` is
included only when it could be recovered safely:

```json
{"protocol":"audio2face/10","type":"error","id":"1","error":{"code":"invalid_params","message":"invalid request","details":{}}}
```

An asynchronous event contains exact `event`, `operation_id`, and object `data`
fields in addition to `protocol` and `type`:

```json
{"protocol":"audio2face/10","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

The only request methods are `hello`, `load_model`, `stream_start`,
`stream_chunk`, `stream_settings`, `stream_end`, `bake_start`, `bake_chunk`,
`bake_prepare`, `bake_frame`, `bake_end`, `cancel`, and `shutdown`. The only
events are `stream_credit`, `stream_frame`, `stream_ended`, `bake_ended`, and
`error`.

The worker accepts one active inference operation: either a live stream or an
offline bake. Method-specific response and event ordering is defined below.

## `hello`

Parameters are exactly `{}`. The result is exactly:

```json
{"worker_profile":"nvidia-a2f3-a2e3-gpu-arkit52/10","worker_version":"0.1.0"}
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

Both values are absolute paths to validated top-level `model.json` files. The
worker selects Audio2Face identity index `0`; identity is not a protocol field.

Loading allocates the device-0 Audio2Face, blendshape-solver, and Audio2Emotion
resources but does not execute audio inference.

The response contains exactly `sample_rate` and `model_schema`. `sample_rate`
is a positive integer. `model_schema` contains exactly `channels`,
`emotion_channels`, and `audio2face_defaults`: `channels` is 52 unique non-empty
strings in model order; `emotion_channels` is an ordered array of exact
`{name, default}` objects with unique names and finite defaults in `[0.0, 1.0]`;
and `audio2face_defaults` is the exact 18-field `audio2face` object defined below,
populated from the loaded executor. Internal graph nodes, tensors, geometry,
identities, and parameter structures are outside the schema.

## Settings document

Every `stream_start`, `stream_settings`, and `bake_frame` settings field uses
this exact object:

```json
{
  "audio2face": {
    "input_strength": 1.0,
    "lower_face_smoothing": 0.006,
    "upper_face_smoothing": 0.001,
    "lower_face_strength": 1.0,
    "upper_face_strength": 1.0,
    "face_mask_level": 0.6,
    "face_mask_softness": 0.0085,
    "skin_strength": 1.0,
    "blink_strength": 1.0,
    "eyelid_open_offset": 0.0,
    "lip_open_offset": 0.0,
    "eyeballs_strength": 1.0,
    "saccade_strength": 0.6,
    "right_eye_rot_x_offset": 0.0,
    "right_eye_rot_y_offset": 0.0,
    "left_eye_rot_x_offset": 0.0,
    "left_eye_rot_y_offset": 0.0,
    "eye_saccade_seed": 0
  },
  "emotion_driver": {
    "emotion_strength": 0.6,
    "generated": null,
    "preferred": null
  }
}
```

`audio2face` contains exactly the 18 keys shown. All fields except
`eye_saccade_seed` are finite JSON floats. Their inclusive ranges are:

- `input_strength`: `[0.0, 3.0]`;
- `lower_face_smoothing`, `upper_face_smoothing`: `[0.0, 0.1]`;
- `lower_face_strength`, `upper_face_strength`, `skin_strength`,
  `blink_strength`, `eyeballs_strength`, `saccade_strength`: `[0.0, 2.0]`;
- `face_mask_level`: `[0.0, 1.0]` and `face_mask_softness`: `[0.001, 0.5]`;
- `eyelid_open_offset`: `[-1.0, 1.0]` and `lip_open_offset`: `[-0.2, 0.2]`;
- the four `*_eye_rot_*_offset` fields: `[-10.0, 10.0]`; and
- `eye_saccade_seed`: JSON integer in `[0, 4999]`.

`emotion_driver` has exactly `emotion_strength`, `generated`, and `preferred`.
`emotion_strength` is a finite float in `[0.0, 2.0]`. The generated source is
`null` or exactly:

```json
{
  "emotion_contrast": 1.0,
  "max_emotions": 6,
  "live_blend_coef": 0.7,
  "transition_smoothing": 0.5
}
```

Its fields are:

- `emotion_contrast`: finite float in `[0.1, 3.0]`;
- `max_emotions`: integer from `1` through the classifier's emotion count;
- `live_blend_coef`: finite float in `[0.0, 1.0]`;
- `transition_smoothing`: finite seconds in `[0.1, 1.0]`.

The Preferred source is `null` or exactly
`{"values":{"<every emotion>":0.5},"strength":0.5}`. `values` contains every
advertised emotion name exactly once with finite values in `[0.0, 1.0]`, and
`strength` is finite in `[0.0, 1.0]`.

Generated `G` is zero when its source is absent. With Preferred `P` and mix
weight `p`, the mixer is `pP + (1-p)G`; without Preferred it is `G`. Global
strength multiplies that result. Thus both absent produces zero, while
Preferred without generated produces the constant `emotion_strength * pP`.
Partial documents and unknown keys are rejected. `stream_start` installs one
complete snapshot, `stream_settings` replaces it at an ordered queue boundary,
and every `bake_frame` carries one complete snapshot.

## `stream_start`

Parameters contain exactly `operation_id`, `sample_rate`, and `settings`.
`operation_id` is non-empty and at most 128 characters, `sample_rate` equals the
rate returned by `load_model`, and `settings` is the complete object defined
under **Settings document**. The worker resets its retained regular executors,
installs that snapshot, and returns exactly:

```json
{"sample_rate":16000,"prebuffer_samples":60000}
```

`prebuffer_samples` is a non-negative integer at the model rate and is always
the greater of the Audio2Face input lead and Audio2Emotion readiness window.
It does not change when automatic emotion is toggled.

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

The worker replies `{}` when it accepts the chunk. After dequeuing that chunk
for inference, and before publishing any frame unlocked by it, the worker emits
one capacity credit:

```json
{"protocol":"audio2face/10","type":"event","event":"stream_credit","operation_id":"stream-1","data":{}}
```

Cancellation may end a stream without a credit for its final queued chunk. A
frame event has exact data fields `timestamp_sample`, `weights`, and
`effective_emotions`:

```json
{
  "protocol": "audio2face/10",
  "type": "event",
  "event": "stream_frame",
  "operation_id": "stream-1",
  "data": {"timestamp_sample": 0, "weights": [0.0], "effective_emotions": [0.0]}
}
```

The abbreviated `weights` array contains exactly 52 finite values in
`[0.0, 1.0]`, ordered by `model_schema.channels`. The abbreviated
`effective_emotions` array contains exactly one finite value for each entry in
`model_schema.emotion_channels`, in that order. These are the effective values
sampled by Audio2Face after Audio2Emotion post-processing and optional preferred
emotion mixing, not the raw classifier output. NVIDIA's SDK does not constrain
effective emotions to `[0.0, 1.0]`, so the transport preserves every finite
value. Both arrays describe the same Audio2Face frame and timestamp.

`timestamp_sample` is a signed 64-bit position at the model sample rate and is
strictly increasing for the operation. Events do not repeat channel names.

## `stream_settings`

Parameters contain exactly `operation_id` and `settings`. The `settings` value
is one complete object with the exact two top-level fields and compositional shape
defined under **Settings document**; partial objects and unknown keys remain
invalid.

The command occupies one position in the same queue as `stream_chunk` and
`stream_end`. All chunks queued before it are processed with the prior
snapshot, and all chunks queued after it are processed with the new snapshot.
The operation ID, models, audio transport, and strictly increasing published
timestamp sequence remain active. The response is exactly `{}` and is emitted
only after the operation thread has installed that queued snapshot. The worker
accepts at most one unapplied settings command; an additional direct request
returns `stream_backpressure`.

Subsequent PCM uses the new snapshot and the published timestamp sequence
remains strictly increasing.

## `stream_end`

Parameters are exactly `{"operation_id":"stream-1"}`. The worker replies `{}`,
processes the remaining safe frames, and then emits:

```json
{"protocol":"audio2face/10","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

Every final `stream_frame` precedes `stream_ended`. Both models remain loaded,
ready for another stream.

## `bake_start`

Parameters contain exactly:

```json
{"operation_id":"bake-1","sample_rate":16000}
```

`operation_id` follows the normal operation-ID rules and `sample_rate` must
equal the rate returned by `load_model`. No settings are installed at start.
The result contains exactly:

```json
{"sample_rate":16000,"max_chunk_samples":65536}
```

The response is emitted before upload begins. Starting a bake while a stream or
another bake is active returns `busy`.

## `bake_chunk`

Parameters contain exactly:

```json
{"operation_id":"bake-1","audio_f32le_base64":"AAAAAA=="}
```

The canonical base64 payload decodes to a non-empty mono block of finite
little-endian IEEE-754 float32 samples at the model rate. A block contains at
most the `max_chunk_samples` returned by `bake_start`. At most two maximum-size
blocks may wait in the worker queue; excess input returns
`stream_backpressure`. Each result is exactly:

```json
{"accepted_samples":1,"total_samples":1}
```

`accepted_samples` is the decoded block length and `total_samples` is the
cumulative uploaded length. Chunks are accepted only during the upload phase.

## `bake_prepare`

Parameters are exactly `{"operation_id":"bake-1"}`. It closes the completed
PCM input. Empty audio and a second prepare request are rejected. The response
contains exactly:

```json
{
  "audio_samples": 48000,
  "source_frame_count": 91,
  "source_frame_rate_numerator": 30,
  "source_frame_rate_denominator": 1,
  "first_source_timestamp_sample": 0,
  "last_source_timestamp_sample": 48000,
  "sampling": "linear"
}
```

`audio_samples` is the positive uploaded sample count; `source_frame_count`
and both frame-rate components are positive integers; the source timestamps
are signed 64-bit sample positions with first not greater than last; and
`sampling` is exactly `"linear"`.

## `bake_frame`

After preparation, the client submits one target-sample request at a time. With
`SETTINGS` denoting the complete Settings document, the request is:

```text
{"operation_id":"bake-1","target_sample":22400,"settings":SETTINGS}
```

`SETTINGS` is the complete object defined under **Settings document**.
`target_sample` is a non-negative signed 64-bit integer in
`[0, audio_samples - 1]`. Under that snapshot, the worker computes the one or
two source frames bracketing `target_sample` and linearly interpolates them,
clamping to an endpoint when needed. The response contains exactly:

```json
{
  "weights": [0.25]
}
```

The result contains exactly `weights`: 52 finite values in `[0.0, 1.0]`, in
`model_schema.channels` order. Only one bake-frame request may be outstanding;
a concurrent request returns `stream_backpressure`. There is no asynchronous
bake-frame event.

## `bake_end`

After the last bake-frame response, parameters are exactly
`{"operation_id":"bake-1"}`. The worker responds `{}`, releases the active
bake state, and then emits:

```json
{"protocol":"audio2face/10","type":"event","event":"bake_ended","operation_id":"bake-1","data":{"reason":"completed"}}
```

`bake_ended` data contains exactly `reason`, which is either `completed` or
`canceled`. A non-cancellation operation failure emits the exact `error` event
instead of `bake_ended`. Both models remain loaded.

## `cancel`

Parameters are exactly `{"operation_id":"<active-id>"}`. A matching active
operation receives an immediate `{}` response. Queued input and execution stop
without draining. A stream emits `stream_ended {}`; a bake emits
`bake_ended {"reason":"canceled"}`. The terminal event follows the cancel
response. An unknown, inactive, or already terminal ID returns
`operation_not_found`.

## `shutdown`

Parameters and result are both `{}`. Shutdown stops and joins any active
operation, responds, and exits the protocol loop.

## Terminal operation errors

An asynchronous inference failure uses an `error` event with exact `code` and
`message` data:

```json
{"protocol":"audio2face/10","type":"event","event":"error","operation_id":"stream-1","data":{"code":"inference_failed","message":"operation failed"}}
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
