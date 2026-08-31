# Worker protocol `audio2face/13`

## Transport

Blender owns one local worker child and exchanges UTF-8 JSON Lines over stdin
and stdout. stdout is protocol-only; diagnostics use stderr. Every record is one
JSON object followed by LF, with a 1 MiB payload limit. Duplicate keys,
non-finite numbers, malformed UTF-8, CR, blank records, and multiple records on
one line are rejected.

A request, response, request error, and asynchronous event have these exact
envelopes:

```json
{"protocol":"audio2face/13","type":"request","id":"1","method":"hello","params":{}}
{"protocol":"audio2face/13","type":"response","id":"1","result":{}}
{"protocol":"audio2face/13","type":"error","id":"1","error":{"code":"invalid_params","message":"invalid request","details":{}}}
{"protocol":"audio2face/13","type":"event","event":"stream_ended","operation_id":"stream-1","data":{}}
```

Request IDs and operation IDs are non-empty strings of at most 128 characters.
The only methods are `hello`, `load_model`, `stream_start`, `stream_chunk`,
`stream_settings`, `stream_end`, `track_start`, `track_chunk`, `track_prepare`,
`track_render`, `cancel`, and `shutdown`. Events are `stream_credit`,
`stream_frame`, `stream_ended`, `track_preview`, `track_frame_batch`,
`track_ended`, and `error`.

One worker accepts one active audio operation: either a sequential Stream or a
persistent Selected track.

## Handshake and model

`hello` takes `{}` and returns exactly:

```json
{"worker_profile":"nvidia-a2f3-a2e3-gpu-arkit52/13","worker_version":"0.1.0"}
```

`load_model` takes the two validated absolute top-level descriptors:

```json
{
  "audio2face_model_path": "/models/audio2face/model.json",
  "audio2emotion_model_path": "/models/audio2emotion/model.json"
}
```

It allocates device-0 resources and returns exactly `sample_rate` and
`model_schema`. The schema contains `channels` (52 unique names in model order),
ordered `emotion_channels` records with `name` and `default`, and the loaded
model's `audio2face_defaults`. Identity is fixed to SDK index 0 and is not a
protocol option.

## Settings document

`stream_start` and `stream_settings` carry one complete object. The first entry
of a `track_render` settings timeline carries the same complete object:

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
    "generated": {
      "emotion_contrast": 1.0,
      "max_emotions": 6,
      "live_blend_coef": 0.7,
      "transition_smoothing": 0.5
    },
    "preferred": {
      "values": {"<every model emotion>": 0.0},
      "strength": 0.5
    }
  }
}
```

Unknown or partial fields are rejected. `generated` and `preferred` may each
be null. Preferred `values` contains every advertised emotion exactly once.
All values are finite. Ranges are:

- `input_strength`: `[0, 3]`;
- face smoothing: `[0, 0.1]`;
- face, blink, eyeball, and saccade strengths: `[0, 2]`;
- `face_mask_level`: `[0, 1]`; `face_mask_softness`: `[0.001, 0.5]`;
- `eyelid_open_offset`: `[-1, 1]`; `lip_open_offset`: `[-0.2, 0.2]`;
- eye rotation offsets: `[-10, 10]`; `eye_saccade_seed`: integer `[0, 4999]`;
- emotion strength: `[0, 2]`; contrast: `[0.1, 3]`;
- max emotions: integer `1..emotion_count`;
- live blend, preferred values, and preferred strength: `[0, 1]`; and
- transition smoothing: `[0.1, 1]` seconds.

## Sequential Stream

### `stream_start`

Parameters are exactly `operation_id`, the model `sample_rate`, and `settings`.
The worker initializes the regular sequential executor family and returns:

```json
{"sample_rate":16000,"prebuffer_samples":60000}
```

### `stream_chunk`

Parameters are exactly:

```json
{"operation_id":"stream-1","audio_f32le_base64":"AAAAAA=="}
```

The canonical base64 payload is a non-empty, finite, mono little-endian float32
block. One request contains at most one model-rate second; the worker bounds
queued PCM to four seconds. Acceptance returns `{}`. When the worker dequeues a
chunk, before frames unlocked by it, it emits `stream_credit {}`.

Each `stream_frame` event contains exactly:

```json
{
  "timestamp_sample": 0,
  "weights": [0.0],
  "effective_emotions": [0.0]
}
```

`weights` has 52 finite values in `[0, 1]` in schema order.
`effective_emotions` has one finite value per emotion channel and is not clamped
to `[0, 1]`. Timestamps are model-rate sample positions and strictly increase.

### `stream_settings`

Parameters are exactly `operation_id` and a complete `settings` document. The
worker coalesces all snapshots pending at the next chunk boundary to their
newest value, acknowledges every request ID, applies that value once, and then
services one already-credited PCM command. It does not reset executors, retain
or replay audio, move timestamps, starve PCM, or affect media transport.

### `stream_end`

Parameters are `{"operation_id":"stream-1"}`. The response is `{}`. The worker
closes input, drains final frames, then emits `stream_ended {}`. All final frame
events precede the terminal event.

## Persistent Selected track

### Upload

`track_start` takes `operation_id` and model `sample_rate`, retains the regular
sequential executor family, and returns `{}`. `track_chunk` accepts the same
canonical PCM encoding as Stream, with at most 65,536 samples per request and
at most two maximum blocks queued. Each accepted chunk returns `{}`.

`track_prepare` takes only `operation_id`, marks the complete retained PCM ready,
and returns `{}`. It does not run inference or change Blender transport.

### `track_render`

After preparation, the request is exactly:

```json
{
  "operation_id": "track-1",
  "revision": 7,
  "settings_timeline": [
    {
      "sample": 0,
      "settings": {"audio2face": {"...": "complete"}, "emotion_driver": {"...": "complete"}}
    },
    {"sample": 16000, "settings": {"audio2face": {"skin_strength": 1.4}}},
    {"sample": 32000, "settings": {"emotion_driver": {"preferred": null}}}
  ],
  "preview_sample": 22400
}
```

The abbreviated first `settings` value above denotes the complete Settings
document. `settings_timeline` is non-empty; every entry has exactly `sample`
and `settings`. The first sample is 0. Later samples are strictly increasing,
inside the retained audio, and carry recursive changed-leaf object patches.
Objects merge recursively, while scalar, array, and null values replace the
previous value; fields are never deleted. The worker cumulatively expands and
validates every entry as a complete Settings document. A setting becomes active
from its sample onward.

`revision` is a strictly increasing positive integer. `preview_sample` is null
or an in-range non-negative audio sample.

The worker resets its regular sequential executors, supplies the retained PCM,
and applies expanded settings as Audio2Emotion and Audio2Face advance through
the source samples. The continuous pass preserves temporal and recurrent state;
the worker does not run stateless per-frame inference.

When `preview_sample` is present, the worker linearly samples the continuous
result and emits `track_preview` as soon as sequential output brackets that
sample. If the requested sample is later than the final output row, the worker
uses that last row. Full-track inference continues before cache transfer:

```json
{
  "revision": 7,
  "timestamp_sample": 22400,
  "weights": [0.25],
  "effective_emotions": [0.0]
}
```

The timestamp equals the requested preview sample. Its values are identical to
sampling the cache rows that follow; the event itself does not publish a cache.

A completed candidate is emitted in ordered batches of 1 through 64 rows:

```json
{
  "revision": 7,
  "offset": 0,
  "total_frames": 67,
  "timestamp_samples": [0, 267],
  "weights": [[0.1], [0.2]],
  "effective_emotions": [[0.0], [0.1]]
}
```

The three arrays are parallel; timestamps strictly increase; `offset` is the
first row's zero-based position. Every `track_frame_batch` for a revision
precedes its completion response:

```json
{"revision":7,"frame_count":67,"superseded":false}
```

That response is the publication barrier. Blender commits the staged cache
only when all `frame_count` rows arrived and `superseded` is false.

A newer revision replaces any queued render. An active render observes the new
revision between sequential executions and stops cooperatively without
canceling the resident track. Displaced requests receive
`{"revision":N,"frame_count":0,"superseded":true}`; stale frame events are
suppressed or ignored. Cancel is reserved for ending the track.

## Cancel, shutdown, and errors

`cancel` takes the active `operation_id` and immediately returns `{}`. It stops
queued execution without draining. A Stream emits `stream_ended {}`; a track
emits `track_ended {"reason":"canceled"}`. Unknown or terminal IDs return
`operation_not_found`.

`shutdown` takes and returns `{}`, stops and joins any operation, and exits.

An operation failure is asynchronous:

```json
{"protocol":"audio2face/13","type":"event","event":"error","operation_id":"track-1","data":{"code":"inference_failed","message":"operation failed"}}
```

Validation failures use request error envelopes. Worker codes include
`invalid_json`, `invalid_request`, `request_too_large`, `protocol_mismatch`,
`method_not_found`, `invalid_params`, `invalid_state`, `busy`, `model_invalid`,
`model_not_loaded`, `operation_not_found`, `sample_rate_mismatch`,
`backpressure`, `inference_failed`, `sdk_error`, `gpu_error`, and
`internal_error`.
