# Worker protocol `audio2face/2`

## Transport

Blender starts one local child process and exchanges UTF-8 JSON Lines over its
stdin and stdout. The process is silent until `hello {}`. stdout is
protocol-only; diagnostics go to stderr. No socket or user-hosted service is
involved.

Each record contains exactly one JSON object and one LF or CRLF line ending.
The JSON payload, excluding that ending, is limited to 1 MiB. Blank records,
duplicate keys, non-finite JSON numbers, invalid UTF-8, and extra records are
rejected.

Every request has exactly these fields:

```json
{"protocol":"audio2face/2","type":"request","id":"1","method":"hello","params":{}}
```

`id` is a non-empty string of at most 128 characters. A successful response
repeats it and contains an object result. A request error contains exact
`code`, `message`, and `details` fields; `id` is present only when recoverable:

```json
{"protocol":"audio2face/2","type":"error","id":"1","error":{"code":"invalid_params","message":"invalid request","details":{}}}
```

Every asynchronous event has exact `event`, `job_id`, and object `data` fields
alongside the common fields. `job_id` correlates either a selected generation
job or a live stream:

```json
{"protocol":"audio2face/2","type":"event","event":"stream_ended","job_id":"stream-1","data":{}}
```

The only request methods are `hello`, `load_model`, `generate`, `stream_start`,
`stream_chunk`, `stream_end`, `cancel`, and `shutdown`. The only events are
`progress`, `result`, `canceled`, `stream_frame`, `stream_ended`, and `error`.

The worker accepts one active generation job or stream. For every asynchronous
operation, its immediate response is written before any event from that
operation.

## Methods

### `hello`

Parameters are exactly `{}`. The result is exactly:

```json
{"worker_profile":"nvidia-a2f3-a2e3-gpu-arkit52/1","worker_version":"0.1.0"}
```

Blender requires that profile and a non-empty version. `hello` must succeed
before model or inference methods. CUDA/model allocation is not a handshake
side effect.

### `load_model`

Parameters are exactly:

```json
{
  "audio2face_model_path": "/absolute/managed/models/audio2face/model.json",
  "audio2emotion_model_path": "/absolute/managed/models/audio2emotion/model.json",
  "identity_index": 0
}
```

Both paths must be absolute managed files. `identity_index` must be a
non-negative integer within the Audio2Face identity range. Loading creates the
device-0 Audio2Face diffusion/blendshape executor, Audio2Emotion classifier
executor and its checked result callback, their shared CUDA stream, and shared
audio and emotion accumulators. The models must agree on sample rate and
emotion-vector shape.
Loading does not execute either model or start continuous inference.

The result has exactly `parameter_defaults`, `emotion_names`, and
`sample_rate`. Nested keys are fixed. Values, the ordered emotion names, and
the positive integer sample rate come from the managed models. The names and
values below are illustrative; `manual_values` must have the exact name set,
while `emotion_names` supplies its canonical order:

```json
{
  "parameter_defaults": {
    "input_strength": 1.0,
    "skin": {
      "lower_face_smoothing": 0.0,
      "upper_face_smoothing": 0.0,
      "lower_face_strength": 1.0,
      "upper_face_strength": 1.0,
      "face_mask_level": 0.5,
      "face_mask_softness": 0.1,
      "skin_strength": 1.0,
      "blink_strength": 1.0,
      "eyelid_open_offset": 0.0,
      "lip_open_offset": 0.0,
      "blink_offset": 0.0
    },
    "emotion": {
      "manual_values": {
        "ModelEmotionA": 0.0,
        "ModelEmotionB": 0.0
      },
      "auto": {
        "strength": 0.6,
        "contrast": 1.0,
        "smoothing": 0.7,
        "transition_time": 0.5,
        "max_emotions": 2
      }
    }
  },
  "emotion_names": ["ModelEmotionA", "ModelEmotionB"],
  "sample_rate": 16000
}
```

`manual_values` is the Audio2Face model's default conditioning vector, keyed
by the names reported by that model. `auto` contains the Audio2Emotion
post-processing defaults. Blender builds its manual emotion controls from this
response; it does not hard-code an emotion list.

### Settings document

Every `generate` and `stream_start` request contains this complete exact
settings shape. The model-defined names shown here are illustrative and must
match the active `load_model` response:

```json
{
  "input_strength": 1.0,
  "skin": {
    "lower_face_smoothing": 0.0,
    "upper_face_smoothing": 0.0,
    "lower_face_strength": 1.0,
    "upper_face_strength": 1.0,
    "face_mask_level": 0.5,
    "face_mask_softness": 0.1,
    "skin_strength": 1.0,
    "blink_strength": 1.0,
    "eyelid_open_offset": 0.0,
    "lip_open_offset": 0.0,
    "blink_offset": 0.0
  },
  "emotion": {
    "auto_audio2emotion": false,
    "manual_values": {
      "ModelEmotionA": 0.0,
      "ModelEmotionB": 0.0
    },
    "auto": {
      "strength": 0.6,
      "contrast": 1.0,
      "smoothing": 0.7,
      "transition_time": 0.5,
      "max_emotions": 2
    }
  }
}
```

`manual_values` must contain exactly every name returned by `load_model`, with
finite values in `[0, 1]`. `auto.strength` and `auto.smoothing` are finite in
`[0, 1]`; `auto.contrast` is finite in `[0.1, 3]`; `auto.transition_time` is
finite in `[0.1, 1]` seconds; and `auto.max_emotions` is an integer from one
through the Audio2Emotion classifier's reported emotion count. All fields
remain required regardless of the toggle.

When `auto_audio2emotion` is false, `manual_values` is accumulated at timestamp
zero as the constant emotion driver. When true, Audio2Emotion v3.0 analyzes the
same operation audio and its timestamped output replaces the manual vector.
There is no blending between the two modes, and preferred-emotion mixing is
disabled. The selected-file and streaming paths use this identical switch and
parameter contract.

### `generate`

Parameters are exactly `job_id`, absolute `audio_path`, absolute managed
`result_path`, and the complete settings document:

```json
{
  "job_id": "job-1",
  "audio_path": "/absolute/path/speech.wav",
  "result_path": "/absolute/managed/results/job-1.a2f.json",
  "settings": {
    "input_strength": 1.0,
    "skin": {
      "lower_face_smoothing": 0.0,
      "upper_face_smoothing": 0.0,
      "lower_face_strength": 1.0,
      "upper_face_strength": 1.0,
      "face_mask_level": 0.5,
      "face_mask_softness": 0.1,
      "skin_strength": 1.0,
      "blink_strength": 1.0,
      "eyelid_open_offset": 0.0,
      "lip_open_offset": 0.0,
      "blink_offset": 0.0
    },
    "emotion": {
      "auto_audio2emotion": false,
      "manual_values": {
        "ModelEmotionA": 0.0,
        "ModelEmotionB": 0.0
      },
      "auto": {
        "strength": 0.6,
        "contrast": 1.0,
        "smoothing": 0.7,
        "transition_time": 0.5,
        "max_emotions": 2
      }
    }
  }
}
```

Partial or unknown settings and non-numeric/non-finite values are rejected.
Audio is one complete RIFF/WAVE file of at most 512 MiB: integer PCM at 8, 16,
24, or 32 bits, or IEEE float32. The worker downmixes channels and linearly
resamples to model rate. Manual mode supplies the configured constant driver;
Auto mode runs Audio2Emotion on that same decoded audio and binds its generated
emotion stream to Audio2Face.

`TongueOut` is already one of the model's 52 skin weights; the separate
non-ARKit tongue-detail solver is not part of this protocol.

The immediate response is `{}`. Progress events contain exact finite
`progress` in `[0,1]` and a non-empty `stage`. The worker feeds the same
incremental executor used by Stream mode, collects all frames, atomically
publishes the five-field result, then emits:

```json
{"protocol":"audio2face/2","type":"event","event":"result","job_id":"job-1","data":{}}
```

Blender derives the managed `results/<job_id>.a2f.json` path it submitted and
requires it to exist. The worker never replaces a result or publishes a partial
document.

### `stream_start`

Parameters are exactly:

```json
{
  "stream_id": "stream-1",
  "sample_rate": 16000,
  "settings": {
    "input_strength": 1.0,
    "skin": {
      "lower_face_smoothing": 0.0,
      "upper_face_smoothing": 0.0,
      "lower_face_strength": 1.0,
      "upper_face_strength": 1.0,
      "face_mask_level": 0.5,
      "face_mask_softness": 0.1,
      "skin_strength": 1.0,
      "blink_strength": 1.0,
      "eyelid_open_offset": 0.0,
      "lip_open_offset": 0.0,
      "blink_offset": 0.0
    },
    "emotion": {
      "auto_audio2emotion": true,
      "manual_values": {
        "ModelEmotionA": 0.0,
        "ModelEmotionB": 0.0
      },
      "auto": {
        "strength": 0.6,
        "contrast": 1.0,
        "smoothing": 0.7,
        "transition_time": 0.5,
        "max_emotions": 2
      }
    }
  }
}
```

`stream_id` follows the job-ID bounds. `sample_rate` must be the exact model
rate returned by `load_model`; resampling belongs to the source adapter.
Settings are validated and frozen until the stream ends. The worker resets its
incremental executors and accumulators, installs either the manual driver or
the Audio2Emotion binding selected by the toggle, and replies with exactly:

```json
{"sample_rate":16000,"prebuffer_samples":60000}
```

`prebuffer_samples` is a non-negative integer at the returned sample rate. It
is the Audio2Face audio lead required before the first face execution can be
ready. In Auto mode it is the maximum of that lead and the Audio2Emotion
executor's readiness window. A source must submit at least this initial lead
before beginning synchronized audible playback; Blender integrations must use
the returned value rather than hard-code it.

### `stream_chunk`

Parameters are exactly:

```json
{"stream_id":"stream-1","audio_f32le_base64":"AAAAAA=="}
```

The base64 text must be canonical and decode to a non-empty block no longer
than one model-rate second, with byte count divisible by four. Each
little-endian float32 mono sample must be finite. The
stream ID must match the active stream. The worker accumulates the samples,
executes every model window made ready, and replies `{}` before publishing any
frames unlocked by that request.

Each frame event has exact data:

```json
{
  "timestamp_sample": 0,
  "weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

`timestamp_sample` is a strictly increasing signed-64-bit model sample
position. `weights` is the fixed 52-channel `arkit-52/1` order with finite
values in `[0,1]`. Audio samples no longer needed by either inference stage are
dropped from the accumulator.

### `stream_end`

Parameters are exactly `{"stream_id":"stream-1"}`. The worker replies `{}`,
closes input, drains all padded tail frames, waits for scheduled GPU work, and
emits the terminal event only after all `stream_frame` events:

```json
{"protocol":"audio2face/2","type":"event","event":"stream_ended","job_id":"stream-1","data":{}}
```

The model remains loaded and ready. No result file is written.

### `cancel`

Parameters are exactly `{"job_id":"<active-id>"}` and address either active
operation type. A matching ID receives an immediate `{}` response.

For selected generation, execution is interrupted, no partial result is
published, and completion is `canceled {}`. For a stream, queued input and
execution stop without draining and completion is `stream_ended {}`. An
unmatched or inactive ID returns `job_not_found`. Atomic result publication
that already won a cancellation race remains successful.

### `shutdown`

Parameters and result are both `{}`. Shutdown cancels the active operation,
joins its thread, responds, and exits the protocol loop. Backend destruction
synchronizes CUDA and releases both executors, both model metadata objects, the
shared accumulators, and the CUDA stream. Blender
enforces bounded graceful, terminate, and kill deadlines.

## Result schema `a2f-animation/1`

The selected-mode document has exactly `schema`, `job_id`, `sample_rate`,
`timestamps_samples`, and `weights`. The schema fixes the channel order to the
52 names in [`audio2face/arkit.py`](../audio2face/arkit.py); names are not
repeated in the file. Validation requires:

- a non-empty `job_id` of at most 128 characters;
- a positive uint32 `sample_rate`;
- non-empty, strictly increasing signed-64-bit sample timestamps;
- one weight row per timestamp;
- exactly 52 numeric values per row; and
- every coefficient finite and within `[0.0, 1.0]`.

The SDK skin names must be the exact unique lowerCamelCase ARKit-52 set. The
worker resolves indices by name and emits the fixed PascalCase order. Six SDK
eye-rotation components resolve into the eight `EyeLook*` values. Raw geometry,
jaw transforms, eye rotations, and other solver outputs are absent.

## Error codes

The stable request/operation codes are:

- `invalid_json`, `invalid_request`, `request_too_large`, `protocol_mismatch`
- `method_not_found`, `invalid_params`, `invalid_state`, `busy`
- `model_not_found`, `model_invalid`, `identity_invalid`, `model_not_loaded`
- `audio_open_failed`, `audio_invalid`, `invalid_audio`, `unsupported_audio`,
  `audio_too_large`
- `job_not_found`, `generation_failed`, `sample_rate_mismatch`,
  `stream_backpressure`
- `sdk_error`, `gpu_error`, `internal_error`
- `invalid_result_path`, `result_exists`, `result_directory_failed`
- `result_too_large`, `result_write_failed`, `result_commit_failed`

Operation failures use an `error` event with exact `code` and `message` data.
Request validation failures use the request error envelope.
