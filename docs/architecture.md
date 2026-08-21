# Architecture

## Fixed scope

Audio2Face Blender has two input modes on one production inference path:

```text
selected WAV -------- complete-file generation -------- a2f-animation/1
     \
      +-- incremental WAV source --+
live mono f32le PCM ----------------+-- stream_start/chunk/end
                                        |
Blender 5.2 extension
    |  private a2f-blender/2 JSONL over stdin/stdout (no listener)
managed native child process and one shared CUDA stream
    |
shared audio accumulator --------+--------------------------+
                                 |                          |
                                 v                          v
                      Audio2Emotion v3.0            Audio2Face-3D v3.0
                         (Auto on)                     ^
                                 \                       /
manual model emotion (Auto off) --+-> shared emotion --+
                                      accumulator       |
                                                 canonical ARKit-52 frames
                                                          |
Blender main thread -> exact-name Shape Key values on enabled target meshes
```

The worker is local, GPU-only, and owned by Blender. It opens no port, and
Blender never loads CUDA, TensorRT, or Audio2X into its own process. Installing
or enabling the extension does not start the worker. Only an explicit
generation request or active PCM stream runs inference.

There is no geometry, bone, vertex-deformation, or animation-baking output.
The sole output is timestamped frames of 52 ARKit coefficients. Selected mode
buffers a complete result; Stream mode delivers frames incrementally and does
not write a result, Action, or animation curve.

## Blender extension

The extension owns:

- managed-runtime installation and receipt validation;
- worker start, handshake, model load, generation, streaming, cancellation,
  and stop;
- WAV, input-mode, and target-mesh selection;
- Audio2Face controls, model-defined manual emotion values, and the
  Audio2Emotion override toggle and post-processing controls;
- strict result and event validation; and
- bounded PCM sourcing, audio synchronization, and Shape Key updates.

Each enabled target is a mesh in the target collection. Selection is only the
onboarding action used by **Add Selected Meshes**. A Shape Key is connected only
when its name exactly matches a PascalCase name in
[`a2f_blender/arkit.py`](../a2f_blender/arkit.py). Matching is case-sensitive,
and each mesh may implement any subset of the 52 names. There is no name map,
per-channel multiplier, offset, or destination selector.

Several objects may share one Shape Key datablock. The playback controllers
deduplicate those datablocks so each frame writes a shared datablock once.
Every linked object using that datablock necessarily reflects the values;
independent motion requires single-user mesh data.

Worker I/O runs on standard-library threads. Those threads enqueue validated
messages and never mutate `bpy`. A registered Blender timer drains the queues
and performs every RNA and Shape Key mutation on the main thread.

## Managed worker

The worker owns:

- CUDA device 0 and the Audio2X runtime;
- one Audio2Face-3D v3.0 diffusion model;
- one Audio2Emotion v3.0 classifier model;
- selected-mode WAV decoding, downmixing, and resampling;
- a shared CUDA stream plus incremental audio and emotion accumulators;
- one non-interactive device blendshape executor and one Audio2Emotion
  executor shared by both input modes;
- direct Audio2Emotion-to-Audio2Face emotion binding when Auto is enabled, or
  one constant model-shaped manual emotion vector when it is disabled;
- fixed SDK-name-to-ARKit resolution;
- atomic selected-mode result publication; and
- live ARKit frame publication.

stdin and stdout carry only UTF-8 JSON Lines using `a2f-blender/2`.
Diagnostics use stderr. Selected mode writes a complete result to the absolute
path submitted by Blender and then emits an empty `result` event. Stream mode
sends one bounded audio chunk per request and returns one 52-value frame per
`stream_frame` event. No geometry or complete streamed track crosses the pipe.

## Managed installation

The release catalog contains one artifact record per supported platform. Each
record has an immutable HTTPS URL, exact compressed size, exact unpacked size,
and SHA-256 digest. Installation:

1. acquires a lock shared by Blender processes;
2. downloads into temporary storage;
3. verifies that the final URL remains credential-free HTTPS, then validates
   byte count and digest;
4. extracts canonical paths with bounded member count and total size;
5. validates `bundle.json`, x86-64 executables, both model input sets,
   libraries, and notices;
6. uses the bundled, release-built NVIDIA TensorRT `trtexec` to create the
   local GPU's separate Audio2Face and Audio2Emotion `network.trt` engines;
7. writes the catalog receipt; and
8. atomically activates the completed platform directory.

Cancellation is honored before activation. A failed installation cannot
replace a verified active runtime. The resolved runtime root and every
manifest member must remain inside Blender's extension data directory:

```python
bpy.utils.extension_path_user(__package__, path="", create=True)
```

The worker, both models, `trtexec`, and runtime libraries all come from that
verified bundle. No executable, SDK, model, working directory, system
installation, hosted service, or external host application is selected or used.

The checked-in catalog is deliberately empty. End-user installation becomes
operational only when release maintainers publish license-reviewed platform
archives and add their measured artifact records. Audio2Emotion v3.0 is gated
by NVIDIA on Hugging Face; the integration remains experimental until model
access, redistribution, GPU, and platform validation is complete. Release
validation must exercise the exact pinned model pair and confirm the
Audio2Emotion post-processed vector ordering against Audio2Face: SDK 1.0.0
reports that vector's width but does not expose its output channel names.
Neither the extension nor its catalog contains user credentials.

## Lifecycle

```text
IDLE --Start Worker--> STARTING --hello--> LOADING_MODEL --> MODEL_READY
                                                            |
                              +-----------------------------+----------------+
                              |                                              |
                           Generate                                    Start Stream
                              v                                              v
                          GENERATING                                     STREAMING
                         /    |     \                                   /   |    \
                   result  canceled  error                           end  stop  error
                     |        |        |                              |    |     |
                COMPLETED  MODEL_READY ERROR                       MODEL_READY ERROR

Any live worker --Stop Worker--> STOPPING --> IDLE
Unexpected exit or rejected contract --> ERROR
```

**Start Worker** launches the verified child, sends `hello {}`, and loads both
managed models. Model loading allocates CUDA/model resources but does not run
continuous inference. One worker runs at most one generation job or stream.

**Generate ARKit Values** reloads the model only when identity changes, then
submits one complete WAV. **Cancel Generation** interrupts active execution;
the operation thread is joined before model reuse or shutdown.

**Start WAV Stream** freezes the same face-and-emotion controls, prepares
enabled targets, incrementally decodes and resamples the selected WAV, and
pushes bounded mono f32le chunks. `stream_start` returns the exact model rate
and an initial `prebuffer_samples` requirement. It is the Audio2Face audio lead
needed before the first face execution can be ready; with Auto enabled it is
the larger of that lead and the Audio2Emotion readiness window. When the first
ARKit frames arrive, Blender starts the selected WAV and thereafter keeps a
bounded PCM lead. Frames are sampled against
audio-device position, rather than applied at pipe-arrival time or scene FPS.

The public `a2f_blender.streaming` module exposes the same source-agnostic
begin/chunk/end contract to integrations already running in Blender. Those
integrations submit model-rate mono f32le and own capture, resampling, and
audible monitoring. After `start_pcm_stream(scene)`, they poll
`get_pcm_stream_requirements(scene)`; it returns `None` while startup is pending
and then returns `(sample_rate, prebuffer_samples)`. Both calls default to the
context scene. Sources must queue that initial lead before synchronized
monitoring. No socket is opened. **Stop Stream** cancels only that stream and
returns to `MODEL_READY`. **Stop Worker** requests bounded shutdown and
escalates to process termination only if the child misses its deadlines.
Destruction releases the Audio2Face and Audio2Emotion executors, model metadata,
accumulators, and CUDA stream in dependency order.

## Model and ARKit contract

Model loading accepts exactly absolute managed `audio2face_model_path` and
`audio2emotion_model_path` values plus a non-negative identity index. The
worker implements only Audio2Face v3 diffusion and Audio2Emotion v3 classifier
APIs. It requires the models to agree on sample rate and emotion-vector shape.
Scene FPS is not an input; output timing uses signed integer audio-sample
positions.

Audio2Face metadata defines the ordered emotion-channel names and default
manual vector. Audio2Emotion classifier results are post-processed and bound
directly into the same Audio2Face emotion accumulator. Emotion names or vector
dimensions are never hard-coded by the Blender extension.

After the SDK removes its leading neutral pose, the skin solver must expose the
exact unique set of 52 lowerCamelCase ARKit names. SDK order is not trusted.
The worker resolves every index by name and emits the fixed PascalCase order.
That skin set already ends in `tongueOut`. The model's separate 16-pose
tongue-detail solver produces non-ARKit geometry controls, so the ARKit-only
worker does not enable or expose it.

Six SDK eye-rotation components (`right XYZ`, `left XYZ`) are required. Their
X/Y components resolve the eight directional `EyeLook*` coefficients with a
fixed 60-degree unit range. Raw geometry, jaw transforms, eye rotations, and
other solver outputs never leave the worker.

Every selected-mode `a2f-animation/1` document has exactly five fields:

- `schema`
- `job_id`
- `sample_rate`
- `timestamps_samples`
- `weights`

Every row contains exactly 52 finite values in `[0.0, 1.0]`. Timestamps are
non-empty, strictly increasing signed 64-bit sample indices, and row count
equals timestamp count. Live `stream_frame` events carry one timestamp and the
same exact 52-value row.

## Controls and playback

Every generation and stream-start request contains the complete
`input_strength`, `skin`, and `emotion` parameter document. Model loading
returns the sample rate, model emotion names, model-defined manual defaults,
and Audio2Emotion post-processing defaults. Unknown, partial, non-numeric, or
non-finite documents are rejected. Stream controls are frozen at start;
changing them requires stopping and starting that stream.

`emotion.auto_audio2emotion` is the sole mode switch. When false, the worker
adds the complete `manual_values` vector at timestamp zero and closes the
emotion accumulator, making that vector the constant driver for the whole
operation. When true, the worker resets the Audio2Emotion executor, sends the
same accumulated audio through it, and binds its timestamped output into the
Audio2Face emotion accumulator; manual values are ignored, not blended. The
automatic postprocessor exposes strength `[0, 1]`, contrast `[0.1, 3]`,
smoothing `[0, 1]`, transition time `[0.1, 1]` seconds, and `max_emotions` from
one through the classifier's reported emotion count. Preferred-emotion mixing
is disabled, so Auto is a complete replacement rather than a blend with the
manual driver. The same behavior applies to Selected and Stream.

After selected generation, Blender loads the result only from its managed
results directory and verifies the active job ID. Playback uses selected WAV
audio position, linearly samples between timestamps, and assigns the 52-value
frame to matching Shape Keys. Play, pause, resume, stop, loop, volume, and
reset-on-stop are Blender-local and do not invoke inference.

For a streamed WAV, a bounded jitter buffer stores incremental ARKit frames and
samples them using local audio position. Source-agnostic PCM uses a monotonic
presentation clock anchored to its first returned timestamp, so a model window
that returns several frames at once is still interpolated over time instead of
collapsing into one visible jump. After worker input ends, the streamed-WAV
transport and Stop button remain active until its buffered audio finishes.
Stream end and explicit stop never create a result file.

## Failure boundaries

The extension accepts only a `hello` result containing the exact
`nvidia-a2f3-a2e3-gpu-arkit52/1` profile and a non-empty worker version. Control
messages reject missing or unknown fields, duplicate JSON keys, non-finite
numbers, invalid IDs, unknown methods or events, malformed UTF-8, and JSON
payloads over 1 MiB.

PCM chunks are non-empty finite little-endian float32 values, capped at one
model-rate second and never above 256 KiB before base64 encoding. At most 64
chunk acknowledgements may be pending,
so a source that outruns inference receives bounded backpressure instead of
causing unbounded Blender memory growth. Stream IDs, event kinds, timestamps,
response sample rates, and every 52-value row are validated before mutation.

Result files are confined to the managed results directory, capped at 512 MiB,
strictly validated, and published without replacing an existing file.
Installer and worker failures appear in Blender's status panel. Unexpected
child exit clears the loaded-model and stream state and reports the latest
worker diagnostic.

Unregistering the extension stops buffered and live playback, cancels its WAV
source and installation, unregisters the timer, and closes the worker before
removing Blender classes and scene RNA.
