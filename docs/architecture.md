# Architecture

## System boundary

Audio2Face has two audio modes on one GPU inference backend:

```text
selected WAV ---------------- complete generation ------------ a2f-animation/2
     |
     +-- incremental WAV decoder --+
live mono f32le PCM --------------+-- stream_start/chunk/end
                                      |
Blender 5.2 extension
    |  private audio2face/3 JSONL over stdin/stdout
managed native worker on CUDA device 0
    |
shared audio accumulator --------+--------------------------+
                                 |                          |
                         Audio2Emotion v3.0         Audio2Face-3D v3.0
                            (Auto on)                       ^
                                 |                          |
manual emotion vector (Auto off)-+-> emotion accumulator --+
                                                            |
                                      model-ordered ARKit coefficients
                                                            |
Blender main thread -> per-frame Shape Key lookup on enabled mesh targets
```

The worker is a local child owned by Blender. It opens no port, and CUDA,
TensorRT, Audio2X, and the model executors remain outside Blender's process.
Installing or enabling the extension does not start the worker. Loading both
models allocates their resources but does not run continuous inference. Only a
generation job or active PCM stream executes the models.

The only animation output is timestamped coefficient frames. The extension
does not write geometry, vertices, bones, Actions, F-curves, or baked
animation. Selected WAV buffers a complete result. Stream delivers frames
incrementally and creates no result file.

## Blender extension

The extension owns:

- Add-on Preferences for managed setup, one NVIDIA terms acceptance, model
  source links, install/repair, cancellation, and progress;
- a compact sidebar readiness notice plus worker and inference controls;
- worker launch, handshake, model load, generation, streaming, cancellation,
  and shutdown;
- WAV, mode, identity, and target selection;
- UI controls materialized from the worker's `model_schema`;
- strict control-message, output-schema, result, and stream-frame validation;
  and
- bounded PCM sourcing, audio synchronization, and Shape Key value delivery.

The **Audio Playback** section is drawn immediately after the Selected WAV /
Stream selector. In Selected WAV mode it controls the completed result and its
audio. In Stream mode it reports streaming audio state and the applicable
volume and reset controls.

### Mesh subscriptions

**Add Selected Meshes** accepts every selected mesh object. It does not inspect
Shape Keys or require a particular channel set. At delivery time, Blender
iterates the negotiated channel list and requests a Shape Key with that exact
model-provided lowerCamel name from each enabled target's current Shape Key
datablock. Existing keys receive the frame values and absent keys are skipped.
There is no admission filter, stored name table, remapping, or bake path.

Objects sharing one Shape Key datablock are deduplicated for each frame. Every
linked object using that datablock reflects the assigned values; independent
motion requires single-user mesh data.

Worker I/O runs on standard-library threads. Those threads enqueue validated
messages and never mutate `bpy`. A registered Blender timer drains the queues
and performs all RNA and Shape Key updates on Blender's main thread.

## Managed worker

The native worker owns:

- CUDA device 0 and one shared CUDA stream;
- one Audio2Face-3D v3.0 diffusion/device-blendshape executor;
- one Audio2Emotion v3.0 classifier executor;
- selected WAV decoding, downmixing, and resampling;
- shared audio and emotion accumulators;
- the typed SDK parameter adapter;
- manual-emotion accumulation or direct Audio2Emotion-to-Audio2Face binding;
- in-place eye-look resolution into the model's output slots;
- atomic Selected WAV result publication; and
- incremental stream-frame publication.

stdin and stdout carry only UTF-8 JSON Lines using `audio2face/3`. Diagnostics
use stderr. Selected generation writes a complete result to the absolute
managed path submitted by Blender, then emits `result {}`. Stream requests send
bounded audio chunks and receive one timestamped weight row per `stream_frame`
event. Channel names are negotiated once in `load_model`; they are not repeated
in each live event.

## Managed installation

Runtime setup is extension-level state, so it appears only in Audio2Face's
Add-on Preferences. One NVIDIA terms link and one acceptance checkbox gate the
single artifact install. Audio2Face and Audio2Emotion source buttons expose the
model source pages; the models are delivered together rather than as separate
user downloads.

Each published platform record contains an immutable HTTPS URL, exact
compressed and unpacked sizes, and a SHA-256 digest. Installation:

1. takes an OS-held lock shared by Blender processes;
2. downloads the one artifact to temporary storage;
3. requires the final URL to remain credential-free HTTPS and verifies its
   byte count and digest;
4. extracts canonical paths under bounded member and total-size limits;
5. validates `bundle.json`, x86-64 executables, runtime libraries, both model
   input trees, licenses, and notices;
6. uses the bundled release-built TensorRT `trtexec` to build separate
   Audio2Face and Audio2Emotion `network.trt` engines for the local GPU;
7. writes the catalog receipt; and
8. atomically activates the completed platform directory.

Cancellation is honored before activation. A failed install cannot replace a
verified active runtime. The runtime root and every manifest member must stay
inside Blender's writable extension directory:

```python
bpy.utils.extension_path_user(__package__, path="", create=True)
```

The verified artifact supplies the worker, Audio2X, reviewed CUDA/TensorRT
user-mode libraries, `trtexec`, both model input trees, licenses, and notices.
The NVIDIA display driver remains a system requirement. No executable, SDK,
model, working directory, system installation, hosted service, or access
credential is selected by the user.

The checked-in catalog contains no artifacts. Installation therefore remains
disabled until release maintainers publish license-reviewed Linux x64 and
Windows x64 archives and enter their measured records. Release validation must
exercise the exact pinned model pair and confirm that Audio2Emotion's
post-processed vector order agrees with Audio2Face's emotion order; SDK 1.0.0
reports the vector width but does not expose names for those output positions.

Clean uninstall presents the same package-path confirmation style as Blender's
legacy add-on remover. The confirming Audio2Face operator returns before a
one-shot main-thread timer delegates to Blender 5.2's native extension
uninstaller, preventing the operator from unregistering its own class while it
is executing. The native disable phase runs Audio2Face's normal process,
stream, playback, timer, and handler cleanup; its package phase removes both
the installed extension and the Audio2Face leaf under Blender's writable
extension directory. That leaf owns the runtime, models, TensorRT engines,
logs, temporary install state, and generated results. User-selected audio,
`.blend` data, and shared GPU caches remain outside this ownership boundary.

## Lifecycle

```text
IDLE --Start Worker--> STARTING --hello--> LOADING_MODEL --> MODEL_READY
                                                            |
                              +-----------------------------+----------------+
                              |                                              |
                     Generate Selected WAV                              Start Stream
                              |                                              |
                         GENERATING                                     STREAMING
                         /    |    \                                   /   |    \
                    result canceled error                            end stop error
                       |      |      |                               |   |    |
                  COMPLETED READY  ERROR                           READY READY ERROR

Any live worker --Stop Worker--> STOPPING --> IDLE
Unexpected exit or rejected contract --> ERROR
```

**Start Worker** launches only a catalog-verified child, sends `hello {}`, and
loads the managed Audio2Face and Audio2Emotion models. One worker accepts at
most one generation or stream operation.

**Generate ARKit Values** reloads when the selected identity changes, freezes
the current model-described settings, and submits a complete WAV. **Cancel
Generation** interrupts active execution and prevents partial result
publication.

**Start WAV Stream** freezes the same settings and target subscriptions,
incrementally decodes and resamples the selected WAV, and sends bounded mono
f32le chunks. `stream_start` returns the exact model sample rate and
`prebuffer_samples`. The built-in source satisfies that lead before starting
audible playback, then samples frames against the audio-device clock.

The public [`audio2face.streaming`](../audio2face/streaming.py) API lets code
already running in Blender provide live model-rate mono f32le PCM. After
`start_pcm_stream(scene)`, the source polls
`get_pcm_stream_requirements(scene)` until it receives
`(sample_rate, prebuffer_samples)`, queues that initial lead, and owns audible
monitoring. No socket is opened.

**Stop Stream** cancels the active stream and keeps the models ready. **Stop
Worker** requests bounded shutdown and escalates to process termination if the
child misses its deadlines. Destruction releases executors, model metadata,
accumulators, and the CUDA stream in dependency order.

## Model schema and settings

`load_model` accepts only the two absolute managed model paths and a
non-negative identity index. It returns a positive `sample_rate` and one
`model_schema` with exactly:

- `identities`: ordered non-empty names from Audio2Face;
- `channels`: the exact 52 unique model-provided names in model order;
- `parameters`: opaque worker paths mapped to numeric SDK defaults; and
- `emotion_channels`: ordered `{name, default}` records from Audio2Face.

Blender owns no independent identity, emotion, output-channel, or parameter
catalog. It validates the schema and materializes RNA collections from it.
SDK 1.0.0 has no parameter reflection, so one typed worker adapter defines the
supported paths; Blender derives labels from those paths and owns no duplicate
parameter metadata. Internal nodes and tensors do not enter the schema.

Every generation and stream-start request contains exactly:

```json
{
  "auto_audio2emotion": false,
  "manual_emotions": {"<every advertised emotion name>": 0.0},
  "parameters": {"<every advertised worker path>": 0.0}
}
```

The worker rejects missing or extra emotion names and parameter paths,
incorrect numeric kinds, and non-finite values. Stream settings remain frozen
until that stream ends.

With `auto_audio2emotion` false, the ordered manual values form one constant
emotion vector for the operation. With it true, the Audio2Emotion executor
analyzes the same audio and its timestamped values fully override the manual
vector. The worker does not blend the two modes. The rule is identical for
Selected WAV and Stream.

## ARKit output and playback

The output preserves the model's reported 52-channel ARKit order. No Python
channel list or reorder table defines it. The worker validates a unique
52-name skin output and locates the eight eye-look semantics by name so six SDK
eye-rotation components can update those slots in place. Raw geometry, jaw
transforms, eye rotations, and other solver outputs do not leave the worker.

Every Selected WAV `a2f-animation/2` document has exactly six fields:

- `schema`
- `job_id`
- `sample_rate`
- `channels`
- `timestamps_samples`
- `weights`

The channel array is the negotiated model order. Every row has exactly 52
finite values in `[0.0, 1.0]`. Timestamps are non-empty, strictly increasing
signed 64-bit audio-sample positions, and row count equals timestamp count.
Live frames use the same negotiated channel order.

Selected playback verifies the managed result and submitted WAV identity,
linearly samples frames against Blender audio position, and delivers values to
current target Shape Keys. Play, pause/resume, stop, loop, volume, and
reset-on-stop are Blender-local and do not invoke inference.

The streamed-WAV path uses a bounded frame buffer and Blender audio position.
A live PCM source uses a monotonic presentation clock anchored to the first
returned timestamp. `stream_end` and explicit stream stop never create a
result file.

## Failure boundaries

The extension accepts only the exact worker profile
`nvidia-a2f3-a2e3-gpu-arkit52/2` and a non-empty worker version. Control records
reject missing or unknown fields, duplicate JSON keys, non-finite numbers,
invalid IDs, unknown methods or events, malformed UTF-8, and payloads over
1 MiB.

PCM chunks contain non-empty finite little-endian float32 samples, cover at
most one model-rate second, and are limited to 256 KiB before base64 encoding.
Blender permits at most 64 pending chunk acknowledgements, providing bounded
backpressure. Stream IDs, timestamps, sample rates, and all coefficient rows
are validated before Blender state changes.

Result files stay inside the managed results directory, are limited to
512 MiB, and are published atomically without replacing an existing file.
Installer and worker failures appear in Blender status. Unexpected child exit
clears model and stream state and reports the latest worker diagnostic.

Unregistering the extension stops result and live playback, cancels its WAV
source and installer, unregisters the timer, and closes the worker before
removing Blender classes and scene RNA.
