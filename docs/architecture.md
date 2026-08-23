# Architecture

## System boundary

Audio2Face has two audio sources and one incremental GPU inference path:

```text
Selected WAV -- Play/Pause/seek/loop -- WAV decoder/resampler --+
                                                               |
external mono f32le PCM -- first chunk auto-starts ------------+-- stream
                                                                   |
Blender 5.2 extension -- private audio2face/5 JSONL -- native worker
                                                                   |
                             +-------------------------------------+------+
                             |                                            |
                    Audio2Emotion v3.0                         Audio2Face-3D v3.0
                             |                                            |
                    emotion accumulator                    raw face geometry
                             |                                            |
                             +------ default-model GPU ARKit solver ------+
                                                   |
                                  52 named scalar coefficient frames
                                                   |
                       Blender main-thread Shape Key delivery to targets
```

Blender owns the package-local worker as a child process. The worker opens no
port, and CUDA, TensorRT, Audio2X, model metadata, and inference executors stay
outside Blender's process. **Start Worker** launches the child and loads the two
models. **Stop Worker** exits it and releases model and CUDA resources.
Installing or enabling the extension does not start the worker. Loading the
models does not execute audio inference continuously.

Both audio modes use the same `stream_start` / `stream_chunk` /
`stream_settings` / `stream_end` worker operation and receive timestamped ARKit
coefficient frames.

## Blender extension

The extension owns:

- Add-on Preferences for NVIDIA terms, model-source links, two persistent
  external model-root selections, model optimization, cancellation, and logs;
- worker launch, handshake, model loading, streaming, cancellation, and
  shutdown;
- Selected WAV playback, seeking, looping, and external PCM ingress;
- Audio2Face and emotion controls and the registered target-mesh list;
- strict protocol, model-schema, and stream-frame validation; and
- audio-clocked delivery of coefficient values to Shape Keys on Blender's main
  thread.

Worker I/O and WAV decoding run on standard-library threads. Those threads
enqueue validated data and never mutate `bpy`. A registered Blender timer
drains the queues and performs RNA and Shape Key updates on the main thread.

### Mesh targets

**Add Selected Meshes** accepts any selected mesh object without inspecting
Shape Keys or topology. At delivery time, Blender iterates the model's
negotiated channel list and requests a Shape Key with each exact model-provided
name from every listed target. Existing keys receive the frame values and
absent keys are skipped.

List membership is the complete delivery state and is resolved again for every
delivered frame; target subscriptions are not cached. Adding or removing a mesh
during playback therefore affects the next delivered frame. An empty list is a
valid no-subscriber state: inference and audio continue, and a later-added mesh
receives the next frame.

Objects sharing one Shape Key datablock are deduplicated for each frame. Every
linked object using that datablock reflects the assigned values; independent
motion requires single-user mesh data.

## Geometry-to-ARKit solve

The Blender target does not need to share the Audio2Face reference mesh's
topology. The default Audio2Face-3D v3.0 repository contains the solver's own
identity-specific 24,002-vertex neutral basis and 52 pose bases. At model load,
the worker selects the default identity at SDK index `0` and creates NVIDIA's
GPU device-blendshape solver from that model-owned data.

During inference, Audio2Face produces raw geometry for its internal identity.
The worker passes that geometry directly to the model-owned blendshape solver,
which returns 52 scalar weights in the model's pose-name order. Six SDK eye
rotation components are resolved into the eight corresponding eye-look
channels in that same list. Only timestamped scalar weights cross the JSONL
boundary. Raw geometry, the 24,002-vertex basis, jaw transforms, and eye
rotations never enter Blender.

Consequently, a target mesh may have different vertex count and topology and
may contain all, some, or none of the reported Shape Keys. Topology affects
only how the artist authored those local Shape Keys; it is not an input to the
Audio2Face-to-ARKit solve.

## Audio lifecycles

### Selected WAV

Pressing **Play** is the inference trigger. Blender validates the WAV, creates
an internal stream operation, incrementally decodes/downmixes/resamples the
file to the model rate, and sends bounded mono f32le chunks. The worker reports
`prebuffer_samples`; the source queues that input lead before audible playback
begins. Returned frames are buffered and sampled against Blender's audio-device
position rather than scene FPS.

The one stateful control changes between **Play** and **Pause**. Pause freezes
both audible audio and WAV source pacing. The duration-based seek control
cancels the current stream and restarts at the requested audio position while
retaining the loaded worker. **Loop** performs the same restart at zero after
natural end. **Prediction Delay** offsets frame sampling relative to the
audible clock; it does not delay audio or keep inference running while paused.

### External PCM

Code already running in Blender uses
[`audio2face.streaming`](../audio2face/streaming.py):

1. `get_pcm_stream_requirements(scene)` reports the loaded model rate and
   `None` before a stream has been accepted.
2. The first `push_audio_f32le(payload, scene_name=...)` queues the exact bytes
   and automatically submits the internal `stream_start` request on Blender's
   main-thread poll.
3. Once accepted, requirements report `(sample_rate, prebuffer_samples)` and
   queued chunks are flushed in order.
4. `end_pcm_stream(scene_name=...)` marks normal end-of-input after every
   previously queued chunk.

The ingress FIFO and pending request count are bounded. External integrations
own capture, resampling, and audible monitoring. They do not start or stop the
worker and no network listener is created.

## Model schema and inference settings

`load_model` receives only the two validated absolute top-level `model.json`
paths. It returns a positive sample rate and one exact `model_schema`:

- `channels`: 52 unique non-empty model-provided names in model order;
- `emotion_channels`: ordered model-provided `{name, default}` records; and
- `audio2face_defaults`: the exact model-reported 18-field Audio2Face settings
  object.

Blender has no independent output-channel list, identity selector, model
variant selector, graph-node controls, or tensor controls. It builds the manual
emotion collection and target delivery from the loaded default model schema,
and seeds all Audio2Face controls from the returned defaults.

Every `stream_start` installs one complete settings snapshot containing the
exact 18-field `audio2face` object, `auto_audio2emotion`, every advertised
`manual_emotions` value, and the seven-field `audio2emotion` object. The field
names, types, and ranges are defined by the
[protocol](protocol.md#settings-document). `stream_settings` replaces that
whole snapshot at an ordered boundary; partial setting updates do not exist.

With `auto_audio2emotion` false, the manual values form a constant emotion
driver. With it true, Audio2Emotion analyzes the same PCM and NVIDIA's
post-processor applies strength, contrast, retained-emotion count, temporal
blend, transition smoothing, and optional preferred-emotion mixing.
**Preferred Emotion > Load** copies the current manual values into a distinct
snapshot; later manual edits do not mutate it. **Clear** restores `null`. Auto
Audio2Emotion and the preferred snapshot are independent controls. Both
Selected WAV and external PCM apply control edits to their current operation by
resetting the two executors and accumulators and replaying a bounded PCM
context; audio transport is not restarted or discarded.

## Runtime and model ownership

Each release is one complete platform extension ZIP:

- Windows x64: PE worker and `trtexec` plus DLLs in `runtime/bin`;
- Linux x64: ELF worker and `trtexec` in `runtime/bin`, with shared objects in
  `runtime/lib`.

The CUDA backend requires a supported NVIDIA GPU and driver. Native formats
are not interchangeable; there is no macOS or ARM package. The add-on validates
`runtime/bundle.json`, native formats, package confinement, and runtime files
before launch. It never searches the host for another worker, CUDA Toolkit,
TensorRT SDK, Audio2Face installation, or executable.

The runtime contains no model files or TensorRT engines. Users select the exact
roots of complete Audio2Face-3D v3.0 and Audio2Emotion v3.0 repositories in
Add-on Preferences. **Optimize Models** validates each root and uses the
bundled `trtexec` to create its GPU-specific `network.trt`. Those external roots
and engines remain user-owned and are not removed when Blender uninstalls the
extension.

Release builds use the pinned inputs in
[`worker/runtime-lock.json`](../worker/runtime-lock.json). The native builder
writes `build/runtime/<platform>`; the extension builder validates that
handoff, embeds it in the temporary Blender package root, pins the manifest to
one platform, validates the source and archive with Blender 5.2, and writes the
platform ZIP. See the [worker build guide](../worker/README.md) for the native
runtime contract.

## Lifecycle

```text
IDLE --Start Worker--> STARTING --hello--> LOADING_MODEL --> MODEL_READY
                                                            |
                      +-------------------------------------+----------------+
                      |                                                      |
             Selected WAV Play                                  first external PCM
                      |                                                      |
                      +--------------------> STREAMING <---------------------+
                                                |
                         natural end / cancel / seek / loop / error
                                                |
                                  MODEL_READY or ERROR

Any live worker --Stop Worker--> STOPPING --> IDLE
Unexpected exit or rejected contract --> ERROR
```

One worker accepts at most one stream. Selected seek and loop use
cancel/restart inside the same worker lifecycle. A normal external end drains
tail frames. Cancel stops queued execution without draining. **Stop Worker**
requests bounded shutdown and escalates to process termination if the child
misses its deadlines.

## Failure boundaries

The extension accepts only protocol `audio2face/5`, worker profile
`nvidia-a2f3-a2e3-gpu-arkit52/5`, and a non-empty worker version. Envelopes
reject missing or unknown fields, duplicate JSON keys, non-finite numbers,
invalid IDs, unknown methods or events, malformed UTF-8, and payloads over
1 MiB. Malformed or misrouted output is a terminal contract violation: Blender
clears active stream presentation and closes that worker before later messages
can mutate scene state.

Each PCM chunk is non-empty, finite little-endian float32 mono at the model
rate and covers at most one second. The worker bounds its queued PCM to four
seconds, and Blender also bounds pending acknowledgements and pre-start ingress.
Operation IDs, timestamps, sample rates, channel layouts, and every coefficient
row are validated before Blender state changes.

Unregistering the extension stops audio presentation, cancels WAV sourcing and
model optimization, unregisters its timer, and closes the worker before
removing Blender classes and scene RNA.
