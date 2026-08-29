# Architecture

## System boundary

Audio2Face has one offline Blender-frame bake path and one live stream path:

```text
Selected WAV -- add-on-owned VSE strip --> native timeline/audio clock
       |                                      |
       +-- native Play: finite PCM stream ----+-----------------------+
       |                                                              |
       +-- Bake: PCM upload -- per-integer-frame requests ------------+
                                                                      |
external mono f32le PCM -- live stream -------------------------------+
                                                                      |
Blender 5.2 extension -- private audio2face/10 JSONL -- native worker |
                                                                      |
                     +-----------------------------------------------+------+
                     |                                                      |
            Audio2Emotion v3.0                                   Audio2Face-3D v3.0
                     |                                                      |
            effective emotions                                   raw face geometry
                     |                                                      |
                     +------------ default-model GPU ARKit solver ----------+
                                                   |
                    52 coefficients (+ live effective emotions)
                                                   |
                  +--------------------------------+-------------------+
                  |                                                    |
       Blender 5.2 Shape Key Actions               transient live Shape Key values
                                                                       |
                                                      read-only Mixed Emotion display
```

Blender owns the package-local worker as a child process. The worker opens no
port, and CUDA, TensorRT, Audio2X, model metadata, and inference executors stay
outside Blender's process. **Start Worker** launches the child, loads the two
models, and creates the regular Audio2Face and Audio2Emotion streaming
executors. **Stop Worker** exits the child and releases the executors, models,
and CUDA resources.
Installing or enabling the extension does not start the worker. Loaded
executors do not execute audio inference continuously without PCM input.

## Blender extension

The extension owns:

- Add-on Preferences for NVIDIA terms, model-source links, two persistent
  external model-root selections, model optimization, cancellation, and logs;
- worker launch, handshake, model loading, streaming, asynchronous baking,
  cancellation, and shutdown;
- one add-on-owned Selected WAV Sequencer strip positioned by **First Frame**,
  Blender-native playback handlers, finite Selected WAV PCM streaming, native
  Shape Key Action output, and external PCM ingress;
- Audio2Face and emotion controls and the registered target-object list;
- strict protocol, model-schema, stream-frame, and bake-frame response
  validation; and
- Action construction and clocked live delivery on Blender's main thread.

Selected WAV playback and external live PCM advance through the stream
protocol's per-chunk dequeue credits. Finite Selected WAV input is sent as fast
as credits permit and its complete output is retained for frame lookup;
external PCM keeps a bounded rolling presentation buffer. Selected bake upload
and serial frame requests advance asynchronously through worker responses. A
registered Blender timer advances these operations; native playback and
frame-change handlers make Blender's current frame authoritative. RNA reads,
Action creation, and Shape Key updates stay on the main thread.

### Shape Key targets

**Add Selected Objects** accepts Blender Mesh, Curve, Surface, and
Lattice objects without requiring existing Shape Keys. These are the exact
object types on which Blender 5.2 supports Shape Keys; Text, Hair Curves, Point
Cloud, Grease Pencil, and other object types are excluded. At delivery time,
Blender iterates the model's
negotiated channel list and requests a Shape Key with each exact model-provided
name from every listed target. Existing keys receive the frame values and
absent keys are skipped.

For a live Stream, list membership is the complete delivery state and is
resolved again for every delivered frame; target subscriptions are not cached.
Adding or removing an object affects the next frame. An empty list is a valid
no-subscriber state: inference continues and a later-added object receives the
next frame.

A Selected WAV bake instead preflights the targets before upload. At least one
target must contain a Shape Key matching a negotiated channel, and only
existing matches receive curves.

Objects sharing one Shape Key datablock are deduplicated for live assignment
and for Action output. A successful bake writes LINEAR F-curves into the active
native Action for each unique compatible Key datablock, or creates and assigns
one when none exists. Matching keys inside the baked span are replaced while
other animation remains. Every linked object reflects the same data;
independent motion requires single-user object data.

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
channels in that same list. Only scalar weights, plus timestamps and effective
emotions for live frames, cross the JSONL boundary. Raw geometry, the
24,002-vertex basis, jaw transforms, and eye rotations never enter Blender.

Consequently, a target object may use geometry unrelated to the Audio2Face
reference mesh and may contain all, some, or none of the reported Shape Keys.
Its local geometry affects only how the artist authored those Shape Keys; it is
not an input to the Audio2Face-to-ARKit solve.

## Audio lifecycles

### Selected WAV

Selecting a valid WAV creates or updates one explicitly add-on-owned VSE sound
strip at the saved **First Frame**; unrelated Sequencer strips are preserved.
Its duration is `ceil(seconds * fps / fps_base)` frames and its inclusive end is
`First Frame + duration_frames - 1`. The add-on never changes the scene or
preview playback range.

The sidebar **Play/Pause** operator is a Blender-native transport toggle; the
Timeline, Spacebar, and other editors enter the same playback lifecycle. On a
cache miss the add-on decodes, downmixes, and resamples the WAV to model-rate
mono f32le, then feeds the finite input through the stream protocol. If the
requested frame is unavailable, native playback pauses and resumes after that
pose is presented, keeping sound and Shape Keys together. Stopping playback
freezes presentation without discarding inferred frames.

`frame_change_post` maps `(scene.frame_current - First Frame)` through effective
FPS and Prediction Delay, then applies one cached result after Blender has
evaluated that frame's RNA. Frames outside the sound interval are neutral.
Selected Audio never evicts its earlier frames, and worker EOF releases the GPU
operation without releasing the Blender presentation cache. Rewind, scrubbing,
repeated Play, and Blender scene/preview-range wrapping therefore reuse exact
frame lookups with no second audio clock. Blender owns native sound seeking and
loop mode; the add-on has no loop or seconds-position state.
If the native scene or preview range ends before the sound, Blender intentionally
wraps both sources there; the user extends that native range to play the full
clip.

**Bake Shape Key Animation** is a separate persistence operation. Blender
asynchronously decodes/downmixes/resamples the entire selected WAV to model-rate
mono f32le, uploads it in bounded chunks, and prepares an offline track. It then
visits every integer Blender frame from **First Frame** through the inclusive
audio end. Starting a bake pauses native playback and discards the transient
preview cache so the baked Action is the sole animation authority. For each
frame it:

1. calls `scene.frame_set(frame)` so keyframes and drivers evaluate;
2. maps the frame through effective FPS (`fps / fps_base`) and the evaluated
   Prediction Delay to the nearest in-range audio sample;
3. reads a complete inference-settings snapshot from scene RNA; and
4. requests and validates one weight row before advancing.

The controller restores the original scene frame when the bake ends or is
canceled. Only after `bake_ended {"reason":"completed"}` does it write the
native Shape Key Actions. It reads animated tuning and emotion values but does
not keyframe those controls; only matching Shape Key `value` curves are output.

### External PCM

[`audio2face.streaming`](../audio2face/streaming.py) is the Blender-side ingress
boundary. Producers supply model-rate mono f32le PCM and own capture,
resampling, and audible monitoring. The main-thread poll starts the operation,
flushes the bounded FIFO in order, and applies worker credits as backpressure.
No network listener is created.

### Native executor lifecycles

Model loading during **Start Worker** creates regular Audio2Emotion and
Audio2Face/device-blendshape executors on CUDA device 0. Selected WAV playback
and external Stream input append each PCM chunk once. A face-first,
emotion-second readiness loop executes whichever retained regular executor can
advance, publishes strictly increasing timestamps, and drops only input that
both consumers have processed. Normal end-of-input closes the accumulators and
drains their tail.

Interactive executors are created only for bake. Before bake begins, the worker
destroys the regular family; this prevents two copies of the Audio2Face and
Audio2Emotion TensorRT contexts from occupying GPU memory. For a bake, the
worker fills the interactive accumulators from the complete uploaded PCM.
Each requested Blender frame supplies its evaluated settings and target sample;
the worker calls `ComputeFrame` for the one or two source frames bracketing that
sample and linearly interpolates their weights. This avoids computing and
caching a complete face track for every settings snapshot. Bake completion
destroys the interactive family and recreates the regular streaming family.

## Model schema and inference settings

`load_model` receives only the two validated absolute top-level `model.json`
paths. It returns a positive sample rate and one exact `model_schema`:

- `channels`: 52 unique non-empty model-provided names in model order;
- `emotion_channels`: ordered model-provided `{name, default}` records; and
- `audio2face_defaults`: the exact model-reported 18-field Audio2Face settings
  object.

Blender has no independent output-channel list, identity selector, model
variant selector, graph-node controls, or tensor controls. It builds the saved
Preferred Emotion and transient Mixed Emotion collections, plus target
delivery, from the loaded default model schema, and seeds all Audio2Face
controls from the returned defaults.

Each operation uses a complete `audio2face` and compositional `emotion_driver`
snapshot defined by the [protocol](protocol.md#settings-document). A Stream
installs its initial snapshot before regular execution. The SDK does not permit
regular executor setters after execution starts, so an ordered settings change
resets only the native executor state, replays its bounded retained context,
and suppresses timestamps that were already published. It does not restart the
worker, emit a presentation reset, or replace Blender's buffered frames. A bake
sends the RNA-evaluated snapshot for each requested Blender frame, so keyframes
and drivers can shape the generated curves without the add-on keyframing those
controls. Once finite Selected Audio reaches worker EOF, a later
inference-setting edit discards its cache so the next native Play creates a
fresh one.

Preferred and Mixed Emotion have disjoint ownership. Blender owns the saved,
animatable Preferred values and represents all-zero values as an absent source.
The worker owns transient Mixed output and returns it aligned with each live
ARKit frame. Delivery cannot feed Mixed values back into inference or mutate
Preferred values.

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

MODEL_READY --Selected WAV Bake--> BAKE_UPLOADING --> BAKE_PREPARING
                                                      |
                                                   BAKING
                                                      |
                                                 BAKE_ENDING
                                                      |
                                             MODEL_READY or ERROR

MODEL_READY --native Play/cache miss--> STREAMING --worker EOF--> MODEL_READY
                                                \--error----------> ERROR
MODEL_READY + Selected cache --native Play/Pause/seek/wrap--> MODEL_READY + cache

MODEL_READY --first external PCM--> STREAMING --end/cancel--> MODEL_READY
                                           \--error----------> ERROR

Any live worker --Stop Worker--> STOPPING --> IDLE
Unexpected exit or rejected contract --> ERROR
```

One worker accepts at most one inference operation: one Stream or one bake. A
normal stream end drains tail frames. Cancel stops queued execution without
draining and returns the model to ready state. **Stop Worker** requests bounded
shutdown and escalates to process termination if the child misses its
deadlines.

## Failure boundaries

The extension accepts only the expected protocol identity and validates every
message before Blender state changes. Malformed or misrouted output is a
terminal contract violation: Blender clears the active operation and its
presentation state, then closes that worker before later messages can mutate
scene state. Exact envelope, payload, ordering, and queue limits are in the
[protocol](protocol.md).

Unregistering the extension cancels active inference and WAV decoding, stops
model optimization, unregisters its timer, and closes the worker before
removing Blender classes and scene RNA. Blender owns native timeline playback
and Sequencer data.
