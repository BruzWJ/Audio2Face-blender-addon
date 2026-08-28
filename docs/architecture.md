# Architecture

## System boundary

Audio2Face has one offline Blender-frame bake path and one live stream path:

```text
Selected WAV -- add-on-owned VSE strip --> native timeline/audio clock
       |                                      |
       +-- Play: lazy PCM -- live stream -----+-----------------------+
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
models, creates their interactive Audio2Face and Audio2Emotion executors, and
keeps those executors available for later audio operations. **Stop Worker**
exits the child and releases the executors, models, and CUDA resources.
Installing or enabling the extension does not start the worker. Loaded
executors do not execute audio inference continuously without PCM input.

## Blender extension

The extension owns:

- Add-on Preferences for NVIDIA terms, model-source links, two persistent
  external model-root selections, model optimization, cancellation, and logs;
- worker launch, handshake, model loading, streaming, asynchronous baking,
  cancellation, and shutdown;
- one add-on-owned Selected WAV Sequencer strip, native timeline Play/Pause,
  lazy Selected WAV PCM streaming, native Shape Key Action output, and
  external PCM ingress;
- Audio2Face and emotion controls and the registered target-object list;
- strict protocol, model-schema, stream-frame, and bake-frame response
  validation; and
- Action construction and clocked live delivery on Blender's main thread.

Selected WAV playback and external live PCM advance through the stream
protocol's per-chunk dequeue credits. Selected bake upload and serial frame
requests advance asynchronously through worker responses. A registered Blender
timer advances these operations; frame evaluation, RNA reads, Action creation,
and live Shape Key updates stay on the main thread.

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
strip at `scene.frame_start`; unrelated Sequencer strips are preserved. Its
duration is `ceil(seconds * fps / fps_base)` frames. The inclusive audio end is
`frame_start + duration_frames - 1`, and `scene.frame_end` is extended to that
frame when necessary but never shortened.

**Play** invokes Blender's native timeline operator and lazily decodes,
downmixes, and resamples the WAV into model-rate mono f32le. That PCM travels
through the existing stream protocol. Blender's native timeline and sound
playback are the presentation clock; incoming stream frames directly update
matching Shape Key values as transient state without creating an Action.
**Pause** invokes only Blender's native playback pause operator. A new preview
starts at `scene.frame_start`; Play after Pause resumes the active stream.

**Bake Shape Key Animation** is a separate persistence operation. Blender
asynchronously decodes/downmixes/resamples the entire selected WAV to model-rate
mono f32le, uploads it in bounded chunks, and prepares an offline track. It then
visits every integer Blender frame in the inclusive audio range. For each frame
it:

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

### Interactive worker execution

Model loading during **Start Worker** creates one interactive Audio2Emotion
executor, one interactive Audio2Face/device-blendshape executor, and their
shared accumulators on CUDA device 0. The worker retains them until **Stop
Worker**. Selected WAV playback, external Stream input, and bake use those same
executors; there is no incremental executor path or mid-stream executor switch.

For a Stream, the worker retains a bounded, frame-aligned PCM window. It refills
the accumulators for that window, computes its effective-emotion curve, and
calls the Audio2Face executor's targeted `ComputeFrame` only for safe frame
indices that have not already been published. Window-local timestamps are
translated to one strictly increasing operation timeline; lookahead frames wait
for more PCM, and normal end-of-input releases the tail.

For a bake, the worker fills the accumulators from the complete uploaded PCM.
Each requested Blender frame supplies its evaluated settings and target sample;
the worker calls `ComputeFrame` for the one or two source frames bracketing that
sample and linearly interpolates their weights. This avoids computing and
caching a complete face track for every settings snapshot.

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
installs snapshots at ordered queue boundaries on the same interactive
executors and only unpublished frames use the replacement. A bake sends the
RNA-evaluated snapshot for each requested Blender frame, so keyframes and
drivers can shape the generated curves without the add-on keyframing those
controls.

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

MODEL_READY --Selected WAV Play--> STREAMING --end/cancel--> MODEL_READY
                                          \--error----------> ERROR

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
