# Architecture

## System boundary

Audio2Face has one cached Selected WAV track path and one sequential
external Stream path:

```text
Selected WAV -- add-on-owned VSE strip --> native timeline/audio clock
       |
       +-- decode/resample/upload once --> persistent interactive track
                                                  |
                                      ComputeAllFrames (A2E, then A2F)
                                                  |
                                      atomic timestamped cache
                                         ^                 ^
native Blender frame -- target sample ---+                 +--- Bake

external mono f32le PCM --> sequential Stream executors

Blender 5.2 extension -- private audio2face/12 JSONL -- native worker
                                                            |
                                 +--------------------------+------------------+
                                 |                                             |
                        Audio2Emotion v3.0                          Audio2Face-3D v3.0
                                 |                                             |
                        effective emotions                          raw face geometry
                                 |                                             |
                                 +--------- default-model GPU ARKit solver ----+
                                                            |
                             52 coefficients (+ effective emotions)
                                                            |
                             +------------------------------+------------------+
                             |                                                 |
                  Blender Shape Key Actions                    transient Shape Key values
                                                                                |
                                                               read-only Mixed Emotion
```

Blender owns the package-local worker as a child process. The worker opens no
port, and CUDA, TensorRT, Audio2X, model metadata, and inference executors stay
outside Blender's process. **Start Worker** launches the child, loads the two
models, and makes the backend ready for an audio operation. A Selected WAV
creates a persistent interactive track when both model and source are ready; a
true Stream creates its sequential executors when external PCM begins. **Stop
Worker** exits the child and releases the active operation, models, and CUDA
resources. Installing or enabling the extension does not start the worker.

Native media and native inference have independent lifecycles. Worker and track
lifecycle changes never start, pause, resume, seek, or loop Blender media.
Blender transport never starts, pauses, resumes, or ends the worker or prepared
track.

## Blender extension

The extension owns:

- Add-on Preferences for NVIDIA terms, model-source links, two persistent
  external model-root selections, model optimization, cancellation, and logs;
- worker launch, handshake, model loading, streaming, asynchronous baking,
  cancellation, and shutdown;
- one add-on-owned Selected WAV Sequencer strip positioned by **First Frame**,
  one-time Selected WAV track upload, `frame_change_post` cache sampling,
  native Shape Key Action output, and external PCM ingress;
- Audio2Face and emotion controls and the registered target-object list;
- strict protocol, model-schema, stream-frame, and track-render response
  validation; and
- Action construction and clocked live delivery on Blender's main thread.

Selected WAV PCM is uploaded once. A settings revision runs Audio2Emotion and
Audio2Face continuously over the complete track, then publishes bounded frame
batches atomically. Native frame changes perform no protocol or GPU work; they
sample the published timestamps on Blender's main thread. Bake samples that
same cache. External PCM alone advances through the sequential Stream protocol's
per-chunk dequeue credits and bounded rolling presentation buffer. A registered
timer advances protocol work, and `frame_change_post` makes Blender's current
frame authoritative without an add-on playback handler.

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

A Selected WAV bake preflights the targets before frame iteration. At least one
target must contain a Shape Key matching a negotiated channel, and only
existing matches receive curves.

Objects sharing one Shape Key datablock are deduplicated for live assignment
and Action output. A successful bake writes LINEAR F-curves into an Action
tagged as owned by the add-on. It reuses the active Action only when that tag is
present; otherwise it creates and assigns a new Action without changing the
artist's prior Action. Re-baking replaces every curve in the owned Action so
stale channels and modifiers cannot alter playback.

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
strip at the saved **First Frame**; unrelated strips are preserved. Blender
owns the strip's native duration and updates it when scene FPS changes. The
add-on does not calculate or override `SoundStrip.duration`, and it does not
change scene or preview playback ranges. It selects Blender's `AUDIO_SYNC`
mode so playback drops delayed viewport frames to remain synchronized to sound.

The Timeline, Spacebar, and other native Blender transports enter the same
media lifecycle. They do not enter or leave a worker operation. As soon as a
valid source and loaded models are both ready, the add-on decodes, downmixes,
and resamples the WAV to model-rate mono f32le, uploads it once through the
`track_*` protocol, and retains the prepared interactive track. Source
replacement or worker shutdown ends that track; play, pause, seek, and loop do
not.

`frame_change_post` maps `(scene.frame_current - strip.content_start)` through
effective FPS and Prediction Delay to a target sample, then linearly samples
the latest complete cache. It sends no worker request. Frames outside the
native sound span are neutral.

A tuning or emotion edit sends a newer `track_render` revision. The worker
interrupts an older revision without canceling the track, runs continuous
Audio2Emotion and Audio2Face passes, and publishes the replacement only after
all frames are complete. The requested current-frame sample is emitted from
that completed continuous result before its cache batches. There is no
inference-side playback clock: rewind, scrub, repeated play, pause, and loop
only change Blender's native frame and sound transport.
If the native scene or preview range ends before the sound, Blender intentionally
wraps both sources there; the user extends that native range to play the full
clip.

**Bake Shape Key Animation** is a separate persistence operation. It snapshots
the current settings, waits for the matching continuous cache if necessary,
then maps every integer Blender frame in the native strip span to a cache sample
using `fps / fps_base` and Prediction Delay. Preview and bake therefore consume
identical values. It writes only matching Shape Key `value` curves without
changing Blender's current frame; it does not run `ComputeFrame`, upload audio
again, or create a second executor family.

### External PCM

[`audio2face.streaming`](../audio2face/streaming.py) is the Blender-side ingress
boundary. Producers supply model-rate mono f32le PCM and own capture,
resampling, and audible monitoring. The main-thread poll starts the operation,
flushes the bounded FIFO in order, and applies worker credits as backpressure.
No network listener is created. This is the only sequential audio mode; it does
not use the Selected WAV track or Blender's media transport.

### Native executor lifecycles

Model loading during **Start Worker** initializes the model-owned CUDA resources
on device 0 and makes the backend ready to create the executor family required
by the chosen audio mode.

A Selected WAV `track_start` creates one interactive Audio2Emotion and
Audio2Face/device-blendshape family. `track_chunk` uploads complete PCM once and
`track_prepare` closes its audio accumulator. Each `track_render` installs a
complete settings snapshot. A changed emotion-driver snapshot uses
Audio2Emotion `ComputeAllFrames` to replace the timestamped emotion accumulator;
an unchanged snapshot reuses that exact curve. Audio2Face then uses
`ComputeAllFrames` to produce temporally regularized weights. Results are
emitted in batches of at most 64 frames and published atomically by Blender.
Stateless `ComputeFrame` is never used; the current-frame preview, full cache,
and bake all share the continuous result.

A true Stream owns the regular sequential Audio2Emotion and
Audio2Face/device-blendshape family. Each PCM chunk is appended once. A
face-first, emotion-second readiness loop advances available work, publishes
strictly increasing timestamps, and drops only inputs consumed by both models.
`stream_settings` applies validated setters between queued chunks. It does not
reset executors, retain or replay PCM, destroy the family, or restart media.

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

Each operation uses the complete `audio2face` and compositional
`emotion_driver` snapshot defined by the [protocol](protocol.md#settings-document).
A Selected edit revises the continuous cache without changing track or media
lifecycle. A Stream edit changes live executor parameters between chunks
without resetting inference or replaying audio.

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

MODEL_READY + Selected WAV --> TRACK_UPLOADING --> TRACK_PREPARING --> MODEL_READY
MODEL_READY + tuning change --> TRACK_PREPARING --> MODEL_READY
MODEL_READY + native frame/transport change -----------------------> MODEL_READY
MODEL_READY + Bake -------------------------------> BAKING --------> MODEL_READY

MODEL_READY --first external PCM--> STREAMING --end/cancel--> MODEL_READY
                                           \--error----------> ERROR

Any live worker --Stop Worker--> STOPPING --> IDLE
Unexpected exit or rejected contract --> ERROR
```

One worker accepts at most one audio operation: one persistent Selected WAV
track or one sequential Stream. Bake is Blender-side iteration over the
existing track, not another uploaded audio operation. Native playback does not
participate in this state machine. A normal Stream end drains tail frames;
cancel stops queued Stream execution without draining and returns the model to
ready state. **Stop Worker** requests bounded shutdown and escalates to process
termination if the child misses its deadlines.

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
