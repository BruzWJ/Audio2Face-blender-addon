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
package-local native worker on CUDA device 0
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
generation operation or active PCM stream executes the models.

The only animation output is timestamped coefficient frames. The extension
does not write geometry, vertices, bones, Actions, F-curves, or baked
animation. Selected WAV buffers a complete result. Stream delivers frames
incrementally and creates no result file.

## Blender extension

The extension owns:

- Add-on Preferences for one NVIDIA terms acceptance, model-source links, two
  persistent external repository-root selections, model optimization,
  cancellation, and progress;
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
model-provided name from each enabled target's current Shape Key
datablock. Existing keys receive the frame values and absent keys are skipped.
There is no admission filter, stored name table, remapping, or bake path.

Objects sharing one Shape Key datablock are deduplicated for each frame. Every
linked object using that datablock reflects the assigned values; independent
motion requires single-user mesh data.

Worker I/O runs on standard-library threads. Those threads enqueue validated
messages and never mutate `bpy`. A registered Blender timer drains the queues
and performs all RNA and Shape Key updates on Blender's main thread.

## Package-local worker

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
add-on-owned path submitted by Blender, then emits `result {}`. Stream requests send
bounded audio chunks and receive one timestamped weight row per `stream_frame`
event. Channel names are negotiated once in `load_model`; they are not repeated
in each live event.

## Bundled runtime and model optimization

Setup is extension-level state, so it appears only in Audio2Face's Add-on
Preferences. The page links to the NVIDIA Audio2Face-3D v3.0 and gated
Audio2Emotion v3.0 repositories. Users download and extract both complete
repositories, then use two persistent Blender `DIR_PATH` properties to select
their exact clone or download roots anywhere on disk. For each selected root,
the add-on derives only `<root>/model.json`. It does not authenticate with a
model host and never downloads, copies, relocates, or deletes either model
root.

NVIDIA provides the Audio2Face-3D SDK source and ONNX-based model inputs. It
does not provide the native Blender child described by this architecture. A
release therefore consists of two self-contained Blender extension ZIPs:

- Windows x64, containing the project worker, Audio2X, the exact Windows
  CUDA/TensorRT user-mode dependency closure, TensorRT's pinned `trtexec`, and
  required notices; and
- Linux x64, containing the corresponding ELF executables, shared objects,
  and required notices.

Each extension contains its one native runtime directly at
`audio2face/runtime/`. The runtime is package-local content, not a separately
installed component. On Linux, Blender's ZIP installer drops executable mode
bits, so the resolver validates the two exact ELF files and restores only
their owner execute bit before launch. It contains no model files or serialized
TensorRT engines. Windows PE files and Linux ELF files are not
interchangeable, so there is no single cross-platform extension ZIP. The
CUDA-only backend requires NVIDIA hardware and has no macOS or ARM package.

The Windows runtime places every DLL beside both executables in `runtime/bin`,
which makes the application directory the one package-local DLL source before
Windows system resolution. Its package has no `runtime/lib` directory. The
Linux runtime keeps executables in `runtime/bin`, shared objects in
`runtime/lib`, and the worker receives only that directory in its child-only
`LD_LIBRARY_PATH`. Neither platform carries duplicate native files in a second
directory.

A clean native release job uses [`tools/build_runtime.py`](../tools/build_runtime.py)
and [`worker/runtime-lock.json`](../worker/runtime-lock.json) to acquire the
exact Audio2Face SDK revision, CUDA 12.9 components, TensorRT 10.13 inputs,
CMake, and Windows CRT inputs. It builds the worker and packages the `trtexec`
shipped in those exact TensorRT binary inputs without consulting a workstation or
runner CUDA or TensorRT installation. CUDA compiler files, headers,
import/static libraries, and driver stubs are build inputs only. The NVIDIA
display driver is the one required host component.

The Windows producer is pinned to VCToolsVersion 14.43.34808, `cl` 19.43.34810,
and Windows SDK 10.0.22621.0. The Linux producer is pinned to the Rocky Linux
8.9 amd64 image digest in `runtime-lock.json`, glibc 2.28, GCC Toolset 11.2.1,
the old libstdc++ ABI, and generic x86-64 code generation. Its runtime contains
the exact locked Rocky BaseOS `libstdc++.so.6` and `libgcc_s.so.1` files rather
than resolving either file from the user's host. The pipeline does not claim
bit-for-bit reproducible native binaries.

The native CMake build writes `audio2face_worker` and `audio2x` directly into
the temporary runtime package. Python adds the exact contract-defined runtime
files after the native build command returns to host Python; there is no custom
CMake staging target. The native builder publishes exactly
`build/runtime/<platform>`. Then
[`tools/build_extension.py`](../tools/build_extension.py) accepts that one
handoff plus an absolute Blender 5.2 executable, creates the complete temporary
extension source directory required by Blender, pins its manifest to one
platform, writes one standard ZIP-LZMA archive, invokes Blender's extension
validator on both the source directory and finished archive, and verifies the
ZIP layout and bytes. The only outputs are
`dist/audio2face-<version>-windows-x64.zip` and
`dist/audio2face-<version>-linux-x64.zip`. Native compilation never occurs inside
Blender's extension command.

`trtexec` comes from the matching locked TensorRT binary input. Windows uses
the official ZIP; Linux streams six pinned NVIDIA RHEL8 RPMs one at a time and
keeps only the required headers, regular runtime libraries, and `trtexec`. This
preserves the model-defined `trt_info.json` command
contract used by NVIDIA's Audio2Face SDK without inventing a second option
schema. TensorRT engines are generated later on the user's GPU because a
serialized engine is not a generally portable model artifact.

At startup and before model optimization, the add-on resolves exactly
`audio2face/runtime/` inside its own installed package. It requires the
platform in `bundle.json` to match the host, checks the native executable
format, confines every declared executable, library directory, and notice to
that runtime tree, and constructs a child environment from only those
package-local executable and library directories plus the operating-system
directory required on Windows. It does not inspect `PATH`, `LD_LIBRARY_PATH`,
an installed CUDA Toolkit, an installed TensorRT SDK, or another Audio2Face
installation. It performs no online request.

**Optimize Models**:

1. validates each selection as a directory with a non-empty, valid top-level
   `model.json` whose `networkPath` is exactly `network.trt`, non-empty
   top-level `network.onnx` and `trt_info.json`, and every other descriptor
   path resolved as a canonical relative, non-empty regular file confined to
   that root; Git LFS pointers are rejected;
2. validates the fixed bundled runtime payload;
3. runs the bundled `trtexec` on CUDA device 0 for each selected model,
   building each completed engine as a temporary sibling candidate;
4. honors cancellation while each native build is running; and
5. atomically replaces both `<selected-root>/network.trt` engines as one
   transaction after both candidates succeed.

Both external model roots must be writable. Re-running the action rebuilds
both engines for the current GPU. A failed build or activation restores the
previous pair and never exposes a partial `network.trt`. The selected roots
remain exactly where the user placed them. Only those two external model roots
are selected by the user; the executable, runtime libraries, SDK, engine
builder, working directory, and CUDA device are fixed by the extension.

A missing, damaged, or wrong-platform runtime invalidates the installed
extension. The supported resolution is to install the correct complete
platform ZIP. Preferences do not mutate or replace package runtime files.
Release validation must exercise the supported model pair and confirm that
Audio2Emotion's post-processed vector order agrees with Audio2Face's emotion
order; SDK 1.0.0 reports the vector width but does not expose names for those
output positions.

The top of Add-on Preferences presents a right-aligned **Uninstall** action
with Blender's familiar two-line add-on-name/package-path confirmation. The
confirming Audio2Face operator returns before a one-shot main-thread timer
delegates to Blender 5.2's native
extension uninstaller, preventing the operator from unregistering its own
class while it is executing. The native disable phase runs Audio2Face's normal
process, stream, playback, timer, and handler cleanup; its package phase
removes the installed extension and its bundled worker and runtime libraries.
The two selected model repository roots and their generated `network.trt`
engines are external user data. Uninstall never deletes them. User-selected
audio, `.blend` data, and shared GPU caches also remain outside this ownership
boundary.

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

**Start Worker** launches only the validated package-local child, sends
`hello {}`, and loads the two selected Audio2Face and Audio2Emotion models.
The controller retains the exact validated runtime/model specification across
that handshake; it does not read changed Preferences between process launch
and `load_model`. One worker accepts at most one generation or stream
operation.

**Generate ARKit Values** reloads when the selected identity changes, freezes
the current model-described settings, and submits a complete WAV. **Cancel
Generation** interrupts active execution and prevents a partial result commit.

**Start WAV Stream** freezes the same settings and target subscriptions,
incrementally decodes and resamples the selected WAV, and sends bounded mono
f32le chunks. `stream_start` returns the exact model sample rate and
`prebuffer_samples`. The built-in source satisfies that lead before starting
audible playback, then samples frames against the audio-device clock.

The public [`audio2face.streaming`](../audio2face/streaming.py) API lets code
already running in Blender provide live model-rate mono f32le PCM. After
`start_pcm_stream(scene)`, the source polls
`get_pcm_stream_requirements(scene)` until it receives
`(sample_rate, prebuffer_samples)`, queues that exact initial lead as immutable
`bytes`, and owns audible monitoring. No socket is opened.

Generation and stream submissions record exact `operation_id`-to-scene
mappings. Asynchronous events route only through those controller-owned maps;
RNA values are checked after lookup but are never scanned as a routing path.

**Stop Stream** cancels the active stream and keeps the models ready. **Stop
Worker** requests bounded shutdown and escalates to process termination if the
child misses its deadlines. Destruction releases executors, model metadata,
accumulators, and the CUDA stream in dependency order.

## Model schema and settings

The UI persists two model roots, but the extension derives their exact
top-level `model.json` paths before protocol submission. `load_model` accepts
only those two validated absolute descriptor paths and a non-negative identity
index. It returns a positive `sample_rate` and one `model_schema` with exactly:

- `identities`: ordered non-empty names from Audio2Face;
- `channels`: the exact 52 unique model-provided names in model order;
- `parameters`: opaque worker paths mapped to numeric SDK defaults; and
- `emotion_channels`: ordered `{name, default}` records from Audio2Face.

Blender owns no independent identity, emotion, output-channel, or parameter
list. It validates the schema and materializes RNA collections from it.
SDK 1.0.0 has no parameter reflection, so one typed worker adapter defines the
supported paths; Blender displays each path exactly and owns no duplicate
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
- `operation_id`
- `sample_rate`
- `channels`
- `timestamps_samples`
- `weights`

The channel array is the negotiated model order. Every row has exactly 52
finite values in `[0.0, 1.0]`. Timestamps are non-empty, strictly increasing
signed 64-bit audio-sample positions, and row count equals timestamp count.
Live frames use the same negotiated channel order.

Selected playback verifies the add-on-owned result and submitted WAV identity,
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
1 MiB. A malformed method response, malformed or misrouted event, unknown
response/error ID, or id-less worker diagnostic is a terminal contract
violation: Blender clears every active operation, stops live presentation,
and shuts down that worker before another control message can mutate scene
state.

PCM chunks contain non-empty finite little-endian float32 samples, cover at
most one model-rate second, and are limited to 256 KiB before base64 encoding.
Blender permits at most 64 pending chunk acknowledgements, providing bounded
backpressure. Operation IDs, timestamps, sample rates, and all coefficient rows
are validated before Blender state changes.

Result files stay inside the add-on-owned results directory, are limited to
512 MiB, and are committed atomically without replacing an existing file.
Runtime validation, model optimization, and worker failures appear in Blender
status. Unexpected child exit clears model and stream state and reports the
latest worker diagnostic.

Unregistering the extension stops result and live playback, cancels its WAV
source and model optimization, unregisters the timer, and closes the worker
before removing Blender classes and scene RNA.
