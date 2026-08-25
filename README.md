# Audio2Face

Audio2Face is a Blender 5.2 extension that runs NVIDIA Audio2Face-3D v3.0 and
Audio2Emotion v3.0 locally on an NVIDIA GPU. Each operating-system-specific
extension ZIP contains one fixed native runtime payload. Blender owns that local
worker process; users do not host a service or choose an executable, SDK,
CUDA installation, TensorRT installation, or working directory. They select
only the exact root folders of the two complete NVIDIA model repositories they
obtained separately.

The extension produces 52-channel ARKit coefficients from a selected WAV or
incremental mono float32 PCM and drives existing Shape Key `value` properties
on listed Mesh, Curve, Surface, and Lattice objects. Inference starts
automatically with WAV playback or incoming PCM.

## Requirements

- Blender 5.2.x on Windows x64 or Linux x64
- The matching Windows x64 or Linux x64 Audio2Face extension ZIP
- A supported NVIDIA GPU and compatible NVIDIA display driver
- Complete Audio2Face-3D v3.0 and Audio2Emotion v3.0 Hugging Face repository
  roots on a writable local filesystem
- Space in each selected root for its GPU-specific `network.trt` engine

The user does not install a CUDA Toolkit, TensorRT, Docker, or a separate
service. The extension ZIP carries the required Audio2X and CUDA/TensorRT
user-mode libraries beside the worker; only the compatible NVIDIA display
driver remains on the host. The installed add-on does not access the network.
Its model-source buttons open NVIDIA model pages in the user's browser.

There are exactly two release packages: one complete Windows x64 extension ZIP
with PE executables and their DLLs together in `runtime/bin`, and one complete
Linux x64 extension ZIP with ELF executables in `runtime/bin` and shared
objects in `runtime/lib`. These native formats are not interchangeable. This
CUDA backend requires NVIDIA hardware and has no macOS or ARM build.

## Installation

Install the downloaded platform ZIP into a **local** Blender extension
repository. In **Preferences > Get Extensions**, open the Repositories menu.
If **User Default** is missing, choose **Add Local Repository** and name it
`User Default`; do not give it a remote URL. Then choose **Install from Disk**,
expand **Extensions** in the file browser, set **Repository** to
**User Default**, and select the ZIP.

Do not install a downloaded ZIP into `extensions.blender.org`. That repository
is the catalog of packages published by Blender, and Audio2Face is not in its
index. Blender labels any locally installed package missing from that remote
index as **Orphan**. If Audio2Face already has that label, uninstall that copy
and repeat the installation into **User Default**.

## Model setup

Model setup lives in **Edit > Preferences > Add-ons > Audio2Face**. The setup
box contains:

- one [NVIDIA terms](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
  link and one acceptance checkbox;
- download buttons for
  [Audio2Face-3D v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) and
  gated [Audio2Emotion v3.0](https://huggingface.co/nvidia/Audio2Emotion-v3.0);
- two persistent folder selectors for the complete downloaded repository
  roots; and
- **Optimize Models**.

Clone or download and extract both complete model repositories yourself. Each
folder selector may point to its exact repository root anywhere on disk. The
root must contain the repository's top-level `model.json`; that is the only
descriptor path the add-on derives. It does not sign in to Hugging Face,
download, copy, relocate, or delete model files.

The Hugging Face repositories intentionally contain `network.onnx`, not
`network.trt`. After selecting both repositories, click **Optimize Models**;
the add-on generates each GPU-specific `network.trt` locally from its ONNX
model. Do not download or create that file yourself.

The extension validates its package-local `runtime/bundle.json`, native
executables, runtime directories, and notices. Setup then validates
`model.json`, `network.onnx`, and `trt_info.json` in each selected root,
rejects unresolved model references and Git LFS pointer files, and runs the
bundled `trtexec` on CUDA device 0. Since the worker owns one audio track, the
add-on builds a one-track profile while preserving NVIDIA's model-provided
buffer ranges and other TensorRT settings. It builds both engine candidates
before atomically replacing `<selected-root>/network.trt` in the two roots, so
both roots must be writable. Preferences provide cancellation, progress,
readable failure summaries, and access to the complete TensorRT logs. The 3D
View sidebar only reports readiness and directs model setup to Preferences.

The runtime is not installed, updated, or repaired after the extension is
installed. A missing, damaged, or wrong-platform runtime means the extension
package is invalid and must be replaced with the correct platform ZIP. The
add-on never searches the host for another worker, CUDA Toolkit, TensorRT,
Audio2Face installation, or executable.

Release workers must be built in clean Windows and Linux CI from pinned NVIDIA
sources and binary archives, never from a developer workstation's installed
GPU development stack. NVIDIA documents this application-local deployment
model for the [CUDA runtime and
libraries](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-best-practices-guide/index.html#cuda-runtime-and-libraries)
and provides [per-component CUDA archives for package maintainers and
CI](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables).

Blender owns the extension lifecycle. To remove Audio2Face, open **Preferences →
Get Extensions**, find Audio2Face, open the down-arrow menu on its card, and
choose **Uninstall**. Blender disables the add-on before removing its package
and bundled runtime, so normal worker, stream, and playback cleanup runs. The
selected external model repositories and their `network.trt` engines remain in
place, as do selected WAV files, `.blend` files, object data, and shared NVIDIA
driver caches.

## Workflow

1. Install the extension ZIP for the current platform into the local **User
   Default** repository, enable Audio2Face, then select and optimize both models
   in Add-on Preferences.
2. In the Audio2Face sidebar, choose **Selected WAV** or **Stream**. The
   **Playback** controls appear immediately below this mode selector.
3. In Selected WAV mode, choose a WAV. In Stream mode, a Blender integration
   supplies live mono f32le PCM through
   [`audio2face.streaming`](audio2face/streaming.py).
4. Optionally select Mesh, Curve, Surface, or Lattice objects and click
   **Add Selected Objects**.
5. Click **Start Worker**. Blender launches the verified package-local worker,
   negotiates the protocol, and loads both selected models.
6. The collapsible **Preferred Emotion** sliders are the saved, editable
   emotion input. Leave **Auto Audio2Emotion** off to use them as the direct
   constant driver, or enable it to use them as the optional preference mixed
   with generated emotion. The separate **Mixed Emotion** values are read-only
   worker output. **Load** copies that current output into Preferred Emotion;
   **Clear** disables the preference and zeros its sliders. A direct Preferred
   edit enables the preference and applies immediately. Mixed Emotion never
   becomes inference input.
7. In Selected WAV mode, press **Play**. Playback automatically starts
   incremental inference and drives matching Shape Keys as frames arrive.
   **Pause** freezes both audio and source pacing; seek and loop restart
   the inference stream at the requested position without restarting the
   worker.
8. In Stream mode, the first `push_audio_f32le` call automatically starts the
   inference stream. `end_pcm_stream` marks normal end-of-input after all
   queued chunks. **Stop Worker** exits the child process and releases its
   models and CUDA resources.

Installing or enabling the extension does not start the worker. Loading the
models does not start continuous inference. GPU inference runs only while
Selected WAV playback or an external PCM stream is active. **Start Worker** and
**Stop Worker** control the GPU/model process lifecycle; audio playback and PCM
arrival control inference within that lifecycle.

## Shape Key targets and channel delivery

Every Mesh, Curve, Surface, or Lattice in the target list receives the model
channel stream, with no existing-Shape-Key admission check. These are all
object types on which Blender 5.2 supports Shape Keys. At each frame, Blender
uses each exact model-provided channel name to look up a Shape Key on each
listed target. The target collection is resolved again for every delivered
frame, so adding or removing an object takes effect on the next frame without
restarting playback or inference. An empty list simply suppresses Blender
writes; it does not stop the audio or worker stream.
If that key exists, its value is assigned; if it does not, that channel is
skipped for that target. Names are never translated or remapped, and there is
no per-target multiplier.

The loaded model supplies the exact ordered 52-channel list. A target can
contain all, some, or none of those Shape Keys. Several objects may share one
Shape Key datablock; delivery writes that shared datablock once per frame, so
linked objects reflect the same values. Use single-user object data when
objects need independent values.

The selected Blender object does not need the Audio2Face reference topology.
The default Audio2Face-3D v3.0 model repository carries its own
identity-specific 24,002-vertex neutral basis and 52 pose bases. Inside the
worker, NVIDIA's GPU blendshape solver converts the model's raw geometry output
against that internal basis into 52 scalar coefficients. Only those named
coefficients cross into Blender, where matching target Shape Keys receive the
values and absent names are skipped. The model basis is never used to deform a
Blender target directly.

## Audio modes and playback

**Selected WAV** uses one stateful **Play/Pause** button. Pressing Play
incrementally decodes, downmixes, and resamples the WAV, starts an internal
`stream_start` / chunk / `stream_end` operation, and begins audible playback
after the worker's required input lead is queued. ARKit frames are sampled
against Blender's audio-device clock as they arrive. **Pause** freezes both
audio and WAV pacing. **Loop** and the duration-based seek control
restart that stream at the requested audio position while keeping the worker
and models loaded. The seek control is an editable Blender slider whose native
range is `0` through the selected WAV's duration, not a normalized progress
display. The elapsed / duration timecode and **Prediction Delay**
from `-1.0` to `1.0` seconds remain playback controls. Positive delay advances
facial motion relative to audible audio; negative delay makes it lag.

**Stream** accepts external model-rate mono f32le chunks. Call
`get_pcm_stream_requirements(scene)` to obtain `(sample_rate, None)` before
input is accepted, then call `push_audio_f32le(payload, scene_name=...)`. The
first chunk automatically creates the internal stream and is retained while
the worker acknowledges it. Subsequent requirements return
`(sample_rate, prebuffer_samples)`. Call
`end_pcm_stream(scene_name=...)` to mark normal end-of-input after every queued
chunk. The integration owns capture, resampling, and audible monitoring. No
port or network listener is opened.

The worker reports an initial model-rate `prebuffer_samples` requirement that
covers both Audio2Face and Audio2Emotion readiness, so automatic emotion can be
toggled during an active stream. Streamed frames are buffered and sampled
against the local audio or presentation clock; scene FPS is not used for
synchronization.

## Model-driven emotion

`load_model` returns a self-describing `model_schema` with exactly `channels`,
`emotion_channels`, and `audio2face_defaults`. Blender builds target-channel
delivery and emotion controls from the first two values and seeds its
fixed Audio2Face controls from the model-reported defaults. It does not define
an independent output or emotion name list. The worker uses the Audio2Face
model's default identity at SDK index `0` internally; Blender has no identity
selector or identity state.

`stream_start` installs one exact initial settings snapshot containing the fixed
18-field `audio2face` object, `auto_audio2emotion`, all model-described
`manual_emotions`, and the `audio2emotion` object. Audio2Emotion controls remain
available regardless of the **Auto Audio2Emotion** toggle. Blender presents two
model-defined emotion collections: saved, editable **Preferred Emotion** and
transient, read-only **Mixed Emotion**. These are the Audio2Emotion defaults:

```json
{
  "emotion_strength": 0.6,
  "emotion_contrast": 1.0,
  "max_emotions": 6,
  "live_blend_coef": 0.7,
  "transition_smoothing": 0.5,
  "preferred_emotion": null,
  "preferred_emotion_strength": 0.5
}
```

With automatic emotion off, the saved Preferred Emotion values populate the
protocol's `manual_emotions` field and form the direct, constant emotion driver.
With it on, Audio2Emotion generates timestamped values and applies strength,
contrast, retained-emotion count, temporal blend, and transition controls
through NVIDIA's SDK post-processor. When preferred mixing is enabled, the same
saved values are supplied as `audio2emotion.preferred_emotion`. Direct edits to
Preferred Emotion enable preferred mixing and immediately refresh the active
inference settings; switching Auto Audio2Emotion does not rewrite the authored
values.

**Load** copies the currently displayed Mixed Emotion values into Preferred
Emotion and enables preferred mixing. **Clear** disables preferred mixing and
zeros the saved Preferred Emotion values. The Preferred Emotion collection is
saved with the Blender scene. For preferred-mix weight `p` the SDK
computes `p * preferred + (1 - p) * generated`, then applies the overall
emotion strength. **Emotion Strength** ranges from `0.0` to `2.0`; values above
`1.0` amplify this automatic-emotion result without changing **Skin Strength**.
Individual emotion values and the preferred-mix weight remain in `[0.0, 1.0]`.
Equal emotion values are learned model-conditioning weights, not equal visual
deformation amplitudes, so different emotions can remain perceptually uneven.

Each returned frame contains the effective, post-processed values sampled by
Audio2Face, aligned with its ARKit weights. Blender writes them only to the
read-only Mixed Emotion display, clamping that display when NVIDIA returns a
finite value outside the factor range. Mixed Emotion is transient, is not saved
with the scene, and never supplies `manual_emotions`, `preferred_emotion`, or
any other inference setting.

The same complete settings contract applies to Selected WAV and Stream. A
control edit queues one complete replacement snapshot on the active operation.
The worker resets only its inference state, replays a bounded recent PCM
context, and publishes replacement face frames on the same timeline. The
operation ID, audible playback, play/pause state, external PCM ingress, loaded
models, and worker process remain unchanged. The exact fields, types, ranges,
and replay boundary are defined in
[`docs/protocol.md`](docs/protocol.md#settings-document). Internal graph nodes
and tensors are not controls.

## Output contract

The worker reports the default v3 model's exact 52 unique channel names in
model order. It resolves eye-look values into the corresponding model-provided
channel slots without reordering the list. Both modes receive incremental
`stream_frame` records in that negotiated order. Each record also carries one
effective, post-processed emotion value per `emotion_channels` entry in model
order. Coefficients are finite and within `[0.0, 1.0]`; effective emotions are
finite but are not clamped by NVIDIA's SDK. Both vectors share one integer
audio-sample timestamp. Raw geometry, jaw transforms, eye rotations, and
internal solver meshes never leave the worker.

See [architecture](docs/architecture.md), [protocol](docs/protocol.md), and the
[worker build guide](worker/README.md) for the full contracts.

## Package and verify

Build each release on its native operating system. On Windows x64:

```sh
python tools/build_runtime.py --platform windows-x64
python tools/build_extension.py --blender "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --platform windows-x64
```

Run these commands from an ordinary PowerShell session. The runtime builder
uses Visual Studio Installer's `vswhere.exe` to locate Visual Studio 2022 or
Build Tools, selects the exact compiler pinned by `runtime-lock.json`, and
initializes `vcvarsall.bat` itself. A Developer Command Prompt and manual
environment setup are not required.

On Linux x64:

```sh
python3 tools/build_runtime.py --platform linux-x64
python3 tools/build_extension.py --blender /absolute/path/to/blender --platform linux-x64
```

The runtime build writes exactly `build/runtime/<platform>`. The extension
builder validates that handoff, creates the complete temporary package root
required by Blender, pins the manifest to one platform, writes one standard
ZIP-LZMA archive, and runs Blender 5.2 validation on both the package root and
the finished archive before verifying its contents byte-for-byte. It writes:

```text
dist/audio2face-<version>-windows-x64.zip
dist/audio2face-<version>-linux-x64.zip
```

Extension packaging does not compile the native worker. ZIP-LZMA keeps the
complete locked NVIDIA runtime in one Blender-installable GitHub release asset
below GitHub's per-asset limit. The two-step release scripts are therefore the
complete production path. `tools/build_runtime.py` routes the selected platform
to its dedicated native builder, rejects a target that differs from its host,
and never reads an installed CUDA or TensorRT development stack.

Run the Python and Blender source smoke suites with:

```sh
python3 -m pytest -q
blender --factory-startup --background --python tests/blender_smoke.py
```

A production release must also pass real NVIDIA GPU inference, dependency,
installation, cancellation, and shutdown tests for both platform extension
ZIPs.

Tagged Windows and Linux packages are built and published only by manually
running the native [GitHub release workflow](docs/releasing.md) on the
repository's default branch. The workflow compares that branch with the latest
published release, generates the current UTC calendar version as `YYYY.M.D`,
tests and commits that version to `audio2face/blender_manifest.toml`, then
freezes the resulting commit for both native builds. The matching `vYYYY.M.D`
tag is created automatically immediately before the verified draft is
published. The release jobs reclaim the standard
`windows-latest` and `ubuntu-latest` GitHub-hosted images, use verified portable
Blender 5.2.0 archives, and run the same two production build scripts shown
above.

## Licensing

The extension and worker source are GPL-3.0-or-later. NVIDIA runtime components
and model files remain under their applicable terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
