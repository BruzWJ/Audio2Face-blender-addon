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
on listed Mesh, Curve, Surface, and Lattice objects. A selected WAV is uploaded
once and rendered as one temporally coherent, timestamped track cache. Native
Blender frames sample that cache for preview, and a separate bake samples the
same cache into Shape Key Actions.

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
driver remains on the host. The installed add-on does not access the network;
Blender's extension manager handles release checks and downloads. The add-on's
model-source buttons open NVIDIA model pages in the user's browser.

There are exactly two release packages: one complete Windows x64 extension ZIP
with PE executables and their DLLs together in `runtime/bin`, and one complete
Linux x64 extension ZIP with ELF executables in `runtime/bin` and shared
objects in `runtime/lib`. These native formats are not interchangeable. This
CUDA backend requires NVIDIA hardware and has no macOS or ARM build.

## Installation and updates

In Blender 5.2, open **Preferences > Get Extensions**, open the Repositories
menu, and choose **Add Remote Repository**. Name it `Audio2Face` and use:

```text
https://github.com/BruzWJ/Audio2Face-blender-addon/releases/latest/download/index.json
```

Enable **Check for Updates on Startup**, refresh the repository, and install
Audio2Face. Blender selects the package for the current operating system and
offers newer releases through its normal extension update controls.

If Audio2Face was previously installed from disk into **User Default**, first
uninstall that copy, then install it from the remote repository above. Do not
keep both copies because they share the extension ID `audio2face`. Uninstalling
does not remove the external model folders or their generated `network.trt`
files, although their paths may need to be selected again.

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

The runtime is replaced only as part of a complete Blender extension update; it
is never installed or repaired separately. A missing, damaged, or
wrong-platform runtime means the extension package is invalid and must be
reinstalled. The add-on never searches the host for another worker, CUDA
Toolkit, TensorRT, Audio2Face installation, or executable.

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
and bundled runtime, so normal worker and active-inference cleanup runs. The
selected external model repositories and their `network.trt` engines remain in
place, as do selected WAV files, `.blend` files, object data, and shared NVIDIA
driver caches.

## Workflow

1. Install and enable Audio2Face from its remote repository, then select and
   optimize both models in Add-on Preferences.
2. In the Audio2Face sidebar, choose **Selected WAV** or **Stream**. Selected
   WAV mode shows Blender timeline playback and bake controls.
3. In Selected WAV mode, choose a WAV. In Stream mode, a Blender integration
   supplies live mono f32le PCM through
   [`audio2face.streaming`](audio2face/streaming.py).
4. Select Mesh, Curve, Surface, or Lattice objects and click **Add Selected
   Objects**. Selected WAV baking requires at least one target with a matching
   model channel; targets remain optional for a live Stream.
5. Click **Start Worker**. Blender launches the verified package-local worker,
   negotiates the protocol, and loads both selected models. When the worker,
   models, and Selected WAV source are ready, Blender uploads that source once
   and keeps its track prepared independently of media playback.
6. Configure the saved, animatable **Preferred Emotion** sliders. Any nonzero
   value enables that source; set every value to zero to clear it. **Mixed
   Emotion** is read-only output from Selected WAV frame evaluation or Stream
   input.
7. In Selected WAV mode, set **First Frame**, then use Blender's Timeline or
   Spacebar transport. Each native frame change samples the corresponding row
   from the prepared cache and transiently updates matching Shape Keys. Model,
   emotion, and Preferred Emotion keyframes are evaluated over the sound span;
   edits revise that animated cache without changing media or worker lifecycle.
   Click **Bake Shape Key Animation** separately when the preview should become
   native Shape Key Actions.
8. In Stream mode, the first `push_audio_f32le` call automatically starts the
   inference stream. `end_pcm_stream` marks normal end-of-input after all
   queued chunks. **Stop Worker** exits the child process and releases its
   models and CUDA resources.

Installing or enabling the extension does not start the worker. Loading the
models prepares the GPU/model process but does not start Blender media.
Selected WAV source readiness creates and retains one prepared track and
its latest complete render; Blender playback merely selects cached samples.
The first external PCM chunk instead starts a true sequential Stream operation.
**Start Worker** and **Stop Worker** alone control the GPU/model process
lifecycle. Worker and track lifecycle changes never start or stop Blender
media, and Blender play or pause never starts or stops the worker or prepared
track.

## Shape Key targets and channel delivery

**Add Selected Objects** accepts Mesh, Curve, Surface, and Lattice objects.
Each exact model-provided channel name drives the matching Shape Key; missing
keys are skipped and names are never remapped. Targets do not need the NVIDIA
reference topology.

Stream targets are resolved for every delivered frame, so edits to the list
take effect without restarting inference. An empty list suppresses Blender
writes without stopping the stream.

A bake requires at least one matching Shape Key. It reuses only an Action
explicitly owned by this add-on; otherwise it creates and assigns one, leaving
the artist's prior Action untouched.
Objects sharing one Key datablock share one result; make the object data
single-user when independent animation is required.

## Audio modes and playback

Choosing a **Selected WAV** creates or updates one add-on-owned sound strip in
Blender's Video Sequencer at the saved **First Frame**. The add-on preserves
unrelated strips and never changes Blender's scene or preview playback range.
When both the source and loaded models are ready, the add-on decodes,
downmixes, and resamples the WAV, uploads it once, and prepares a persistent
track operation.

Blender's Timeline and Spacebar are the only Selected WAV transport controls.
The strip keeps Blender's native duration, and the scene uses **Sync to Audio**
so delayed viewport evaluation drops frames instead of allowing sound to run
ahead. Blender's current frame maps to one audio sample in the completed track
cache. Pause freezes it, scrubbing samples it immediately, and a native range
loop wraps sound and facial values together. The worker has no play, pause,
loop, or seconds-position state. Frames outside the sound interval are neutral.

Changing model tuning, emotion tuning, Preferred Emotion, or their keyframes
starts a newer continuous render revision. Blender sends the values evaluated
at native frames as one sample-based settings timeline. Editing an Action
invalidates and rebuilds the complete schedule before transport is involved, so
future FCurve edits are included even when the current value is unchanged. A
newer revision supersedes older work without canceling the resident track, and
a completed revision is published atomically; native playback keeps sampling
the prior complete cache while that work runs. Edits made during playback or a
bake stay pending until it stops, avoiding a hidden full-timeline seek inside
native transport. The requested current-frame sample is emitted from the new
continuous result before its cache batches, so paused preview and bake remain
identical.

**Bake Shape Key Animation** is asynchronous and separate from playback. From
the native sound-strip start through its inclusive end, it samples the same
published cache used by preview and writes LINEAR Shape Key curves into an
add-on-owned native Action. Bake does not upload the WAV, run stateless
per-frame inference, or replace the track. Re-baking replaces the add-on-owned
Action's curves, while unrelated artist Actions remain unchanged.
**Prediction Delay** shifts the sampled audio position; positive values advance
the face and negative values make it lag. Canceling before completion writes no
Action curves.

**Stream** accepts external model-rate mono f32le chunks. Call
`get_pcm_stream_requirements(scene)` to obtain `(sample_rate, None)` before
input is accepted, then call `push_audio_f32le(payload, scene_name=...)`. The
first chunk automatically creates the internal stream and is retained while
the worker acknowledges it. Subsequent requirements return
`(sample_rate, prebuffer_samples)`. Call
`end_pcm_stream(scene_name=...)` to mark normal end-of-input after every queued
chunk. The integration owns capture, resampling, and audible monitoring. No
port or network listener is opened.

## Model-driven emotion

The loaded models define the channel list, emotion names, and Audio2Face
defaults shown in Blender. **Preferred Emotion** values are saved and
animatable; any nonzero channel enables that source and setting all channels to
zero clears it. **Mixed Emotion** is transient, read-only output from Selected
WAV frame evaluation and external Stream input, and never overwrites Preferred
values.

In Selected WAV mode, Blender evaluates the rendered animated settings on
native frames. Paused edits revise the continuous cache and update the current
frame; edits authored during playback are retained and rendered after it stops,
without interrupting the active cache. In Stream mode, frame-evaluated emotion,
Preferred, and model-tuning changes update the live executors in place before
subsequent PCM; they do not reset inference, replay PCM, or restart the worker.
Exact settings fields and ranges are in the
[protocol](docs/protocol.md#settings-document).

## Output contract

A bake returns 52 finite ARKit coefficients in model order for each requested
Blender frame. Selected WAV frame evaluation and external Stream frames also
return aligned effective emotion values. Only those scalars enter Blender;
internal geometry, solver meshes, and transforms remain in the worker.

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
