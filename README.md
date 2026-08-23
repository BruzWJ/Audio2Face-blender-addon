# Audio2Face

Audio2Face is a Blender 5.2 extension that runs NVIDIA Audio2Face-3D v3.0 and
Audio2Emotion v3.0 locally on an NVIDIA GPU. Each operating-system-specific
extension ZIP contains one fixed native runtime payload. Blender owns that local
worker process; users do not host a service or choose an executable, SDK,
CUDA installation, TensorRT installation, or working directory. They select
only the exact root folders of the two complete NVIDIA model repositories they
obtained separately.

The extension produces 52-channel ARKit coefficients from a selected WAV or
incremental mono float32 PCM. It drives existing Shape Key `value`
properties on enabled mesh targets. It does not write vertices, bones,
Actions, F-curves, or baked animation.

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

The top of Add-on Preferences includes a right-aligned **Uninstall** action. It
opens Blender's familiar two-line **Remove Add-on** confirmation with the
Audio2Face name and installed package path, then delegates removal to Blender's
extension uninstaller. Disabling the extension stops its worker, streams, and
playback before the package and its bundled runtime are removed. The selected
external model repositories and their `network.trt` engines remain in place,
as do selected WAV files, `.blend` files, meshes, and shared NVIDIA driver
caches.

## Workflow

1. Install the extension ZIP for the current platform, enable Audio2Face, then
   select and optimize both models in Add-on Preferences.
2. In the Audio2Face sidebar, choose **Selected WAV** or **Stream**. The
   **Audio Playback** controls appear immediately below this mode selector.
3. Choose a WAV for complete generation or for the built-in streamed-WAV
   source. A Blender integration can use Stream mode without a WAV by pushing
   live mono f32le PCM through [`audio2face.streaming`](audio2face/streaming.py).
4. Select any mesh objects and click **Add Selected Meshes**. Shape Keys are
   not required when a mesh is added.
5. Click **Start Worker**. Blender launches the verified package-local worker,
   negotiates the protocol, and loads both selected models.
6. Choose a model identity and adjust the controls reported by that model.
7. Leave **Auto Audio2Emotion** off to use the manual emotion values. Enable it
   to replace the manual driver with emotions inferred from the same audio.
8. In Selected WAV mode, click **Generate ARKit Values**, then **Play Result**.
   In Stream mode, click **Start WAV Stream** or submit live PCM through the
   integration API.
9. **Stop Stream** ends only the active stream and keeps the loaded model
   ready. **Stop Worker** exits the child process and releases its model and
   CUDA resources.

Installing or enabling the extension does not start the worker. Loading the
models does not start continuous inference. GPU inference runs only for an
explicit generation operation or active PCM stream. A completed Selected WAV result
can play after the worker has stopped.

## Mesh targets and channel delivery

Every enabled target mesh subscribes to the model channel stream, with no
Shape Key admission check. At each frame, Blender uses each exact
model-provided channel name to look up a Shape Key on each target.
If that key exists, its value is assigned; if it does not, that channel is
skipped for that target. Names are never translated or remapped, and there is
no per-target multiplier, bake step, or direct mesh deformation.

The loaded model supplies the exact ordered 52-channel list. A target can
contain all, some, or none of those Shape Keys. Several objects may share one
Shape Key datablock; delivery writes that shared datablock once per frame, so
linked objects reflect the same values. Use single-user mesh data when objects
need independent values.

## Audio modes and playback

**Selected WAV** sends one complete RIFF/WAVE file to the worker. The worker
decodes, downmixes, resamples, generates all frames, and atomically commits a
strict result. **Audio Playback** uses Blender's audio-device clock to sample
that result and provides play, pause/resume, stop, loop, volume, and
reset-on-stop controls.

**Stream** uses one `stream_start` / chunk / `stream_end` lifecycle. The
built-in source incrementally decodes a selected WAV, resamples it to the model
rate, and keeps a bounded lead over Blender audio playback. Integrations can
instead call `start_pcm_stream`, poll `get_pcm_stream_requirements`, and submit
model-rate mono f32le chunks. Those integrations own capture, resampling, and
audible monitoring. No port or network listener is opened.

The worker reports the initial model-rate `prebuffer_samples` requirement.
With automatic emotion enabled, it covers both Audio2Face and Audio2Emotion
readiness. Streamed frames are buffered and sampled against the local audio or
presentation clock; scene FPS is not used for synchronization.

## Model-driven controls

`load_model` returns a self-describing `model_schema` with exactly
`identities`, `channels`, `parameters`, and `emotion_channels`. Blender builds
its selectors and numeric controls from those values. `parameters` is one
object mapping opaque worker paths to numeric defaults; JSON integer and float
types select the corresponding Blender control. Blender displays each opaque
parameter ID exactly as advertised and does not parse, rename, group, or map it.

NVIDIA SDK 1.0 exposes parameter structures but no parameter reflection. The
worker therefore contains the one typed path-to-member adapter; Blender owns
no parameter list. Defaults, identities, emotion channels, and output channels
come from the loaded SDK/model. Internal graph nodes and tensors are not
controls. Both input modes submit one exact settings object:

```json
{
  "auto_audio2emotion": false,
  "manual_emotions": {"<model emotion name>": 0.0},
  "parameters": {"/advertised/path": 0.0}
}
```

Manual emotion values are a constant, model-shaped conditioning vector. When
`auto_audio2emotion` is true, Audio2Emotion analyzes the same input and fully
replaces that vector. Manual values are ignored in automatic mode. The same
semantics and complete frozen settings apply to Selected WAV and Stream; a
stream must be restarted to apply changed controls.

## Output contract

The worker reports the model's exact 52 unique channel names in
model order. It resolves eye-look values into the corresponding
model-provided channel slots without reordering the list. Raw geometry, jaw
transforms, eye rotations, and other solver data are not serialized.

Selected mode stores `a2f-animation/2` with exactly `schema`, `operation_id`,
`sample_rate`, `channels`, `timestamps_samples`, and `weights`.
Stream mode uses the same negotiated channel order for incremental frames and
does not create a result file. Coefficients are finite and within `[0.0, 1.0]`;
timestamps are integer audio-sample positions.

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
