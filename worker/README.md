# Audio2Face native runtime

This directory builds the native GPU child bundled in the Audio2Face Blender
5.2 extension. It runs Audio2Face-3D v3.0 and Audio2Emotion v3.0 on CUDA device
0 and emits model-described ARKit coefficient frames. End users do not build,
install, locate, configure, or host this executable or any of its runtime
libraries.

Each release has exactly two complete extension ZIPs:

- a Windows x64 ZIP containing PE executables and DLLs; and
- a Linux x64 ZIP containing ELF executables and shared objects.

The native formats are not interchangeable. The CUDA backend requires NVIDIA
hardware and a compatible NVIDIA display driver. macOS and ARM are not
supported.

## Canonical release build

[`runtime-lock.json`](runtime-lock.json) is the complete release-input
contract. It pins the Audio2Face-3D SDK revision, CUDA 12.9 component archives,
the platform's TensorRT 10.13 binary inputs, CMake distribution, Windows
CRT package, Rocky Linux producer image and toolchain RPMs, and Rocky BaseOS
GNU runtime RPMs. Artifact size and SHA-256 values are part of that lock.

[`tools/build_runtime.py`](../tools/build_runtime.py) is the supported native
runtime entry point. Its required `--platform` argument routes to the dedicated
[`build_windows_runtime.py`](../tools/build_windows_runtime.py) or
[`build_linux_runtime.py`](../tools/build_linux_runtime.py) implementation. Each
implementation runs only on its matching native operating system. A Windows
release host provides the pinned Visual C++ toolset, Windows SDK, Git, and
Python. A Linux release host provides Docker, Git, and Python; the compiler and
system headers are installed from locked RPMs in the locked Rocky producer
image. The selected builder acquires its CUDA, TensorRT, CMake, SDK, and
platform runtime inputs from the lock into an isolated work directory.

On Windows, run the command from ordinary PowerShell. The builder locates
Visual Studio 2022 or Build Tools with Visual Studio Installer's `vswhere.exe`,
selects the exact locked toolset, and initializes `vcvarsall.bat` automatically.
It does not require a Developer Command Prompt or caller-provided Visual Studio
environment variables.

Windows is pinned to VCToolsVersion 14.43.34808, `cl` 19.43.34810, and Windows
SDK 10.0.22621.0. Linux is pinned to the Rocky Linux 8.9 amd64 image digest,
glibc 2.28, GCC Toolset 11.2.1, the old libstdc++ ABI, and generic x86-64 code
generation. The Linux package carries the exact locked Rocky BaseOS
`libstdc++.so.6` and `libgcc_s.so.1` bytes. Neither path claims bit-for-bit
reproducible native binaries.

The release build never discovers or consumes a host CUDA Toolkit, TensorRT
SDK, Audio2Face installation, worker executable, or GPU development
environment. It builds the worker, selects the exact `trtexec` shipped in the
pinned TensorRT ZIP or Linux RPM set, checks source revisions and native paths,
packages only the reviewed runtime dependency closure, and records the
required notices.
CUDA compiler files, headers, import/static libraries, and driver stubs are
build inputs and are not shipped. On Linux, the builder moves the
redistributable archives' single `lib` tree to the canonical `lib64` toolkit
location required by their pinned `nvcc`; it does not create an alias or search
a second CUDA library path.

CMake has no install, staging, or packaging target. It builds
`audio2face_worker` and its `audio2x` dependency directly into the temporary
runtime package directory. After the native build command returns, host Python
adds the contract-defined libraries and notices.
Those package entries are hard links within the repository build filesystem,
so the multi-gigabyte native payload remains single-copy. There is no
cross-device operation or link/copy fallback.
The same Python runtime contract records the selected Linux libraries' exact
SONAME and RUNPATH identities; the release audit rejects every DT_RPATH and
any SONAME or RUNPATH that differs from that contract.

Build on the native target host:

```sh
# Windows x64
python tools/build_runtime.py --platform windows-x64
python tools/build_extension.py --blender "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --platform windows-x64

# Linux x64
python3 tools/build_runtime.py --platform linux-x64
python3 tools/build_extension.py --blender /absolute/path/to/blender --platform linux-x64
```

The runtime builder writes only `build/runtime/<platform>`, requires that
handoff path not to exist before the build, and uses one temporary isolated
working tree. The
[`tools/build_extension.py`](../tools/build_extension.py) packaging step
requires the exact handoff, validates Blender 5.2, embeds the runtime, and
writes `dist/audio2face-<version>-<platform>.zip`. It does not invoke a compiler or
inspect the build machine's GPU stack.

## Runtime layout

The Windows runtime package contains exactly:

```text
runtime/windows-x64/
  bundle.json
  bin/audio2face_worker.exe
  bin/trtexec.exe
  bin/*.dll
  licenses/...
```

The Linux runtime package contains exactly:

```text
runtime/linux-x64/
  bundle.json
  bin/audio2face_worker
  bin/trtexec
  lib/*.so*
  licenses/...
```

Windows DLLs live beside the two executables so the application directory is
the canonical package-local DLL source. They are not copied into a second
directory. Linux executables use the one sibling `lib` directory.

The platform extension builder adds the contents of that tree to the package
location Blender archives:

```text
audio2face/runtime/
  bundle.json
  bin/...
  lib/...  # Linux x64 only
  licenses/...
```

`bundle.json` is the exact package-local resolver contract. The installed
extension verifies that its platform matches the host, checks every packaged
executable and DLL/shared object, confines every declared path to this tree,
and launches the child with only the platform's package-local native directory
in its search path. Because Blender's Linux ZIP installer does not preserve
execute bits, the resolver validates the exact worker and `trtexec` ELF files,
then idempotently restores their owner execute bit. There is no separate
runtime setup, executable selector, or online request.

The bundled runtime contains no Audio2Face or Audio2Emotion model files and no
serialized TensorRT engines. Users obtain both complete model repositories
from NVIDIA and select their exact roots in Audio2Face Add-on Preferences:

- [Audio2Face-3D v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0)
- [Audio2Emotion v3.0](https://huggingface.co/nvidia/Audio2Emotion-v3.0)

Preferences validate the two top-level `model.json`, `network.onnx`, and
`trt_info.json` inputs, then run the bundled `trtexec` to build a
GPU-specific `network.trt` beside each `model.json`. Both candidates must
succeed before the current engine pair is atomically replaced. Uninstalling
the extension removes the bundled native runtime but does not delete either
external model root or its generated engine.

## Runtime contract

The process is silent until `hello {}` and communicates only through strict
`audio2face/4` UTF-8 JSON Lines on stdin/stdout. It reports worker profile
`nvidia-a2f3-a2e3-gpu-arkit52/4`.

`load_model` returns the model sample rate and one exact `model_schema` with
`channels` and `emotion_channels`. Channels retain the model's exact 52-name
order, and emotion names and defaults are model-provided. The worker selects
the Audio2Face model's default identity at SDK index `0` internally; identity
is not a protocol input, schema field, or Blender control.
The default v3 model's internal solver data supplies its identity-specific
24,002-vertex neutral basis and 52 pose bases. NVIDIA's GPU blendshape solver
converts raw Audio2Face geometry against that basis into the 52 scalar channel
values. Blender target mesh topology never enters the worker; only matching
Shape Key names are relevant after delivery.

Audio2Emotion post-processing is configured per `stream_start` operation
through this exact nested settings object (values shown are Blender defaults):

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

The complete operation settings also carry `auto_audio2emotion` and a snapshot
of every model-described manual emotion. With automatic emotion off, that
snapshot is accumulated directly as a constant driver. With it on, the worker
resets the Audio2Emotion executor before setting the SDK post-process
parameters. `preferred_emotion` is independently either `null` or an exact
model-named snapshot captured by Blender's **Load** action; **Clear** unsets it,
and later manual changes do not mutate the loaded snapshot. When present, for
preferred strength `p` the SDK mixes
`p * preferred + (1 - p) * generated`, then applies overall
`emotion_strength`. Selected WAV and Stream use the same frozen settings
contract.

One non-interactive diffusion/device-blendshape executor and one
Audio2Emotion executor serve both modes. Selected WAV playback and external
PCM both use start/chunk/end input and emit incremental frames in the negotiated
channel order. There is no complete-file generation method or animation result
file. The worker opens no socket. An idle loaded model does not run inference;
Blender's **Start Worker** and **Stop Worker** control the process and its CUDA
resource lifetime.

The exact transport, settings, model schema, streaming, cancellation, and
shutdown contracts are in [`docs/protocol.md`](../docs/protocol.md). Process
ownership, package-local runtime validation, target delivery, and
audio-clocked playback are in
[`docs/architecture.md`](../docs/architecture.md).
