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
TensorRT 10.13 binary archive and source revision, CMake distribution, Windows
CRT package, Rocky Linux producer image and toolchain RPMs, and Rocky BaseOS
GNU runtime RPMs. Artifact size and SHA-256 values are part of that lock.

[`tools/build_runtime.py`](../tools/build_runtime.py) is the only supported
native runtime build entry point. Run it natively on the target operating
system: Windows builds `windows-x64`, and Linux builds `linux-x64`. A Windows
release host provides the pinned Visual C++ toolset, Windows SDK, Git, and
Python. A Linux release host provides Docker, Git, and Python; the compiler and
system headers are installed from locked RPMs in the locked Rocky producer
image. The script acquires every CUDA, TensorRT, CMake, SDK, Windows CRT, and
Linux GNU runtime input from the lock into an isolated work directory.

Windows is pinned to VCToolsVersion 14.43.34808, `cl` 19.43.34810, and Windows
SDK 10.0.22621.0. Linux is pinned to the Rocky Linux 8.9 amd64 image digest,
glibc 2.28, GCC Toolset 11.2.1, the old libstdc++ ABI, and generic x86-64 code
generation. The Linux package carries the exact locked Rocky BaseOS
`libstdc++.so.6` and `libgcc_s.so.1` bytes. Neither path claims bit-for-bit
reproducible native binaries.

The release build never discovers or consumes a host CUDA Toolkit, TensorRT
SDK, Audio2Face installation, worker executable, or GPU development
environment. It builds the worker and `trtexec` from the pinned sources, checks
source revisions and native paths, stages only the reviewed runtime dependency
closure, and records the required notices. CUDA compiler files, headers,
import/static libraries, driver stubs, and TensorRT development sources are
build inputs and are not shipped.

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
writes `dist/audio2face-0.1.0-<platform>.zip`. It does not invoke a compiler or
inspect the build machine's GPU stack.

## Runtime layout

Windows staging produces exactly:

```text
runtime/windows-x64/
  bundle.json
  bin/audio2face_worker.exe
  bin/trtexec.exe
  bin/*.dll
  licenses/...
```

Linux staging produces exactly:

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

The platform extension builder copies the contents of that tree to the
immutable package location:

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
in its search path. There is no separate runtime setup, runtime mutation,
executable selector, or online request.

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
`audio2face/3` UTF-8 JSON Lines on stdin/stdout. It reports worker profile
`nvidia-a2f3-a2e3-gpu-arkit52/2`.

`load_model` returns the model sample rate and one exact `model_schema` with
`identities`, `channels`, `parameters`, and `emotion_channels`. Channels retain
the model's exact 52-name order. Parameters are an object mapping opaque worker
paths to numeric SDK defaults. SDK 1.0 has no parameter reflection, so the
worker is the single typed adapter from those paths to the SDK structures;
Blender owns no duplicate list or presentation metadata.

One non-interactive diffusion/device-blendshape executor and one
Audio2Emotion executor serve both modes. A complete WAV produces a six-field
`a2f-animation/2` result. Start/chunk/end PCM input emits incremental frames in
the negotiated channel order. The worker opens no socket. An idle loaded model
does not run inference; **Stop Stream** keeps it ready, and **Stop Worker**
releases its CUDA and model resources.

The exact transport, settings, model schema, result, cancellation, and
shutdown contracts are in [`docs/protocol.md`](../docs/protocol.md). Process
ownership, package-local runtime validation, target delivery, and
audio-clocked playback are in
[`docs/architecture.md`](../docs/architecture.md).
