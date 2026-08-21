# Audio2Face worker

This directory builds the native GPU child used by the Blender 5.2 extension.
It runs Audio2Face-3D v3.0 and Audio2Emotion v3.0 on CUDA device 0 and emits
model-described ARKit coefficient frames. End users do not build,
locate, configure, or host the executable. Audio2Face Add-on Preferences
installs the native runtime separately from the user-selected models.

NVIDIA publishes the Audio2Face-3D SDK source and the ONNX-based model inputs;
it does not publish this Blender worker. Release maintainers build and publish
the worker and its native dependencies as separate Windows x64 and Linux x64
packages for this add-on.

Production archives must be built only in clean platform CI. The jobs fetch the
pinned SDK source, NVIDIA's checksummed CUDA 12.9 component archives, the
matching TensorRT 10.13 GA package, and the exact TensorRT source revision used
for `trtexec`. They do not inspect or consume a GPU development stack installed
on a developer machine. NVIDIA publishes the component manifest specifically
for [package maintainers and CI/CD
systems](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables).

## Development build

The worker has one SDK source input: a pinned NVIDIA Audio2Face-3D-SDK 1.0.0
tree.

```sh
cmake -S worker -B build/worker \
  -DA2F_SDK_SOURCE_DIR=/absolute/path/to/Audio2Face-3D-SDK \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/worker --config Release
```

CMake verifies `audio2x-sdk/VERSION.md`, adds that source tree directly, and
links `audio2x-sdk::audio2x`. The SDK does not publish an installed CMake
package, so the project has no SDK finder, include/library override, binary SDK
path, or SDK environment variable. Building requires the CUDA and TensorRT
development inputs required by the pinned SDK. A local development build may
supply them explicitly; it is not the source of a publishable runtime archive.

## Release runtime artifact

Release maintainers enable strict runtime staging and provide reviewed,
absolute inputs:

```sh
cmake -S worker -B build/worker-release \
  -DA2F_SDK_SOURCE_DIR=/absolute/path/to/Audio2Face-3D-SDK \
  -DA2F_WORKER_STAGE_RUNTIME=ON \
  -DA2F_CUDA_RUNTIME_DIR=/absolute/path/to/reviewed/cuda-runtime \
  -DA2F_TENSORRT_RUNTIME_DIR=/absolute/path/to/reviewed/tensorrt-runtime \
  -DA2F_BUNDLE_TRTEXEC=/absolute/path/to/release-built/trtexec \
  -DA2F_BUNDLE_TRTEXEC_SOURCE_LICENSE=/absolute/path/to/tensorrt-license \
  -DA2F_BUNDLE_TRTEXEC_PROVENANCE=/absolute/path/to/trtexec-provenance \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/worker-release \
  --target audio2face_runtime_archive --config Release
```

`A2F_BUNDLE_TRTEXEC` must be NVIDIA TensorRT `trtexec` built from the pinned
source recorded by its license and provenance files. Release staging accepts
only Linux x64 and Windows x64. It requires the reviewed CUDA/TensorRT runtime
filenames and all native dependency license and notice inputs. It rejects
symlinks, unsupported binary formats, unexpected layout, and empty files.

Those platform checks are required: Windows packages carry PE executables and
DLLs, while Linux packages carry ELF executables and shared objects. One
format cannot be loaded in place of the other. The worker is CUDA-only, so
this release design does not support macOS or ARM systems.

The generated ZIP contains only the platform-specific managed runtime:

```text
runtime/<platform>/
  bundle.json
  bin/audio2face_worker[.exe]
  bin/trtexec[.exe]
  lib/...
  licenses/...
```

`StageRuntime.cmake` creates and validates this canonical tree.
`MeasureRuntimeArchive.cmake` verifies the ZIP member set and emits the exact
`sha256`, `size`, and `unpacked_size` values. Release automation adds the
immutable HTTPS URL and inserts that record under the matching platform key in
[`audio2face/runtime_catalog.json`](../audio2face/runtime_catalog.json).

Each downloadable artifact is OS/architecture-specific and contains no model
files, model licenses, or GPU-specific serialized engines. Users download the
gated NVIDIA Audio2Face and Audio2Emotion model packages themselves and select
each package's `model.json` in Add-on Preferences. The add-on remembers those
locations and uses the bundled project-built `trtexec` to create each model's
local `network.trt` for CUDA device 0. The NVIDIA display driver remains a
system requirement and is not bundled.

The Preferences UI links to the two gated model sources:
[Audio2Face-3D v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) and
[Audio2Emotion v3.0](https://huggingface.co/nvidia/Audio2Emotion-v3.0).
The add-on does not download these gated assets, embed credentials, or
redistribute their licenses. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The checked-in catalog contains no platform records. The current extension ZIP
is therefore a development package: install remains disabled and GPU inference
is unavailable until reviewed Windows x64 and Linux x64 archives and their
measured records are published. The diagnostic calls this an unpublished
release asset; it does not reject the detected host or GPU.

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
ownership, managed installation, target delivery, and audio-clocked playback
are in [`docs/architecture.md`](../docs/architecture.md).
