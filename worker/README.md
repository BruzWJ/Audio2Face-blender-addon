# Audio2Face worker

This directory builds the native child process used by the Blender 5.2
extension. It runs NVIDIA Audio2Face-3D v3 diffusion and the device blendshape
solver on CUDA device 0. End users never build, locate, configure, or host this
executable; Blender's **Install Runtime & Models** workflow installs and launches
it.

## Build for development

The worker has one SDK integration input: a pinned NVIDIA
Audio2Face-3D-SDK 1.0.0 source tree.

```sh
cmake -S worker -B build/worker \
  -DA2F_SDK_SOURCE_DIR=/absolute/path/to/Audio2Face-3D-SDK \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/worker --config Release
```

CMake verifies the SDK's `audio2x-sdk/VERSION.md`, adds that source tree
directly, and links `audio2x-sdk::audio2x`. There is no SDK finder, include/lib
override, binary-only SDK path, or SDK environment variable.

The source tree implements only the production backend. Building requires the
CUDA and TensorRT development environment required by the pinned SDK.

## Build a managed runtime archive

Release maintainers enable the strict staging target and provide only reviewed,
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
  -DA2F_BUNDLE_AUDIO2FACE_MODEL_DIR=/absolute/path/to/audio2face-v3-model \
  -DA2F_BUNDLE_AUDIO2FACE_MODEL_LICENSE=/absolute/path/to/audio2face-model-license \
  -DA2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR=/absolute/path/to/audio2emotion-v3-model \
  -DA2F_BUNDLE_AUDIO2EMOTION_MODEL_LICENSE=/absolute/path/to/audio2emotion-model-license \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/worker-release \
  --target audio2face_runtime_archive --config Release
```

`A2F_BUNDLE_TRTEXEC` must be NVIDIA TensorRT `trtexec` built from the pinned
release source recorded by its license and provenance files. A locally found
binary is not accepted as release provenance.

The staging path admits only Linux x86-64 and Windows AMD64 builds. It requires
the exact reviewed CUDA/TensorRT runtime filenames, complete Audio2Face v3 and
Audio2Emotion v3 model trees, and separate license inputs for both models. It
rejects symlinks, unsupported binary formats, unexpected layout, empty files,
and prebuilt `.trt` or `.engine` files in either model tree.

The generated ZIP has one canonical payload:

```text
runtime/<platform>/
  bundle.json
  bin/audio2face_worker[.exe]
  bin/trtexec[.exe]
  lib/...
  models/audio2face/model.json
  models/audio2face/network.onnx
  models/audio2face/trt_info.json
  models/audio2face/...
  models/audio2emotion/model.json
  models/audio2emotion/network.onnx
  models/audio2emotion/trt_info.json
  models/audio2emotion/...
  licenses/...
```

`StageRuntime.cmake` creates and validates that tree.
`MeasureRuntimeArchive.cmake` verifies the ZIP member set against it and emits
the platform's exact `sha256`, `size`, and `unpacked_size` fragment. Release
automation adds the immutable HTTPS URL and inserts the artifact under the
matching platform key in
[`audio2face/runtime_catalog.json`](../audio2face/runtime_catalog.json).

The downloadable archive remains GPU-neutral. During add-on installation,
Blender uses the bundled `trtexec` and each model's `trt_info.json` to build the
Audio2Face engine first and the Audio2Emotion engine second. Each build has its
own progress stage and install log. Blender validates both generated
`network.trt` files before atomically activating the one service-free runtime.
The NVIDIA display driver remains a system requirement and is not bundled.

The Audio2Emotion integration is experimental, and Audio2Emotion v3.0 is a
gated model. Its acquisition and redistribution terms must be reviewed
explicitly; release packaging does not embed credentials, download it from
Hugging Face, or fabricate a catalog artifact. Archive contents and
redistribution rights must be reviewed for every release.
See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The checked-in
catalog currently publishes no artifacts, so the source extension ZIP alone is
not yet a complete end-user installation.

## Runtime contract

The process is silent until `hello {}` and communicates only through strict
UTF-8 JSON Lines on stdin/stdout. One non-interactive diffusion/device-
blendshape executor serves both input modes: a complete WAV produces one
five-field `a2f-animation/1` result, while start/chunk/end PCM input emits
incremental canonical ARKit-52 frames. There is no socket or separately hosted
service. A loaded but idle model does not run inference, **Stop Stream** keeps
the model ready, and **Stop Worker** releases its CUDA/model resources.

The exact message, settings, ARKit-52, result, cancellation, and shutdown
contracts are documented in [`docs/protocol.md`](../docs/protocol.md). Process
ownership, managed installation, target meshes, and audio-clocked Blender
preview are documented in
[`docs/architecture.md`](../docs/architecture.md).
