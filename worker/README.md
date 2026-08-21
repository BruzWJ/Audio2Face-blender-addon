# Audio2Face worker

This directory builds the native GPU child used by the Blender 5.2 extension.
It runs Audio2Face-3D v3.0 and Audio2Emotion v3.0 on CUDA device 0 and emits
model-described ARKit coefficient frames. End users do not build,
locate, configure, or host the executable. Audio2Face Add-on Preferences
installs it together with the runtime and both models.

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
development environment required by the pinned SDK.

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
  -DA2F_BUNDLE_AUDIO2FACE_MODEL_DIR=/absolute/path/to/audio2face-v3-model \
  -DA2F_BUNDLE_AUDIO2FACE_MODEL_LICENSE=/absolute/path/to/audio2face-model-license \
  -DA2F_BUNDLE_AUDIO2EMOTION_MODEL_DIR=/absolute/path/to/audio2emotion-v3-model \
  -DA2F_BUNDLE_AUDIO2EMOTION_MODEL_LICENSE=/absolute/path/to/audio2emotion-model-license \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/worker-release \
  --target audio2face_runtime_archive --config Release
```

`A2F_BUNDLE_TRTEXEC` must be NVIDIA TensorRT `trtexec` built from the pinned
source recorded by its license and provenance files. Release staging accepts
only Linux x64 and Windows x64. It requires the reviewed CUDA/TensorRT runtime
filenames, both complete model input trees, and all license and notice inputs.
It rejects symlinks, unsupported binary formats, unexpected layout, empty
files, and prebuilt `.trt` or `.engine` files in either model tree.

The one generated ZIP contains both models and the full managed runtime:

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

`StageRuntime.cmake` creates and validates this canonical tree.
`MeasureRuntimeArchive.cmake` verifies the ZIP member set and emits the exact
`sha256`, `size`, and `unpacked_size` values. Release automation adds the
immutable HTTPS URL and inserts that record under the matching platform key in
[`audio2face/runtime_catalog.json`](../audio2face/runtime_catalog.json).

The downloadable artifact is GPU-neutral. During Add-on Preferences setup,
Blender verifies and extracts it, then runs the bundled `trtexec` once for
Audio2Face and once for Audio2Emotion. Both generated `network.trt` files are
validated before the runtime is activated atomically. The NVIDIA display
driver remains a system requirement and is not bundled.

The Preferences UI presents one NVIDIA terms link and acceptance, plus source
buttons for
[Audio2Face-3D v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) and
[Audio2Emotion v3.0](https://huggingface.co/nvidia/Audio2Emotion-v3.0). The
buttons identify the model sources; the one managed artifact delivers both.
Release maintainers must review model access and redistribution rights and may
not embed credentials. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The checked-in catalog contains no platform records, so install remains
disabled until reviewed archives and their measured records are published.

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
