# Third-party notices

This source checkout does not contain an NVIDIA native runtime or either
NVIDIA model. It must not be distributed as a complete end-user installation
until the platform worker artifacts, catalog URLs, measured sizes, SHA-256
digests, and required notices have passed release validation.

A native worker archive may include the following separately licensed
components:

- NVIDIA Audio2Face-3D SDK and Audio2X runtime;
- NVIDIA CUDA user-mode runtime libraries;
- NVIDIA TensorRT user-mode runtime libraries;
- NVIDIA TensorRT `trtexec`, built from the release's pinned TensorRT source,
  and its dependencies.

The worker archive contains no Audio2Face or Audio2Emotion model files and no
serialized TensorRT engines. Each worker archive must contain only files
approved for redistribution and must reproduce every license, copyright
notice, and third-party acknowledgement required by those exact files. The
release record must identify the exact TensorRT source revision, version,
license, and build provenance for the bundled `trtexec`.

Users obtain the complete Audio2Face-3D v3.0 and Audio2Emotion v3.0 folders
directly from NVIDIA's Hugging Face repositories under their applicable terms.
The add-on provides source links and persistent folder selectors for the exact
roots of those complete repositories. It derives only the top-level
`model.json` from each root and performs no recursive search or fallback. It
does not authenticate, download, copy, relocate, redistribute, or delete those
model files.

After validating the top-level `model.json`, `network.onnx`, `trt_info.json`,
and every descriptor-referenced file, setup builds or rebuilds separate
Audio2Face and Audio2Emotion `network.trt` files on CUDA device 0. The engines
and worker are committed together with rollback, and each engine ends at
`<selected-root>/network.trt`. These external engines are not part of the
worker archive or the add-on's managed data, and uninstall does not delete
them. The NVIDIA display driver is not downloaded or redistributed by this
project.

The Audio2Emotion integration is experimental, and Audio2Emotion v3.0 is gated
by NVIDIA on Hugging Face. This project does not embed access credentials or
automate acceptance of the repository's access terms. Users complete that flow
on Hugging Face before selecting the downloaded repository root.

Relevant upstream terms include:

- [Audio2Face-3D SDK repository and license](https://github.com/NVIDIA/Audio2Face-3D-SDK)
- [Audio2Face-3D v3.0 model repository](https://huggingface.co/nvidia/Audio2Face-3D-v3.0)
- [Audio2Emotion v3.0 model repository](https://huggingface.co/nvidia/Audio2Emotion-v3.0)
- [Audio2Emotion v3.0 model license](https://huggingface.co/nvidia/Audio2Emotion-v3.0/blob/main/LICENSE)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
- [TensorRT source and license](https://github.com/NVIDIA/TensorRT)
- [Blender licensing](https://www.blender.org/about/license/)

If any required component cannot be redistributed under its applicable terms,
release production must fail. The installer must never substitute a separately
configured executable, runtime, model download, or hosted service.
