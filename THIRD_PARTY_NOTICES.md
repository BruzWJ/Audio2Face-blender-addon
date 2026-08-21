# Third-party notices

This source checkout does not contain the managed NVIDIA runtime or model
archive. It must not be distributed as a complete end-user installation until
the platform artifacts, catalog URLs, measured sizes, SHA-256 digests, and
required notices have passed release validation.

A managed runtime may include the following separately licensed components:

- NVIDIA Audio2Face-3D SDK and Audio2X runtime;
- NVIDIA CUDA user-mode runtime libraries;
- NVIDIA TensorRT user-mode runtime libraries;
- NVIDIA Audio2Face-3D v3.0 model files;
- NVIDIA Audio2Emotion v3.0 model files; and
- NVIDIA TensorRT `trtexec`, built from the release's pinned TensorRT source,
  and its dependencies.

Each runtime archive must contain only files approved for redistribution and
must reproduce every license, copyright notice, model notice, and third-party
acknowledgement required by those exact files. The release record must identify
the exact TensorRT source revision, version, license, and build provenance for
the bundled `trtexec`.

The model archive contains ONNX and related inputs, not GPU-specific TensorRT
engines. The installer builds separate Audio2Face and Audio2Emotion
`network.trt` files locally; those generated engines remain inside the user's
Blender extension-data directory. The NVIDIA display driver is not downloaded
or redistributed by this project.

The Audio2Emotion integration is experimental, and Audio2Emotion v3.0 is gated
by NVIDIA on Hugging Face. This project does not embed access credentials or
assume redistribution permission. A managed runtime cannot be published until
its release review confirms how the exact model files may be acquired and
distributed with all applicable notices.

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
configured executable, runtime, model path, or hosted service.
