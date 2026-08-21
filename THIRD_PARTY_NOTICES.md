# Third-party notices

This source checkout does not contain an NVIDIA native runtime or either
NVIDIA model. A complete end-user release is one platform-specific extension
ZIP whose `audio2face/runtime/` tree has passed dependency and license review.

A Windows x64 or Linux x64 extension ZIP may include the following separately
licensed components in its bundled runtime:

- NVIDIA Audio2Face-3D SDK and Audio2X runtime;
- NVIDIA CUDA user-mode runtime libraries;
- NVIDIA TensorRT user-mode runtime libraries;
- NVIDIA TensorRT `trtexec`, built from the release's pinned TensorRT source,
  and its dependencies;
- on Linux, the exact Rocky Linux 8.9 `libstdc++.so.6` and `libgcc_s.so.1`
  runtime files pinned in `worker/runtime-lock.json`.

The extension contains no Audio2Face or Audio2Emotion model files and no
serialized TensorRT engines. Each platform runtime must contain only files
approved for redistribution and must reproduce every license, copyright
notice, and third-party acknowledgement required by those exact files. The
runtime notices must identify the exact TensorRT source revision, version,
license, and build provenance for the bundled `trtexec`. Windows and Linux
native files are packaged only in their corresponding extension ZIP. Linux
packages preserve the five GCC runtime license texts from the locked Rocky
`libgcc` RPM and generated provenance for both binary RPMs and the matching
Rocky source RPM.

Users obtain the complete Audio2Face-3D v3.0 and Audio2Emotion v3.0 folders
directly from NVIDIA's Hugging Face repositories under their applicable terms.
The add-on provides source links and persistent folder selectors for the exact
roots of those complete repositories. It derives only the top-level
`model.json` from each root. It does not authenticate, download, copy,
relocate, redistribute, or delete those model files.

After validating the top-level `model.json`, `network.onnx`, `trt_info.json`,
and every descriptor-referenced file, setup builds or rebuilds separate
Audio2Face and Audio2Emotion `network.trt` files on CUDA device 0. Both engine
candidates must succeed before they atomically replace the current pair. Each
engine ends at `<selected-root>/network.trt`. These external engines are not
part of the extension package, and uninstall does not delete them. The NVIDIA
display driver is not downloaded or redistributed by this project.

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
- [Rocky Linux 8.9 Vault](https://dl.rockylinux.org/vault/rocky/8.9/)
- [GNU GCC runtime-library exception](https://www.gnu.org/licenses/gcc-exception-3.1.html)
- [Blender licensing](https://www.blender.org/about/license/)

If any required component cannot be redistributed under its applicable terms,
release production must fail. The add-on uses only its package-local worker,
runtime libraries, and `trtexec`; it never substitutes a separately configured
executable, runtime, automatic model downloader, or hosted service.
