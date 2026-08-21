# Release automation

`.github/workflows/release.yml` is the only GitHub release path. Pushing the
exact version tag from `audio2face/blender_manifest.toml` starts two native
builds, uploads the two complete platform extension ZIPs to one draft release,
verifies that the draft contains exactly those assets, and then publishes it.
For manifest version `0.1.0`, the only accepted tag is `v0.1.0` and the assets
are:

```text
audio2face-0.1.0-windows-x64.zip
audio2face-0.1.0-linux-x64.zip
```

The workflow reuses an existing draft for the same tag, so a failed job can be
rerun without deleting the draft. It refuses an already-published release.
Native uploads use `--clobber`; the final job publishes only after both jobs
succeed and the draft contains exactly the two expected, non-empty assets.

## Native runner labels

Configure one native x64 GitHub Actions runner for each literal workflow label:

- `audio2face-windows-x64`
- `audio2face-linux-x64`

Each label may name a GitHub-hosted larger runner or a dedicated self-hosted
runner. Standard GitHub-hosted runners are not release builders for this
project: GitHub guarantees only 14 GB of SSD space, while the locked compressed
inputs alone total approximately 2.66 GiB on Windows and 7.77 GiB on Linux,
before extraction, compilation, staging, Blender, and the final extension ZIP.
The workflow requires at least 64 GiB free on every Windows workspace/temp
volume and 96 GiB free on every Linux workspace/temp filesystem.

The Windows runner requires:

- Windows x64;
- current GitHub Actions Runner, Git for Windows, and GitHub CLI on `PATH`;
- Visual Studio 2022 or Build Tools with component
  `Microsoft.VisualStudio.Component.VC.14.43.17.13.x86.x64`;
- MSVC toolset `14.43.34808` / compiler `19.43.34810`; and
- Windows SDK `10.0.22621.0`.

The Linux runner requires:

- x86-64 Linux;
- current GitHub Actions Runner, Git, `curl`, `tar`, `xz`, and GitHub CLI; and
- Docker Engine, with the runner account authorized to run Docker containers.

Neither runner needs an NVIDIA GPU, CUDA Toolkit, TensorRT installation,
Audio2Face installation, or either model repository. The runtime builder
downloads and verifies every locked development/runtime input, and the model
files are deliberately not part of the extension package.

## Release lifecycle

1. Update `version` in `audio2face/blender_manifest.toml` and merge the complete
   release source.
2. Create and push the exact `v<version>` tag.
3. The prepare job runs the Python tests and creates or verifies a draft.
4. The native jobs download the official portable Blender 5.2.0 archives,
   verify their pinned SHA-256 digests, run Blender smoke tests, build the
   locked runtime, package the platform extension, enforce GitHub's per-asset
   size limit, and upload directly to the draft.
5. The final job requires exactly the Windows and Linux asset names, verifies
   each upload state and size, and publishes the draft.

GitHub Releases requires every individual asset to be smaller than 2 GiB. The
workflow enforces that limit before upload and again before publication. If a
complete platform ZIP reaches that limit, it cannot be published as a
ready-to-install GitHub release asset; the runtime/package composition or
distribution host must be changed rather than splitting the Blender extension
ZIP.
