# Release automation

`.github/workflows/release.yml` is the only GitHub release path. It runs only
when started manually with an existing version tag that exactly matches
`audio2face/blender_manifest.toml`. The workflow starts two native builds,
uploads the two complete platform extension ZIPs to one draft release, verifies
that the draft contains exactly those assets, and then publishes it.
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

## Standard GitHub-hosted runners

The two native jobs use only GitHub's standard `windows-latest` and
`ubuntu-latest` x64 labels. There are no custom, larger, or self-hosted runner
labels to configure. The `-latest` names intentionally follow GitHub's current
stable images; the Windows job installs its locked compiler and SDK instead of
depending on the compiler version preinstalled on that moving image.

GitHub specifies 14 GB of SSD storage for a standard runner. The standard
images also contain development stacks that this release never consumes, so
each native job removes an explicit allowlist before selecting its one Python:

- Windows removes the hosted tool cache, Android SDK, GHCup, .NET, Miniconda,
  MSYS2, and vcpkg trees.
- Ubuntu removes the hosted tool cache, Android SDK, GHCup, .NET, Swift,
  Miniconda, Homebrew, and vcpkg trees.

The cleanup does not remove Git, GitHub CLI, Docker on Ubuntu, the Windows
installer, Windows Kits, or the Ubuntu swap file. Workspace and temporary
locations still come from `GITHUB_WORKSPACE` and `RUNNER_TEMP`; the workflow
does not assume a `D:` drive or `/mnt` scratch disk. Both native jobs require at
least 12 GiB free after cleanup and use two-way build parallelism. See GitHub's
[standard runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [six-hour hosted-job limit](https://docs.github.com/en/actions/reference/limits).

The Windows job downloads Microsoft's official Visual Studio 2022 Build Tools
bootstrapper and installs only the components required by the locked build:

- `Microsoft.VisualStudio.Component.VC.14.43.17.13.x86.x64`; and
- `Microsoft.VisualStudio.Component.Windows11SDK.22621`.

It then verifies MSVC toolset `14.43.34808` and Windows SDK
`10.0.22621.0` before invoking the runtime builder. The Ubuntu image supplies
Git, `curl`, `tar`, `xz`, GitHub CLI, and Docker Engine; the locked Rocky Linux
producer container supplies the compiler and system headers.

Neither runner needs an NVIDIA GPU, CUDA Toolkit, TensorRT installation,
Audio2Face installation, or either model repository. The runtime builder
downloads and verifies every locked development/runtime input, and the model
files are deliberately not part of the extension package.

## Release lifecycle

1. Update `version` in `audio2face/blender_manifest.toml` and merge the complete
   release source.
2. Create and push the exact `v<version>` tag.
3. In GitHub Actions, open **Release platform extensions**, keep the workflow
   selector on the repository's default branch, choose **Run workflow**, enter
   that exact tag, and start the workflow. The prepare job resolves
   `refs/tags/<tag>` once; both native jobs build that frozen commit.
4. The prepare job runs the Python tests and creates or verifies a draft.
5. Each native job reclaims its standard runner, selects Python 3.11.9, and
   builds the locked runtime. Windows first installs and verifies the exact
   Visual Studio 2022 components listed above.
6. The native jobs download the official portable Blender 5.2.0 archives,
   verify their pinned SHA-256 digests, discard the downloaded archives after
   extraction, run Blender smoke tests, package the platform extensions,
   enforce GitHub's per-asset size limit, and upload directly to the draft.
7. The final job requires exactly the Windows and Linux asset names, verifies
   each upload state and size, confirms that the tag was not moved during the
   run, and publishes the draft.

GitHub Releases requires every individual asset to be smaller than 2 GiB. The
workflow enforces that limit before upload and again before publication. If a
complete platform ZIP reaches that limit, it cannot be published as a
ready-to-install GitHub release asset; the runtime/package composition or
distribution host must be changed rather than splitting the Blender extension
ZIP.
