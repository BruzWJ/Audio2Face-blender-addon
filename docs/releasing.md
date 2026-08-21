# Release automation

`.github/workflows/release.yml` is the only GitHub release path. It runs only
when started manually from GitHub Actions. The branch selected in the **Run
workflow** interface must be the repository's protected default branch; there
are no workflow inputs. The workflow freezes that branch's dispatch commit,
compares it with the published release marked **Latest**, and derives
`v<version>` from `audio2face/blender_manifest.toml`. After the source tests
pass, it creates that tag at the frozen commit. It then starts two native builds,
uploads the two complete platform extension ZIPs to one draft release, verifies
that the draft contains exactly those assets, and publishes it. For manifest
version `0.1.0`, the generated tag and assets are:

```text
v0.1.0
audio2face-0.1.0-windows-x64.zip
audio2face-0.1.0-linux-x64.zip
```

The latest published release tag must resolve to an ancestor of the selected
commit, and the range from that tag to the selected commit must contain at least
one commit. The selected manifest version must also be newer than the version
embedded at the latest release tag. This makes the manifest the sole version
source of truth; the workflow does not infer a semantic-version bump from commit
subjects.

The workflow reuses an existing lightweight tag and draft only when both still
point to the same frozen source commit, so a failed job can be rerun without
deleting the draft. It refuses a moved tag or an already-published release.
It also refuses to skip over an unrelated pending draft. Native uploads use
`--clobber`; the final job publishes only after both jobs succeed and the draft
contains exactly the two expected assets with the SHA-256 digests and byte sizes
reported by their native build jobs.

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

1. Confirm `version` in `audio2face/blender_manifest.toml` is the intended stable
   semantic version. After the first release, it must be newer than the manifest
   version at the release marked **Latest**. Merge the complete release source to
   the protected default branch.
2. In GitHub Actions, open **Release platform extensions**, keep the branch
   selector on the repository's default branch, choose **Run workflow**, and
   start the workflow. There are no workflow inputs.
3. The prepare job freezes the dispatch SHA, resolves the latest published
   release tag, requires that tag to be an ancestor, and requires a non-empty
   commit range. It also verifies the selected manifest version increased.
4. The prepare job runs the Python tests, creates `v<manifest version>` as a
   lightweight tag at the frozen SHA, and creates or verifies the corresponding
   draft. Generated notes start at the previous published release tag.
5. Each native job checks out the same frozen SHA, reclaims its standard runner,
   selects Python 3.11.9, and builds the locked runtime. Windows first installs
   and verifies the exact Visual Studio 2022 components listed above.
6. The native jobs download the official portable Blender 5.2.0 archives,
   verify their pinned SHA-256 digests, discard the downloaded archives after
   extraction, run Blender smoke tests, package the platform extensions,
   enforce GitHub's per-asset size limit, and upload directly to the draft.
7. The final job requires exactly the Windows and Linux asset names, sizes, and
   SHA-256 digests, confirms that the tag was not moved during the run, publishes
   the draft, and explicitly marks it as the latest release.

GitHub Releases requires every individual asset to be smaller than 2 GiB. The
workflow enforces that limit before upload and again before publication. If a
complete platform ZIP reaches that limit, it cannot be published as a
ready-to-install GitHub release asset; the runtime/package composition or
distribution host must be changed rather than splitting the Blender extension
ZIP.
