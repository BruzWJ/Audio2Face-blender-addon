# Release automation

`.github/workflows/release.yml` is the only GitHub release path. It runs only
when started manually from GitHub Actions. The branch selected in the **Run
workflow** interface must be the repository's default branch; there are no
workflow inputs. The workflow freezes that branch's dispatch commit, compares
it with the published release marked **Latest**, and generates the current UTC
date as a stable semantic version. After the source tests pass, it commits that
version to `audio2face/blender_manifest.toml`, freezes the resulting commit, and
uses the same calendar identity for the GitHub release. For August 21, 2026, the
generated manifest version, tag, and assets are:

```text
2026.8.21
v2026.8.21
audio2face-2026.8.21-windows-x64.zip
audio2face-2026.8.21-linux-x64.zip
```

Calendar components are deliberately not zero-padded. Blender requires the
manifest `version` to follow semantic versioning, whose numeric identifiers
cannot contain leading zeroes. The Git tag is exactly the stamped manifest
version with a `v` prefix. One published release is allowed per UTC date; a
rerun can reuse that date's exact manifest commit and draft after a failed run.

The latest published release tag must resolve to an ancestor of the selected
commit, and the range from that tag to the selected commit must contain at least
one commit. The generated calendar version must also be newer than the version
embedded at the latest release tag. Before committing the dated manifest, the
workflow verifies that the default branch still identifies the dispatch commit.
It updates the branch atomically only when the expected dispatch commit is still
its head. When a version commit is needed, it is a direct child that changes
only the manifest; same-date retries verify and reuse that exact child instead
of creating another commit.

GitHub gives unpublished releases an internal `untagged-*` web URL. That is
normal and is not the release tag. The workflow captures the numeric release ID
from the Create Release API response, or resolves the ID through GraphQL when it
reuses an existing draft. Reuse requires an exact date, title, and target commit,
and every draft upload uses that immutable ID. The workflow refuses an
already-published release or an unrelated pending draft. The final job publishes
only after the draft contains
exactly the two expected assets with the SHA-256 digests and byte sizes reported
by their native build jobs. It then creates the dated Git tag at the frozen
manifest commit, publishes the draft, and explicitly marks the release as
**Latest**.

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

- `Microsoft.VisualStudio.Component.VC.CoreBuildTools`;
- `Microsoft.VisualStudio.Component.VC.14.43.17.13.x86.x64`; and
- `Microsoft.VisualStudio.Component.Windows11SDK.22621`.

It then verifies the x64 build-environment script, MSVC toolset `14.43.34808`,
and Windows SDK `10.0.22621.0` before invoking the runtime builder. The builder
includes error-free, registered, reboot-pending Visual Studio Setup instances in
discovery because installer exit code `3010` is accepted, but still selects only
an instance containing both `vcvars64.bat` and the locked compiler. The Ubuntu
image supplies Git, `curl`, `tar`, `xz`, GitHub CLI, and Docker Engine; the locked
Rocky Linux producer container supplies the compiler and system headers.

Neither runner needs an NVIDIA GPU, CUDA Toolkit, TensorRT installation,
Audio2Face installation, or either model repository. The runtime builder
downloads and verifies every locked development/runtime input, and the model
files are deliberately not part of the extension package.

## Release lifecycle

1. Merge the complete release source to the repository's default branch. Do not
   edit the manifest for the release date; the workflow owns that version stamp.
2. In GitHub Actions, open **Release platform extensions**, keep the branch
   selector on the repository's default branch, choose **Run workflow**, and
   start the workflow. There are no workflow inputs.
3. The prepare job freezes the dispatch SHA, resolves the latest published
   release tag, requires that tag to be an ancestor, and requires a non-empty
   commit range. It generates `YYYY.M.D` from the current UTC date and verifies
   that version is newer than the manifest embedded in the latest release.
4. The prepare job tests the source with the dated manifest, confirms the branch
   has not moved, atomically commits only the manifest when needed, and freezes
   the stamped source SHA. A rerun verifies and reuses the exact stamp if it
   already exists. The job then creates or verifies the dated draft and retains
   its numeric ID. GitHub generates the notes from the previous published
   release.
5. Each native job checks out the same frozen SHA, reclaims its standard runner,
   selects Python 3.11.9, and builds the locked runtime. Windows first installs
   and verifies the exact Visual Studio 2022 components listed above.
6. The native jobs download the official portable Blender 5.2.0 archives,
   verify their pinned SHA-256 digests, discard the downloaded archives after
   extraction, run Blender smoke tests, package the platform extensions,
   enforce GitHub's per-asset size limit, and upload directly to the draft by ID.
7. The final job requires exactly the Windows and Linux asset names, sizes, and
   SHA-256 digests, atomically creates the lightweight date tag or verifies an
   exact existing one, publishes the draft, verifies the tag again, and confirms
   that the release is marked **Latest**.

GitHub Releases requires every individual asset to be smaller than 2 GiB. The
workflow enforces that limit before upload and again before publication. If a
complete platform ZIP reaches that limit, it cannot be published as a
ready-to-install GitHub release asset; the runtime/package composition or
distribution host must be changed rather than splitting the Blender extension
ZIP.
