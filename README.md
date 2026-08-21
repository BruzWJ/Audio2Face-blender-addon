# Audio2Face Blender Addon

Audio2Face-blender-addon is a Blender 5.2 extension that runs NVIDIA Audio2Face-3D
v3.0 and its Audio2Emotion v3.0 driver locally on an NVIDIA GPU. Blender
launches one add-on-managed native child process; users do not host a service
or select an executable, SDK, model, CUDA installation, or working directory.

The worker generates one fixed `arkit-52/1` value stream from either a complete
selected WAV or incremental mono float32 PCM. Blender writes those values to
existing, exact-name Shape Key `value` properties on every enabled target
mesh. Vertex data, Actions, F-curves, and bones are never modified.

## Requirements

- Blender 5.2.x on Linux x64 or Windows x64
- A supported NVIDIA GPU and display driver
- Blender Online Access while installing the managed runtime
- Disk space for the runtime, Audio2Face v3.0 and Audio2Emotion v3.0 model
  inputs, and their locally optimized TensorRT engines

## Workflow

1. Install and enable the extension.
2. Accept the applicable NVIDIA terms and click **Install Runtime & Models**.
3. Choose **Selected WAV** or **Stream**, then select a WAV file.
4. Select one or more meshes whose Shape Keys use canonical ARKit names such as
   `EyeBlinkLeft`, `JawOpen`, and `MouthSmileRight`, then click **Add Selected**.
5. Click **Start Worker**. The extension starts its bundled process, performs
   the protocol handshake, and loads the managed Audio2Face v3.0 and
   Audio2Emotion v3.0 models.
6. Adjust the face and emotion controls. Leave **Auto Audio2Emotion** off to
   use the model-defined manual emotion driver, or enable it to replace that
   driver with emotions inferred from the input audio.
7. In **Selected WAV**, click **Generate ARKit Values**, then **Play Selected
   Audio** to drive the targets from the completed result.
8. In **Stream**, click **Start WAV Stream** to decode, resample, and submit the
   file as bounded PCM chunks. Playback starts after the first model frames are
   buffered and drives the targets incrementally. **Stop Stream** ends only the
   active stream and keeps the model ready.
9. Click the worker **Stop** button to exit the child process and release its
   model and CUDA resources.

Installing or enabling the extension does not start the worker. Loading the
model does not start continuous inference. GPU inference runs only for an
explicit selected-file job or an active PCM stream. Buffered selected-file
playback can continue after the worker has stopped.

## Target meshes

Each target is an enabled Blender mesh object. A Shape Key is connected only
when its name exactly matches one of the 52 PascalCase names in
[`audio2face/arkit.py`](audio2face/arkit.py). Matching is case-sensitive.
There are no aliases, destination selectors, multipliers, or offsets.

A target may implement any exact-name subset of the 52 channels. The worker
always produces the complete ordered stream; Blender updates the matching keys
that exist on each target. Objects sharing one Shape Key datablock are updated
once per preview sample. Because Shape Key values belong to that shared
datablock, linked objects reflect the same values even when only one is a
target; make the mesh data single-user when independent motion is required.

## Input modes and controls

**Selected WAV** reads the complete file in the native worker, writes one
strict timestamped result, binds that result to the exact submitted WAV, and
plays it against Blender's audio-device clock.

**Stream** has an explicit `stream_start` / chunk / `stream_end` lifecycle. The
built-in source incrementally decodes a selected WAV to mono f32le, resamples
it to the model rate, and keeps a bounded lead over local playback. Integrations
running inside Blender can instead use
[`audio2face/streaming.py`](audio2face/streaming.py) to push source-agnostic
live PCM at the model rate. That integration owns capture and audible monitoring;
the built-in streamed-WAV source performs Blender audio playback itself. No port,
network listener, or separately hosted service is involved.

After `start_pcm_stream(scene)`, an integration polls
`get_pcm_stream_requirements(scene)` until it returns
`(model_sample_rate_hz, required_initial_lead_samples)`. Both functions default
to `bpy.context.scene`. The source must submit PCM at that rate and queue at
least the required lead before starting synchronized audible monitoring. The
requirements function returns `None` while `stream_start` is still pending, so
integrations never hard-code either model-dependent value.

The v3 diffusion model has inherent audio lookahead. The worker reports the
model-rate audio lead needed before its first Audio2Face frame can be ready.
With Auto emotion enabled, that requirement is the larger of the Audio2Face
and Audio2Emotion readiness windows. The built-in streamed-WAV source satisfies
that requirement before synchronized playback begins.

The face controls are:

- input strength;
- lower and upper face smoothing and strength;
- face-mask level and softness;
- skin and blink strength;
- blink, eyelid-open, and lip-open offsets.

The emotion driver exposes every emotion channel reported by the loaded
Audio2Face model. With **Auto Audio2Emotion** disabled, their manual values form
one constant conditioning vector for the operation. The sliders begin at the
model's defaults and are preserved by name across model reloads. With the
toggle enabled, Audio2Emotion v3.0 analyzes the same selected or streamed audio
and replaces the manual vector with timestamped values. Its controls are
strength, contrast, temporal smoothing, transition time, and the maximum number
of simultaneous emotions. Auto is a full override: manual values are ignored,
and preferred-emotion mixing is disabled.

The same complete, frozen face-and-emotion settings document is used by both
modes. A live stream must be restarted to apply changed controls. The worker's
stream prebuffer always covers Audio2Face readiness and additionally covers
Audio2Emotion readiness when Auto is enabled.

The model's canonical skin solver already contains `TongueOut`; its separate
16-pose tongue-detail solver and geometry-only controls are intentionally not
loaded into this ARKit-52 path.

## Managed runtime

The add-on catalog pins one immutable HTTPS archive for each supported
platform. **Install Runtime & Models** downloads to temporary storage, verifies
that the final URL remains credential-free HTTPS, verifies size and SHA-256,
safely extracts the fixed bundle, builds both `network.trt` engines for the
local GPU, and atomically activates the completed runtime below Blender's
extension user-data directory.

The archive must contain the production worker, Audio2X runtime, reviewed CUDA
and TensorRT user-mode libraries, release-built NVIDIA TensorRT `trtexec`, both
model input sets, and required licenses and notices. Separate `network.trt`
engines are built for Audio2Face and Audio2Emotion. The NVIDIA display driver
remains a system requirement. Start accepts only the catalog-pinned
installation and its verified receipt. No external host application, hosted
service, or user-selected executable is used.

### Release status

The catalog in this source checkout contains no platform artifacts. The source
extension and worker architecture are implemented, but the install button
cannot deliver a runtime until license-reviewed Linux and Windows archives are
published at immutable HTTPS URLs and their measured sizes and SHA-256 digests
are added to [`runtime_catalog.json`](audio2face/runtime_catalog.json).
Audio2Emotion v3.0 is gated by NVIDIA on Hugging Face, and this integration is
experimental until its GPU, platform, model-access, and license requirements
have passed release validation. That validation must also confirm that the
pinned Audio2Emotion post-processed vector ordering matches the pinned
Audio2Face emotion ordering; SDK 1.0.0 exposes the resulting vector width but
not output channel names. The project neither embeds access credentials nor
assumes model redistribution permission.

## ARKit output

The Audio2Face SDK exposes the model's 52 skin channels in lowerCamelCase. The
worker requires that exact unique set, resolves each channel by name, and emits
the fixed PascalCase `arkit-52/1` order. Six SDK eye-rotation components resolve
into the eight `EyeLook*` values. Raw geometry, jaw transforms, and eye rotations
are not serialized. Selected mode stores those frames in `a2f-animation/1`;
Stream mode emits the same ordered 52 values as incremental frame events and
does not create a result or animation datablock.

Every coefficient is finite and within `[0.0, 1.0]`. Frame timestamps remain
integer audio-sample positions, so playback is synchronized to Blender's audio
clock and does not depend on scene FPS.

See [architecture](docs/architecture.md), [protocol](docs/protocol.md), and the
[worker build guide](worker/README.md) for the exact contracts.

## Package and verify

Using Blender 5.2:

```sh
blender --command extension validate audio2face
blender --command extension build --source-dir audio2face --output-dir dist
blender --factory-startup --background --python tests/blender_smoke.py
```

Run the Python suite with:

```sh
python3 -m pytest -q
```

A production release must additionally pass real NVIDIA GPU inference,
dependency, install, cancellation, and shutdown tests for every published
platform archive.

## Licensing

The add-on and worker source are GPL-3.0-or-later. NVIDIA runtime components and
model files remain under their applicable terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
