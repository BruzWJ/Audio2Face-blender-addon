# Audio2Face Blender Addon

Audio2Face is a Blender 5.2 extension that runs NVIDIA Audio2Face-3D v3.0 and
Audio2Emotion v3.0 locally on an NVIDIA GPU. Blender owns one managed native
worker process. Users do not host a service or choose an executable, SDK,
model, CUDA installation, or working directory.

The extension produces 52-channel ARKit coefficients from a selected WAV or
incremental mono float32 PCM. It drives existing Shape Key `value`
properties on enabled mesh targets. It does not write vertices, bones,
Actions, F-curves, or baked animation.

## Requirements

- Blender 5.2.x on Linux x64 or Windows x64
- A supported NVIDIA GPU and display driver
- Blender Online Access during managed installation
- Space for the runtime, both model inputs, and two GPU-specific TensorRT
  engines

## Runtime setup

Runtime setup lives in **Edit > Preferences > Add-ons > Audio2Face**. The setup
box contains:

- one [NVIDIA terms](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
  link and one acceptance checkbox;
- source buttons for
  [Audio2Face-3D v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) and
  [Audio2Emotion v3.0](https://huggingface.co/nvidia/Audio2Emotion-v3.0); and
- **Install Runtime & Models**.

The install action downloads one catalog-pinned artifact containing the worker,
runtime libraries, model inputs for both models, TensorRT engine builder, and
notices. It then builds both model engines for the local GPU. The source
buttons identify the model sources; they are not separate install steps. The
3D View sidebar only reports runtime readiness and directs setup to Add-on
Preferences.

The checked-in [`runtime_catalog.json`](audio2face/runtime_catalog.json) has no
published artifacts. **Install Runtime & Models** therefore remains disabled
in this source release. It becomes available only after release maintainers
publish reviewed Linux and Windows archives with immutable HTTPS URLs, exact
sizes, and SHA-256 digests. No URL, checksum, or model payload is inferred.

The top of Add-on Preferences includes the same right-aligned **Uninstall**
action used for Blender 5.2 Legacy (User) add-ons. It opens Blender's two-line
**Remove Add-on** confirmation with the add-on name and installed package path.
After confirmation it delegates to Blender's extension uninstaller, which
first disables the add-on and stops its worker, streams, and playback, then
removes the extension and its complete managed data directory: runtime
libraries, both models and TensorRT engines, installer leftovers, logs, and
generated results. Selected WAV files, `.blend` files, meshes, and shared
NVIDIA driver caches are not managed by the add-on and are not removed.

## Workflow

1. Install and enable the extension, then complete **Runtime setup** in its
   Add-on Preferences.
2. In the Audio2Face sidebar, choose **Selected WAV** or **Stream**. The
   **Audio Playback** controls appear immediately below this mode selector.
3. Choose a WAV for complete generation or for the built-in streamed-WAV
   source. A Blender integration can use Stream mode without a WAV by pushing
   live mono f32le PCM through [`audio2face.streaming`](audio2face/streaming.py).
4. Select any mesh objects and click **Add Selected Meshes**. Shape Keys are
   not required when a mesh is added.
5. Click **Start Worker**. Blender launches the verified managed worker,
   negotiates the protocol, and loads both managed models.
6. Choose a model identity and adjust the controls reported by that model.
7. Leave **Auto Audio2Emotion** off to use the manual emotion values. Enable it
   to replace the manual driver with emotions inferred from the same audio.
8. In Selected WAV mode, click **Generate ARKit Values**, then **Play Result**.
   In Stream mode, click **Start WAV Stream** or submit live PCM through the
   integration API.
9. **Stop Stream** ends only the active stream and keeps the loaded model
   ready. **Stop Worker** exits the child process and releases its model and
   CUDA resources.

Installing or enabling the extension does not start the worker. Loading the
models does not start continuous inference. GPU inference runs only for an
explicit generation job or active PCM stream. A completed Selected WAV result
can play after the worker has stopped.

## Mesh targets and channel delivery

Every enabled target mesh subscribes to the model channel stream, with no
Shape Key admission check. At each frame, Blender uses each exact
model-provided lowerCamel channel name to look up a Shape Key on each target.
If that key exists, its value is assigned; if it does not, that channel is
skipped for that target. Names are never translated or remapped, and there is
no per-target multiplier, bake step, or direct mesh deformation.

The loaded model supplies the exact ordered 52-channel list. A target can
contain all, some, or none of those Shape Keys. Several objects may share one
Shape Key datablock; delivery writes that shared datablock once per frame, so
linked objects reflect the same values. Use single-user mesh data when objects
need independent values.

## Audio modes and playback

**Selected WAV** sends one complete RIFF/WAVE file to the worker. The worker
decodes, downmixes, resamples, generates all frames, and atomically publishes a
strict result. **Audio Playback** uses Blender's audio-device clock to sample
that result and provides play, pause/resume, stop, loop, volume, and
reset-on-stop controls.

**Stream** uses one `stream_start` / chunk / `stream_end` lifecycle. The
built-in source incrementally decodes a selected WAV, resamples it to the model
rate, and keeps a bounded lead over Blender audio playback. Integrations can
instead call `start_pcm_stream`, poll `get_pcm_stream_requirements`, and submit
model-rate mono f32le chunks. Those integrations own capture, resampling, and
audible monitoring. No port or network listener is opened.

The worker reports the initial model-rate `prebuffer_samples` requirement.
With automatic emotion enabled, it covers both Audio2Face and Audio2Emotion
readiness. Streamed frames are buffered and sampled against the local audio or
presentation clock; scene FPS is not used for synchronization.

## Model-driven controls

`load_model` returns a self-describing `model_schema` with exactly
`identities`, `channels`, `parameters`, and `emotion_channels`. Blender builds
its selectors and numeric controls from those values. `parameters` is one
object mapping opaque worker paths to numeric defaults; JSON integer and float
types select the corresponding Blender control. Labels and groups are derived
mechanically from path segments instead of duplicating UI metadata.

NVIDIA SDK 1.0 exposes parameter structures but no parameter reflection. The
worker therefore contains the one typed path-to-member adapter; Blender owns
no parameter list. Defaults, identities, emotion channels, and output channels
come from the loaded SDK/model. Internal graph nodes and tensors are not
controls. Both input modes submit one exact settings object:

```json
{
  "auto_audio2emotion": false,
  "manual_emotions": {"<model emotion name>": 0.0},
  "parameters": {"/advertised/path": 0.0}
}
```

Manual emotion values are a constant, model-shaped conditioning vector. When
`auto_audio2emotion` is true, Audio2Emotion analyzes the same input and fully
replaces that vector. Manual values are ignored in automatic mode. The same
semantics and complete frozen settings apply to Selected WAV and Stream; a
stream must be restarted to apply changed controls.

## Output contract

The worker reports the model's exact 52 unique lowerCamel channel names in
model order. It resolves eye-look values into the corresponding
model-provided channel slots without reordering the list. Raw geometry, jaw
transforms, eye rotations, and other solver data are not serialized.

Selected mode stores `a2f-animation/2` with exactly `schema`, `job_id`,
`sample_rate`, `channels`, `timestamps_samples`, and `weights`.
Stream mode uses the same negotiated channel order for incremental frames and
does not create a result file. Coefficients are finite and within `[0.0, 1.0]`;
timestamps are integer audio-sample positions.

See [architecture](docs/architecture.md), [protocol](docs/protocol.md), and the
[worker build guide](worker/README.md) for the full contracts.

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

A production release must also pass real NVIDIA GPU inference, dependency,
installation, cancellation, and shutdown tests for every published artifact.

## Licensing

The extension and worker source are GPL-3.0-or-later. NVIDIA runtime components
and model files remain under their applicable terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
