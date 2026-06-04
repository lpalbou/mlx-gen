# FAQ

## Where Is The Save Command?

Use `mlxgen prepare`.

`prepare` is the public MLX-Gen workflow for creating a reusable local model folder. It can quantize the model with `--quantize` and writes a Hugging Face `README.md` model card into the prepared folder.

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

New MLX-Gen integrations should not call a separate save workflow; use `mlxgen prepare`.

## What Is The Difference Between Download And Prepare?

`mlxgen download` populates the local Hugging Face cache with source files. It does not create a separate model folder and does not write a model card.

`mlxgen prepare` creates a reusable local MLX-Gen folder at `--path`. It can quantize weights and writes a generated Hugging Face model card.

## Can Generate Prepare A Model Folder?

No. `mlxgen generate` is only for inference. It does not accept `--path` for model preparation.

To create a model folder, use:

```sh
mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4
```

To choose the output image path during generation, use `--output`.

## Does Generation Download Missing Files?

No. Generation and ordinary Python model construction are cache-only by default. Missing artifacts raise `DownloadRequiredError` with the exact `mlxgen download` or `mlxgen prepare` command to run.

## Is `HF_HUB_ENABLE_HF_TRANSFER=1` Required?

No. It is optional acceleration for explicit Hugging Face downloads and prepare operations. `mlxgen download` and `mlxgen prepare` already authorize network access.

## Can Prepared Folders Load In Diffusers Or Transformers?

No. Prepared MLX-Gen folders use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not direct Diffusers or Transformers `from_pretrained()` loading.

## Can I Quantize ERNIE Image Turbo?

Yes. ERNIE Image Turbo supports q8 and q4 prepared folders:

```sh
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-8bit --quantize 8
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-4bit --quantize 4
```

ERNIE q4 uses a model-specific mixed q4/q8 policy. Fully q4 ERNIE checkpoints can drift from BF16/q8 behavior, so MLX-Gen keeps Mistral3 text linears plus selected ERNIE transformer attention-output and conditioning paths at q8.

## Can I Prepare Or Quantize Bonsai Image?

No. Bonsai Image repositories from Prism are already packed MLX artifacts. Use `download` and
generate directly:

```sh
mlxgen download --model prism-ml/bonsai-image-ternary-4B-mlx-2bit

mlxgen generate \
  --model prism-ml/bonsai-image-ternary-4B-mlx-2bit \
  --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --guidance 1 \
  --output bonsai.png
```

The ternary 2-bit checkpoint is supported. The binary 1-bit checkpoint is detected and rejected
with an explicit unsupported-runtime message until stock MLX can execute the required 1-bit packed
affine matmul. The latest published stock MLX checked for the 0.18.7 release was 0.31.2, and it
still rejected `bits=1`.

## Does ERNIE Image Turbo Support Image Input Or Prompt Enhancer?

Prompt Enhancer is supported for ERNIE Image Turbo when the full source snapshot is available:

```sh
mlxgen download --model baidu/ERNIE-Image-Turbo --all-files

mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A ceramic mug" \
  --use-prompt-enhancer
```

ERNIE Image Turbo supports experimental single-image image-to-image in MLX-Gen:

```sh
mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --image input.png \
  --prompt "Turn the scene into a pencil sketch" \
  --width 512 \
  --height 512 \
  --steps 8 \
  --guidance 3 \
  --image-strength 0.25 \
  --output edited.png
```

Multi-image edit is not supported for ERNIE. `--image-strength` follows the MLX-Gen image-influence convention: higher values preserve more of the init image, while lower positive values allow more transformation.

Prepared ERNIE q8/q4 folders do not bundle Prompt Enhancer files; use the full source snapshot path or the Hugging Face repo after `mlxgen download --all-files` when you need `--use-prompt-enhancer`.

## How Should I Prompt Wan Image-To-Video?

Wan image-to-video responds best when the input image and prompt agree on a plausible motion path.
Use a source frame with the whole subject visible, enough margin around moving limbs or objects, and
minimal occlusion for body parts that need to move. Match the source image's aspect ratio and
composition to the requested video canvas instead of stretching a portrait source into a landscape
video or the reverse. TI2V-5B normalizes width and height to multiples of 32, so compose the source
image for the adjusted dimensions when the requested size is rounded.

Keep the main subject inside the rendered frame for the whole intended motion. If a face, hand,
foot, product edge, or other identity-critical region leaves the frame and later re-enters, the
model may reconstruct it inconsistently. Once that region leaves the visible image, the model no
longer has rendered pixels to anchor its identity or geometry. When it comes back, it is effectively
reconstructed from latent and temporal context, so faces, hands, limbs, logos, edges, and other
details can drift or mutate. For human motion, a pose that already suggests the intended action is
usually more reliable than a neutral pose, but avoid edge-reaching poses unless the prompt keeps the
camera wide enough to preserve head, hands, and feet.

For faces and other front-facing identity cues, also constrain rotation. If a person turns far enough
that the face is no longer visible, the model may reassign the back of the head, hair, shoulders, or
clothing as the new front when the subject turns back. When face continuity matters, ask for
front-facing or three-quarter-front motion, keep torso pivots below about 60-90 degrees, and block
rear views in the negative prompt. This usually improves identity stability, but it can make motion
more restrained.

Write the positive prompt as a concrete motion plan instead of a general style request. Name the
subject, camera style, body parts or object parts that should move, and the continuity constraints
that should remain stable:

```text
Cinematic 5 second full-body motion video of the adult athlete performing controlled lateral steps:
torso pivots, head turns, arms sweeping naturally from high and low positions, legs crossing and
uncrossing, weight shifting forward and backward, knees bending and straightening, full head visible,
full body visible, arms attached naturally at shoulders and wrists, natural hands and feet,
consistent outfit, smooth studio camera.
```

For object motion, use the same pattern:

```text
Cinematic 5 second takeoff video of the spacecraft clearly lifting away from the snowy landing
field, landing gear leaving the snow, bright engines firing underneath, snow and ice blowing
outward, stable hull geometry, consistent metal panels, smooth upward camera tracking, no scene cut.
```

Use the negative prompt to block common failure modes for the subject and motion:

```text
static still image, only arms moving, only camera movement, cropped head, cropped feet, hands out of
frame, back to camera, rear view, turned away, profile-only view, over-rotated body, detached arm,
disconnected arm, detached hand, broken wrist, extra limbs, malformed hands, oversized foot, melted
foot, deformed face, duplicate body, black frames, green frames, low quality, flicker, subject exits
frame, sudden scene cut
```

Prompting reduces but does not eliminate video-model brittleness. Complex human motion can still
degrade around hands, wrists, feet, ankles, self-occluding poses, and out-of-frame re-entry. To
reduce those failures, make the subject smaller in the source frame, leave generous margins above
raised hands and below feet, choose poses whose full motion stays inside the frame, ask for a wide
or camera-follow shot, keep motion restrained near boundaries, constrain subject rotation when face
identity matters, and shorten clips when identity details approach the edge. If the action requires
the subject to leave the frame, turn away fully, or return from an occluded/back-facing pose, split it
into separate clips or use a later keyframe/image input rather than relying on one long single-image
conditioning run. For release or production checks, use seed sweeps and inspect decoded frames or
contact sheets rather than relying only on MP4 existence.

## Why Do Some Imports Or Paths Still Say `mflux`?

MLX-Gen is currently built on the mflux codebase. Some internal modules and compatibility entry points still use `mflux.*` names while the public package and command surface evolve under `mlx-gen` and `mlxgen`.

## How Does This Relate To AbstractVision?

MLX-Gen is intended to be the Apple Silicon / MLX backend dependency for AbstractVision. AbstractVision remains a cross-platform orchestration layer, while MLX-Gen owns MLX model loading, quantized local formats, and Apple Silicon runtime behavior.
