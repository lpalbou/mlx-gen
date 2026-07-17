# Model Recommendations

Use this page when you want a conservative starting point for MLX-Gen model selection on Apple
Silicon Macs with `18 GB`, `24 GB`, `32 GB`, `64 GB`, or `128+ GB` unified memory.

These recommendations are based on published MLX-Gen validation profiles, not on package size
alone. When a benchmark includes full-process Darwin physical peak memory, that is the number used
for tiering. Otherwise this page falls back to the closest published peak RSS figure for the same
route. Larger canvases, more denoise steps, longer videos, and heavier editing modes can require
more memory than the example profile shown here.

Use the linked topic pages for the full command surface and benchmark details:

- [Quantization](quantization.md)
- [Image edit capabilities](edit-capabilities.md)
- [Image upscaling](upscaling.md)
- [Wan video](wan-video.md)

## Recommended By Memory Tier

| Memory tier | Text-to-image | Image-to-image and edit | Video generation | Upscaling and video restore |
| --- | --- | --- | --- | --- |
| `18 GB` | Start with `prism-ml/bonsai-image-ternary-4B-mlx-2bit`, `AbstractFramework/flux.2-klein-4b-8bit`, or `AbstractFramework/ernie-image-turbo-8bit`. | Use `ernie-image-turbo-8bit` for latent restyle and `flux.2-klein-4b-8bit` for lighter edit/reference work. Avoid Qwen edit and control routes here. | No conservative Wan recommendation from the published full-memory proofs. | `seedvr2-3b` and `seedvr2-7b` image upscaling packages fit comfortably. Treat published SeedVR2 video restore as a higher-memory workflow. |
| `24 GB` | Add `AbstractFramework/fibo-8bit`, `AbstractFramework/z-image-turbo-8bit`, and `AbstractFramework/qwen-image-2512-8bit` text-only runs. | `z-image-turbo-8bit` latent and native inpaint become practical. `flux.2-klein-4b-8bit` remains the lighter edit default. | No conservative Wan A14B recommendation since the 2026-06-12 q8 runtime-precision fix (A14B now needs BF16-class memory, ~33 GiB physical peak). | SeedVR2 image upscaling remains easy. The published five-second video-restore profile still needs more headroom than this tier. |
| `32 GB` | The 24 GB image recommendations remain good with more headroom. | `qwen-image-2512-8bit` text-only runs are comfortable. For edit-heavy work, `z-image-turbo-8bit` and `flux.2-klein-4b-8bit` are still the safer defaults than Qwen Edit. | Wan A14B runs at BF16-class memory since 2026-06-12 (~33 GiB physical peak at the small low-RAM profile), above this tier's conservative envelope. | `seedvr2-3b-8bit` published `29/8` five-second video restore is in range; `seedvr2-7b` video restore is still above this tier in the published profile. |
| `64 GB` | All image families above are practical. | Add `AbstractFramework/qwen-image-edit-2511-8bit` for strong layout-preserving edit/reference work, plus `AbstractFramework/qwen-image-8bit` control and control-inpaint. | Wan A14B (q8 storage or BF16; both run at BF16-class memory, ~33 GiB physical peak in the small low-RAM profile) is the first conservative tier for A14B video. | `seedvr2-7b` image upscaling is easy. For published video-restore proofs, keep using `seedvr2-3b` unless you have more than 64 GB free. |
| `128+ GB` | Everything above. | Everything above, plus room for larger or more exploratory runs. | Add native `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` high-resolution video profiles such as `1280x704, 17 frames, 20 steps`. | Add published `seedvr2-7b` five-second `29/8` video restore and larger Wan experimentation with real headroom. |

## Benchmark Basis

| Route | Model | Published profile | Peak memory used for recommendations |
| --- | --- | --- | --- |
| T2I | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | `512px`, `4` steps | `3.57 GiB` peak RSS |
| T2I | `AbstractFramework/flux.2-klein-4b-8bit` | `512px`, `4` steps | `9.23 GiB` peak RSS |
| T2I / latent I2I | `AbstractFramework/ernie-image-turbo-8bit` | `512px`, `8` steps | `12.9 GiB` peak RSS |
| T2I | `AbstractFramework/fibo-4bit` | `512px`, `8` steps | `11.39 GB` max RSS |
| T2I | `AbstractFramework/fibo-8bit` | `512px`, `8` steps | `15.89 GB` max RSS |
| T2I | `AbstractFramework/qwen-image-2512-8bit` | `768x336`, `4` steps | `10.73 GiB` peak RSS |
| Latent I2I | `AbstractFramework/z-image-turbo-8bit` | `768x432`, `9` steps | `11.49 GiB` peak RSS |
| Native inpaint | `AbstractFramework/z-image-turbo-8bit` | `768x432`, `9` steps | `10.57 GiB` peak RSS (improved by the 0.23.0 denoise dtype fix; pre-fix value was `18.11 GiB`) |
| Edit/reference I2I | `AbstractFramework/qwen-image-edit-2511-8bit` | `768x432`, `4` steps | `30.91 GiB` peak RSS |
| Control-inpaint | `AbstractFramework/qwen-image-8bit` | `768x432`, `4` steps | `34.94 GiB` peak RSS |
| Image upscale | `AbstractFramework/seedvr2-3b-8bit` | `5x` from `133x113` source | `4.73 GiB` max RSS |
| Image upscale | `AbstractFramework/seedvr2-7b-8bit` | `5x` from `133x113` source | `8.90 GiB` max RSS |
| Video restore | `ByteDance-Seed/SeedVR2-3B` | five-second `149` frame `29/8` published profile | `27.40 GB` max RSS |
| Video restore | `ByteDance-Seed/SeedVR2-7B` | five-second `149` frame `29/8` published profile | `66.18 GB` max RSS |
| T2V | `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` | `384x224`, `33` frames, `12` steps, low-RAM | `33.0 GiB` physical peak (BF16-class since the 2026-06-12 q8 runtime fix; pre-fix value was `20.7 GiB`) |
| I2V | `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` | `384x384`, `33` frames, `12` steps, low-RAM | `33.7 GiB` physical peak (BF16-class since the 2026-06-12 q8 runtime fix; pre-fix value was `21.5 GiB`) |
| T2V / I2V | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | `1280x704`, `17` frames, `20` steps | `103.7 GiB` physical peak |

## Practical Reading

- `18 GB` is enough for compact image families and SeedVR2 image upscaling, but not for the
  published full-memory Wan or SeedVR2 video profiles.
- `64 GB` is the first conservative tier for Wan A14B video: since the 2026-06-12 q8
  runtime-precision fix, A14B runs at BF16-class memory (~33 GiB physical peak even at the small
  low-RAM benchmark profile).
- `32 GB` is the first tier where the published five-second `SeedVR2 3B` video-restore profile fits
  the published memory envelope.
- `64 GB` is the first tier where Qwen edit and Qwen control routes become practical without
  running at the ceiling.
- `128+ GB` is where the current native TI2V-5B high-resolution benchmark belongs.

## Not Yet Published As Memory Recommendations

MLX-Gen has additional validated quality surfaces that are not promoted here because the current
docs do not yet include a strong enough route-specific memory benchmark for a conservative public
recommendation. That includes `FLUX.2 Klein 9B`, `Qwen` q4 edit packages, and broader `Z-Image`
q4/q8 route combinations outside the measured rows above.
