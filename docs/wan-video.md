# Wan Video

MLX-Gen supports Wan2.2 text-to-video and image-to-video through `mlxgen generate`. Use this page
for practical size, frame, and runtime guidance; use [API and CLI](api.md#wan-video) for the full
command surface.

## Current Practical Guidance

Wan A14B is the stronger default choice for local cinematic clips when you can accept a smaller
canvas. On an Apple M5 Max, a 5.05 second clip at `480x240` or `240x480`, `101` frames, `20` fps,
and `20` to `25` steps takes about 30 minutes in the local profiles below. For the starship prompt
shown here, the A14B text-to-video result at `480x240` is the preferred practical setting over
TI2V-5B at `832x480`.

TI2V-5B remains useful as the smaller 5B route and supports both text-to-video and first-frame
image-to-video. It uses 32-pixel spatial multiples and is designed around `1280x704` or
`704x1280`; `832x480` is a practical lower-cost size. A `1280x704`, `25` step, `101` frame local
run takes about the same time as the A14B `480x240` profile in this page.

Wan uses a flow-matching schedule shift. MLX-Gen uses the selected model's default unless you pass
`--flow-shift`: TI2V-5B defaults to `5.0` for native 720p-class runs, while A14B defaults to `3.0`.
For new 480p-class TI2V-5B checks such as `832x480`, use `--flow-shift 3`.

## Example Prompt

The comparison clips use this prompt:

```text
A cinematic wide-angle movie shot of a massive futuristic starship taking off from a frozen tundra. The ship features sleek dark metallic armor. Two massive warp nacelles pulsate with intensely glowing blue plasma. Violent snow squalls and heavy blizzards whip around the hull. The swirling snow is illuminated by stark volumetric blue light from the engines. The camera slowly tilts up. The camera simulates a violent shake as the thrusters ignite. Massive clouds of pristine white snow and ice blast away from the launch pad. Photorealistic, highly detailed, dramatic lighting.
```

## M5 Max Comparison Clips

| Model | Size | Steps | Frames / FPS | Approx. time on M5 Max | Asset |
| --- | ---: | ---: | ---: | ---: | --- |
| Wan2.2 TI2V-5B | `832x480` | 25 | 101 / 20 | 12 min | [MP4](assets/examples/wan-video-comparison/wan22-ti2v-5b-832x480-25steps-20fps-101frames.mp4) |
| Wan2.2 T2V-A14B | `480x240` | 25 | 101 / 20 | 30 min | [MP4](assets/examples/wan-video-comparison/wan22-t2v-14b-480x240-25steps-20fps-101frames.mp4) |
| Wan2.2 TI2V-5B | `1280x704` | 25 | 101 / 20 | 35 min | [MP4](assets/examples/wan-video-comparison/wan22-ti2v-5b-1280x704-25steps-20fps-101frames.mp4) |

### TI2V-5B At 832x480

![Wan2.2 TI2V-5B 832x480 frame strip](assets/examples/wan-video-comparison/wan22-ti2v-5b-832x480-frame-strip.jpg)

<video controls src="assets/examples/wan-video-comparison/wan22-ti2v-5b-832x480-25steps-20fps-101frames.mp4"></video>

### T2V-A14B At 480x240

![Wan2.2 T2V-A14B 480x240 frame strip](assets/examples/wan-video-comparison/wan22-t2v-14b-480x240-frame-strip.jpg)

<video controls src="assets/examples/wan-video-comparison/wan22-t2v-14b-480x240-25steps-20fps-101frames.mp4"></video>

### TI2V-5B At 1280x704

![Wan2.2 TI2V-5B 1280x704 frame strip](assets/examples/wan-video-comparison/wan22-ti2v-5b-1280x704-frame-strip.jpg)

<video controls src="assets/examples/wan-video-comparison/wan22-ti2v-5b-1280x704-25steps-20fps-101frames.mp4"></video>

## Command Shape

Use A14B T2V when the prompt does not need an input image:

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit \
  --prompt "A cinematic wide-angle movie shot of a massive futuristic starship taking off from a frozen tundra. The ship features sleek dark metallic armor. Two massive warp nacelles pulsate with intensely glowing blue plasma. Violent snow squalls and heavy blizzards whip around the hull. The swirling snow is illuminated by stark volumetric blue light from the engines. The camera slowly tilts up. The camera simulates a violent shake as the thrusters ignite. Massive clouds of pristine white snow and ice blast away from the launch pad. Photorealistic, highly detailed, dramatic lighting." \
  --width 480 \
  --height 240 \
  --frames 101 \
  --steps 25 \
  --guidance 4 \
  --guidance-2 3 \
  --fps 20 \
  --seed 42 \
  --output starship_takeoff_a14b.mp4
```

Use TI2V-5B when you want the 5B route or first-frame image-to-video route:

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit \
  --prompt "A cinematic wide-angle movie shot of a massive futuristic starship taking off from a frozen tundra. The ship features sleek dark metallic armor. Two massive warp nacelles pulsate with intensely glowing blue plasma. Violent snow squalls and heavy blizzards whip around the hull. The swirling snow is illuminated by stark volumetric blue light from the engines. The camera slowly tilts up. The camera simulates a violent shake as the thrusters ignite. Massive clouds of pristine white snow and ice blast away from the launch pad. Photorealistic, highly detailed, dramatic lighting." \
  --width 832 \
  --height 480 \
  --frames 101 \
  --steps 25 \
  --guidance 5 \
  --flow-shift 3 \
  --fps 20 \
  --seed 42 \
  --output starship_takeoff_ti2v5b.mp4
```

For native TI2V-5B runs at `1280x704` or `704x1280`, omit `--flow-shift` or pass `--flow-shift 5`.
For image-to-video, pass one `--image`. A14B I2V uses the separate
`AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` package; TI2V-5B uses the same TI2V package and
selects first-frame image-to-video when one image is supplied.
