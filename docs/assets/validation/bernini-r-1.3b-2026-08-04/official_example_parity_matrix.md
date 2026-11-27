# Bernini-R 1.3B official example parity matrix

## Metadata

- Created: 2026-08-10
- Scope date: 2026-08-12
- Scope: official public `ByteDance/Bernini-R-1.3B-Diffusers` examples only
- Official source revision: `2d2b4591ac053ec25c6371b01a5a6746679e5793`

## 2026-08-12 update: the judging bar itself was re-based

Two findings from 2026-08-11/12 change how the open rows must be judged:

1. **The retained upstream example clips are almost certainly 14B output, not
   1.3B.** The upstream helper scripts default to the 14B
   `Bernini-R-Diffusers` checkpoint for every testcase, and the upstream
   README states the 1.3B "performs close to the 14B variant on simple
   tasks... while lagging behind on more complex tasks such as human
   generation" — exactly the class of the open rows. No public artifact
   attests a 1.3B origin for any retained clip.
2. **A local ground-truth oracle now exists.** The official ByteDance
   inference code was run locally with the official 1.3B checkpoint
   (`untracked/oracle/oracle_run.py`, CPU float32; see
   `validation_outputs/bernini_r_1_3b_2026_08_11/oracle_official_1_3b/README.md`
   for validity analysis, including why the torch-MPS path is excluded).
   With bit-identical initial noise, the oracle and mlx-gen produce
   near-identical trajectories where the model succeeds (ads2v) and
   identical failure modes where it does not (mv2v reduced-domain collapse,
   v2v_case3 non-restructure, r2v_case2 reference-binding loss).

Open rows are therefore judged as: *does mlx-gen match or exceed what the
official 1.3B implementation itself produces at identical settings, and does
the official task outcome hold at the profile this host can verify?* Matching
the retained (14B-class) clips pixel-for-pixel is explicitly not the bar.

## What this matrix means

This matrix defines the stricter target the current Bernini work should be judged against:

- use the public `ByteDance/Bernini-R-1.3B-Diffusers` checkpoint;
- use the public official Bernini code and bundled public case files;
- use the official example inputs without local workaround assets;
- judge success qualitatively on the official task outcome, not on pixel identity.

This is narrower than "general-case Bernini is product-ready" and stricter than "some custom local
Bernini runs looked interesting." Accepted rows below means the current local MLX output was
manually judged qualitatively good against the public official example, not that the output is
pixel-identical.

## Official public example inventory

The public 1.3B model card, testcase tree, and helper scripts expose these example rows:

| Row | Upstream case | Upstream notes | Current mlx-gen status | Immediate next action |
| --- | --- | --- | --- | --- |
| T2I | `assets/testcases/t2i/t2i.json` | single GPU, `--num_frames 1` | Accepted | Keep as accepted current parity evidence; public CLI surfacing is optional follow-up. |
| I2I | `assets/testcases/i2i/i2i.json` | single GPU, `--num_frames 1` | Accepted with minor wall-texture caveat | Keep as accepted current parity evidence; public CLI surfacing is optional follow-up. |
| T2V | `assets/testcases/t2v/t2v.json` | official multi-GPU renderer-only text-to-video example | Accepted | Keep as accepted current parity evidence; public CLI surfacing is optional follow-up. |
| V2V case 1 | `assets/testcases/v2v/v2v_case1.json` | add a snowman | Accepted (2026-08-11 review) | [v2v_case1 proof](../bernini-r-1.3b-2026-08-11/v2v_case1/README.md). |
| V2V case 2 / MV2V | `assets/testcases/v2v/v2v_case2.json` | make the person crouch down | Accepted with three named caveats (2026-08-12) | Evidence: `head_canvasfix_mv2v_full_v2` full-profile run + `mv2v_reduced_cpu_f32` oracle parity. See the row notes below. |
| V2V case 3 | `assets/testcases/v2v/v2v_case3.json` | replace humanoid robot with robotic dog | Official recipe fails at 1.3B in both implementations (oracle-proven); outcome achieved via the documented mv2v-prefix recipe, recipe-parity-proven | With the official recipe (task_type `v2v`) the humanoid is never restructured — dog-headed biped at guidance 4.0 (mlx full: `head_v2v_case3_full_v2/`; oracle at two profiles) AND at guidance 5.0 (ablation `exp_v2v_case3_v2vprefix_g5/`). The mv2v task prefix + guidance 5.0 produces a full, stable quadruped robotic dog in BOTH implementations: mlx-gen at full profile (`exp_v2v_case3_mv2vprefix/`) and the official-1.3B oracle at mid profile (`oracle_official_1_3b/v2v_case3_mv2vprefix_mid480_cpu_f32/`). Ablation attributes the restructure to the mv2v prefix, not the guidance. Delta vs the retained clip: mostly-stationary pose sequence rather than walking. |
| R2V case 1 | `assets/testcases/r2v/r2v.json` | five-reference statue-on-bench case | Accepted | Keep as accepted current parity evidence. |
| R2V case 2 | `assets/testcases/r2v/r2v_case2.json` | eight-reference statue-with-cup case | Partial at the official recipe, model-limited (oracle-proven); tuned recipe recovers substantially more | At the official recipe the full-profile rerun (`head_r2v_case2_full_v2`) binds the statue, bench scene, floral shorts, and cup but not the headphones or tee; the official-1.3B oracle binds almost none of the eight references at matched settings (reduced AND mid profiles: `oracle_official_1_3b/r2v_case2_{reduced,mid480_49f}_cpu_f32/`), and the MLX twins reproduce or exceed the oracle at both, so the shortfall is the model's binding ceiling, not an mlx-gen defect. A tuned non-official recipe (reference guidance 6.0, demonstrated at seed 43: `exp_r2v_case2_full_refg6_s43/`) recovers substantially more: stable pink cat-ear elements (not full headphones — no band or ear cups), legible "bernini" chest branding (not a black fabric tee), and the cup with drinking motion (cup absent in early frames). Binding is strongly guidance- and seed-sensitive. Note: the upstream prompt is 571+ UMT5 tokens truncated to 512 on both engines. |
| RV2V case 1 | `assets/testcases/rv2v/rv2v_case1.json` | garment replacement | Accepted | Keep as accepted current parity evidence; shirt closure differs slightly from the official clip, but the garment transfer and body fit now read correctly. |
| ADS2V case 1 | `assets/testcases/rv2v/rv2v_case2.json` | two-video insertion at `121` frames, `24` fps, `1280` max image size | Accepted at mid profile; full official profile unverifiable on this host | Implementation parity is proven at matched settings (the reduced MLX twin tracks the CPU oracle nearly frame-for-frame after the 2026-08-12 canvas fix — reference videos now conditioned with the official long-edge cap instead of an area target). The mid-profile run (`head_ads2v_mid480_61f`, 480 cap/61f/24fps/40 steps) demonstrates the full task outcome: the inserted ad plays on the laptop screen and resolves into the "Bernini" logo while the surrounding scene stays intact. The official `1280x672/121f` recipe remains computationally intractable on this host for either engine and is noted as unverified scope, not a failure. |

## Current gap, stated plainly

As of Wednesday, August 12, 2026:

- 7 official public rows are accepted on current local review:
  - `i2i`, `t2i`, `t2v`, `r2v`, `rv2v_case1`, `v2v_case1`,
    and `v2v_case2`/`mv2v` (accepted with the three named caveats below);
- 1 row is accepted at mid profile:
  - `ads2v` — implementation parity proven at matched settings, and the task
    outcome (ad inserted on the laptop screen through its "Bernini" logo
    reveal) demonstrated at 480 cap/61f/24fps/40 steps
    (`head_ads2v_mid480_61f/`); the full `1280x672/121f` official recipe is
    computationally intractable on this host for either engine and is
    unverified scope, not a failure;
- 2 rows fail at their official recipes in both implementations
  (oracle-proven, so not mlx-gen defects), with documented non-official
  recipes recovering them substantially:
  - `v2v_case3` — official recipe yields a dog-headed biped everywhere,
    at guidance 4.0 and 5.0 (oracle:
    `oracle_official_1_3b/v2v_case3_mid480_49f_cpu_f32/`; ablation:
    `exp_v2v_case3_v2vprefix_g5/`); the mv2v-prefix recipe yields a full
    stable quadruped in BOTH implementations (`exp_v2v_case3_mv2vprefix/`,
    `oracle_official_1_3b/v2v_case3_mv2vprefix_mid480_cpu_f32/`) — the
    prefix, not the guidance, is the decisive variable
  - `r2v_case2` — partial at the official recipe (statue/scene/shorts/cup
    bound, headphones/tee not); reference guidance 6.0 at seed 43 recovers
    cat-ear elements, chest branding, and the drinking motion
    (`exp_r2v_case2_full_refg6_s43/`)

No open row is attributable to an mlx-gen implementation defect. The one
real divergence found by code review (reference-video canvas
sizing) was fixed on 2026-08-12 and directly repaired `ads2v`.

### MV2V acceptance caveats (all three are part of the acceptance)

1. Frame 0 starts from an invented standing pose with the head cropped;
   the source clip and the retained upstream clip both start bent-over with
   the full body visible.
2. The crouch lands around frame 40, later than the retained clip's ~frame
   20.
3. Late in the clip the dog stands and walks, while the prompt explicitly
   asks for the dog to stay seated and the scene otherwise unchanged. This
   is a prompt-explicit deviation, larger in kind than the accepted
   `rv2v_case1` shirt-closure caveat, and is accepted under the same
   caveated-acceptance precedent.

No valid full-profile oracle exists for this row (the official code is
CPU-intractable at 848x480/81f/40 steps on this host), so the acceptance
rests on the qualitative task outcome at the official profile plus the
reduced-domain oracle parity proof (`mv2v_reduced_cpu_f32` vs
`baseline_after_revert_mv2v_reduced_25f_20step`: identical trajectory and
identical late-clip failure with bit-identical noise).

## Accepted evidence on Monday, August 10, 2026

Committed proof with full prompts and contact sheets lives in
[docs/assets/validation/bernini-r-1.3b-2026-08-11/](../bernini-r-1.3b-2026-08-11/README.md):

- `i2i` — [i2i/README.md](../bernini-r-1.3b-2026-08-11/i2i/README.md)
- `t2i` — [t2i/README.md](../bernini-r-1.3b-2026-08-11/t2i/README.md)
- `t2v` — [t2v/README.md](../bernini-r-1.3b-2026-08-11/t2v/README.md)
- `r2v` — [r2v/README.md](../bernini-r-1.3b-2026-08-11/r2v/README.md)
- `rv2v_case1` — [rv2v_case1/README.md](../bernini-r-1.3b-2026-08-11/rv2v_case1/README.md)
  - accepted with the narrow caveat that the shirt stays more closed than the official output, so
    less undershirt is visible, but the replacement is stable and fits correctly on-body.

## Accepted evidence added on Tuesday-Wednesday, August 11-12, 2026

- `v2v_case1` — [v2v_case1/README.md](../bernini-r-1.3b-2026-08-11/v2v_case1/README.md)
- `mv2v` (`v2v_case2`) — [mv2v/README.md](../bernini-r-1.3b-2026-08-11/mv2v/README.md)
- `ads2v` (mid profile) — [ads2v/README.md](../bernini-r-1.3b-2026-08-11/ads2v/README.md)
- `v2v_case3` / `r2v_case2` (oracle-dispositioned and tuned recovery recipes)
  - official: [v2v_case3_official](../bernini-r-1.3b-2026-08-11/v2v_case3_official/README.md),
    [r2v_case2_official](../bernini-r-1.3b-2026-08-11/r2v_case2_official/README.md)
  - tuned recovery: [v2v_case3_mv2vprefix](../bernini-r-1.3b-2026-08-11/v2v_case3_mv2vprefix/README.md),
    [r2v_case2_tuned](../bernini-r-1.3b-2026-08-11/r2v_case2_tuned/README.md)

## Remaining scope

- Close the `v2v_case3` recipe evidence: the mv2v-prefix + guidance-5.0
  CPU-oracle run (recipe parity) and the prefix-vs-guidance ablation.
- Decide promotion policy for `v2v_case3` and `r2v_case2`: their official
  recipes are oracle-proven 1.3B failures in both implementations, while
  documented tuned recipes succeed substantially — decide whether tuned-recipe
  success plus matched-settings parity qualifies for promotion.
- Unverifiable on this host (for either engine, noted as scope, not failure):
  the full `1280x672/121f` ads2v recipe and full-profile official-code oracle
  runs (the CPU-only valid oracle path is intractable above mid profile).

## Known upstream ambiguities that must be resolved explicitly

### 1. Prompt enhancer ambiguity

The public 1.3B model card says `--use_pe` is "highly recommended" for best quality, but the
published example commands do not state which retained outputs were generated with it. Official
parity should therefore be tracked in two layers:

1. raw public case-file parity without prompt enhancement;
2. prompt-enhanced parity only if the raw case still misses and the upstream retained output is
   shown to depend on `--use_pe`.

### 2. Multiple official R2V rows, not one

The current pinned upstream revision publicly contains both `assets/testcases/r2v/r2v.json` and
`assets/testcases/r2v/r2v_case2.json`, and the helper script launches both. The parity target must
therefore treat them as two distinct official rows rather than treating one as a naming mistake.

### 3. Multi-GPU recipe versus semantic contract

The public docs show the video examples with `torchrun --nproc-per-node 8 --ulysses 8`, but the
same docs also state that the single-GPU and multi-GPU scripts take the same inputs. The
eight-GPU launcher is therefore an execution recipe, not by itself a different semantic contract.

### 4. ADS2V guidance-mode mismatch inside upstream public materials

The shipped public renderer script for the `ads2v` testcase uses `guidance_mode=rv2v`, while the
Gradio renderer defaults map `ads2v` to `v2v_apg`. For exact public-example parity, treat the
scripted example contract as the target for this row and run `rv2v`.

## Why this matters

Without this matrix, "Bernini parity" can mean three incompatible things:

- current narrow renderer routes pass local custom diagnostics;
- currently surfaced official-like cases pass;
- all public 1.3B examples from the vendor pass.

This file fixes the target to the third meaning.

## Source references

- Public 1.3B model card: https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers
- Public 1.3B files tree: https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers/tree/main
- Locked inventory snapshot: [official_example_inventory.json](official_example_inventory.json)
- Official testcases README: https://github.com/bytedance/Bernini/blob/main/assets/testcases/README.md
- Official repository README/examples: https://github.com/bytedance/Bernini
- Official Gradio task/defaults file: https://github.com/bytedance/Bernini/blob/main/gradio_demo.py
