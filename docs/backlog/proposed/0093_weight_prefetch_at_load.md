# Proposed: Sequential weight prefetch at model load (kill the lazy-mmap cold tax)

## Metadata

- Created: 2026-07-23
- Status: Implemented (pending release) — 2026-07-23, cycle-1 implementation wave
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None expected (load-time behavior, no output change)

## Context

Measured 2026-07-23 on M5 Max 128 GB (FLUX.2 Klein 9B q8 package, page-cold):
safetensors weights load via lazy mmap, so the first USE of each tensor
faults pages in at an effective 120-320 MB/s (random-ish touch order through
compute), while the same SSD sustains ~12.6 GB/s sequential. Concretely: the
7.5 GB Qwen3 text encoder faulted in over ~65 s DURING the first prompt
encode, and the ~9.5 GB transformer added ~28 s inside denoise step 1 —
~100 s of a 164.7 s cold generation was disk-wait disguised as compute. A
sequential prefetch of the same bytes would take ~2-4 s.

The embedding-host impact is larger than the CLI case: BlackPixel's worker
reloads the model on every task-mode flip (t2i <-> edit have different
handlers) and after every cancel/displacement kill, and the owner's setup
runs the bf16 repo (~35 GB TE+transformer — roughly double the q8 bytes).
The measured cold click was 447 s vs ~250 s warm at the same settings; the
difference is dominated by exactly this fault-in tax.

## Proposal

At load time (or immediately after building the module tree), sequentially
read each weight file's bytes to materialize them in the page cache before
compute first touches them — e.g. a bounded-buffer sequential read of every
safetensors file selected for the load, or `mx.eval` over freshly loaded
arrays where that already forces materialization in file order. Options to
evaluate:

1. Plain sequential file read (warms page cache; simplest; benefits every
   consumer including torch-side loads; wasted only if memory pressure
   evicts pages before use).
2. Eager `mx.eval` of the loaded parameter tree at load (materializes MLX
   buffers directly; moves the cost into load where progress can be shown).
3. `madvise(MADV_WILLNEED)`-style hints if exposed through the stack.

Ship as default-on with an opt-out flag if any host prefers lazy behavior
(low-RAM mode likely wants to keep laziness — decide explicitly).

## Expected impact (from measured numbers)

- Cold CLI generation at 1024x768 q8: ~165 s -> ~70 s.
- BlackPixel cold click / post-kill reload at bf16: ~200 s of fault-in
  compressed to ~10-20 s of sequential read (~447 s -> ~270 s for the
  owner's measured cold case).

## Promotion criteria

Promote when a bounded prototype shows (a) page-cold end-to-end time drops
by roughly the predicted amount on at least one q8 and one bf16 family, and
(b) warm runs and low-RAM mode are not regressed. Validate with the
page-cache purge methodology from the 2026-07-23 audit (purge, run with
--json-events, compare phase walls).

## Implementation record (2026-07-23)

### Design decision

Option 1 (plain sequential read) shipped, refined with two gates; options 2
and 3 rejected:

- `mx.eval` of loaded trees (option 2) changes memory-timing semantics — it
  would materialize full-precision trees before quantize-at-apply and break
  MLX lazy semantics that low-RAM paths depend on.
- `madvise(MADV_WILLNEED)` (option 3) is asynchronous with no completion
  signal and uncertain Darwin semantics; revisit only if the synchronous
  read's wall time ever matters.
- Gate 1 — residency probe: before reading, each file's page-cache residency
  is measured via `mincore(2)` (ctypes; fail-safe to "cold" on any probe
  error). Files >= 97% resident are skipped, so warm reloads pay only the
  probe (~0.03 s/GB measured) and re-read zero bytes. This matters because a
  16-35 GB package would otherwise pay a 2-5 s page-cache-speed re-read on
  every warm reload (BlackPixel reloads on every task flip).
- Gate 2 — RAM headroom: prefetch is skipped when the selected files exceed
  half of physical RAM (or when RAM size is unknown). This is the
  general-purpose low-RAM protection — the loader has no `--low-ram`
  knowledge, and the actual enemy on small machines is page-cache pressure,
  which this gate measures directly.
- Opt-out: `MFLUX_NO_WEIGHT_PREFETCH=1`.
- Seam: `WeightPrefetcher.prefetch(paths)` called at every resolved
  weight-file list in `WeightLoader` (mflux-format shards, mlx-native,
  torch-convert, multi-json, torch-bfloat16, single, multi-glob, torch
  checkpoint/tensor directories, direct-URL single files), so image + wan +
  seedvr2 families all benefit and the read is bounded to exactly the files
  being loaded. Reads are 64 MiB `readinto` chunks on an unbuffered handle;
  interruptible-safe (read-only, no state).

### Measured numbers (M5 Max 128 GB, BlackPixel idle at 0.0% CPU, serialized)

Eviction methodology: no sudo purge available; eviction = streaming ~161 GB
of OTHER model blobs through the page cache (~14.5 s at ~11 GB/s), then
verifying the target's residency via the mincore probe. Residency reached
0.000 on every target shard before each cold leg — eviction was reliable.

Target: FLUX.2 Klein 9B q8 text encoder (4 shards, 8.04 GB) and the full
q8 package (17.85 GB) from the HF cache.

| Leg | Result |
| --- | --- |
| Mechanism, cold random-order access (1 byte per 4th 16 KiB page, shuffled) | 12.83 s for a quarter of the pages — ~157 MB/s effective, reproducing the audit's 120-320 MB/s fault-in regime |
| Mechanism, prefetch then same access | 0.91 s prefetch (8.04 GB at ~8.8 GB/s) + 0.18 s access = ~12x faster on the measured subset (~46x extrapolated to full materialization); byte checksums identical |
| Mechanism, cold bulk `mx.load`+`mx.eval` | 0.61 s (~13 GB/s) — bulk eval is already sequential, NOT the disease path |
| q8 package end-to-end cold, no prefetch | construct 2.46 s, first 512x512 2-step generate 4.95 s, second 3.50 s |
| q8 package end-to-end cold, with prefetch | construct 4.47 s (includes 1.96 s prefetch of 17.85 GB), first generate 4.69 s, second 3.64 s |
| Warm re-prefetch (residency skip) | 0 bytes read, 0.208 s per 8.04 GB |

### Honest caveats (what could and could not be measured)

- The end-to-end q8 prepared-package path did NOT exhibit the audit's
  disease in isolation: its cold-vs-warm delta was ~1.5 s total, and the
  prefetch is a ~1.7 s net regression there on an idle machine. Weights on
  that path are consumed near-sequentially at apply/eval time, so the page
  cache warms itself. The disease regime (random-order faulting at
  ~150 MB/s) was reproduced tonight only at the mechanism level.
- The audit's in-situ ~100 s tax was measured under the BlackPixel worker
  with system memory pressure (32.9 GB MLX cache resident — see 0094) and
  the owner's bf16 repo, whose non-quantized apply path stays lazy through
  first compute. The local bf16 BFL snapshot is incomplete (active ref lacks
  the transformer shards), so the bf16 end-to-end leg could not be run
  offline tonight.
- Decision under that evidence: ship default-on (bounded worst case
  ~0.11 s/GB cold + ~0.03 s/GB warm probe) as insurance against the measured
  in-situ regime; hosts that only ever load prepared-q8 packages on idle
  machines can set `MFLUX_NO_WEIGHT_PREFETCH=1`. Cycle-2 / BlackPixel-side
  validation should re-measure in situ (worker process, real pressure) and
  may flip the default if the q8-path regression dominates in practice.

### Cycle-2 ruling (2026-07-23, adversarial review): skip prefetch on prepared packages

The net-regression-on-q8 was independently reproduced and the default was
narrowed: `_try_load_mflux_format` (the prepared-package path) no longer
prefetches; every HF-repo loading mode keeps the default-on prefetch.

Reproduction (same eviction methodology: stream 150 GiB of other blobs,
verify 0.000 residency on every target shard via the mincore probe;
BlackPixel idle at 0.1% CPU; Klein 9B q8 package, 16.6 GiB, 512x512,
2 steps, fresh process per leg):

| Leg | construct | gen1 | gen2 | total |
| --- | --- | --- | --- | --- |
| with prefetch, cold #1 | 4.45 | 4.92 | 3.73 | 13.09 |
| with prefetch, cold #2 | 4.37 | 4.33 | 3.49 | 12.19 |
| without prefetch, cold #1 | 2.47 | 4.66 | 3.88 | 11.01 |
| without prefetch, cold #2 | 2.49 | 4.79 | 3.55 | 10.83 |

Prefetch cost +1.7 to +2.1 s end-to-end (~16%), all of it in construct;
gen1 was unchanged — the package's bytes fault in at sequential speed
during first compute anyway. Post-fix verification runs (default-on, cold,
same eviction) landed construct at 3.02/2.46 s — the no-prefetch profile
(gen legs were noisier because unrelated system load appeared; construct is
the disk-dominated metric).

Why the boundary is the FORMAT, not the quantization level:
`ModelSaver._save_weights` flattens with `tree_flatten` and shards in that
order, so prepared packages are laid out on disk in module-tree order — the
same order apply/quantize/first-forward traverses. Kernel readahead already
runs at SSD speed on that pattern, quantized or not. HF repos have no such
guarantee (alphabetical shard layout vs module-order consumption diverges,
e.g. blocks 0,1,10,11,...,2 — the audit's 120-320 MB/s regime), so they keep
the prefetch as bounded insurance; the local bf16 BFL snapshot is still
incomplete, so that leg remains unmeasured end-to-end.

Residual risk (cycle 3): the audit's in-situ ~100 s tax on this same q8
package was recorded under the pre-0094 pressure regime (32.9 GB uncapped
MLX cache). 0094 removed that cause; if BlackPixel-side validation still
sees slow cold loads on prepared packages under real pressure, prefetch for
prepared packages under a pressure signal is the follow-up, not a blanket
re-enable. Regression test:
`test_weight_loader_skips_prefetch_for_prepared_packages`.

### Files changed

- `src/mflux/models/common/weights/loading/weight_prefetcher.py` (new)
- `src/mflux/models/common/weights/loading/weight_loader.py` (prefetch at
  every resolved file list; 0094 hook at `load`/`load_single`)
- `tests/weights/test_weight_prefetcher.py` (new: order/full-read, env skip,
  RAM gate, unknown-RAM skip, resident skip, missing-file behavior, real
  mincore probe, bitwise load parity, loader-seam wiring)
- Docs: `docs/api.md` (Runtime Memory Defaults), `docs/python-integration.md`,
  `CHANGELOG.md` [Unreleased].

### Gates

Cycle-1: `make lint` clean; fast suite 1457 passed / 7 skipped (baseline
1413/7; +44 new tests, zero regressions). Cycle-2 (after the
prepared-package skip): `make lint` clean; fast suite 1465 passed /
7 skipped.
