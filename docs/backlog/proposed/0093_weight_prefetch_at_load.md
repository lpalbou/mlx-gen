# Proposed: Sequential weight prefetch at model load (kill the lazy-mmap cold tax)

## Metadata

- Created: 2026-07-23
- Status: Proposed (from the 2026-07-23 BlackPixel i2i latency audit)
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
