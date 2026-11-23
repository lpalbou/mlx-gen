# Proposed: MiniMax-H3 memory follow-ups after staged loading

## Metadata

- Created: 2026-11-23
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [0002](../../adr/0002_no_silent_automatic_fallbacks.md) (a memory option that
  cannot apply is reported, not dropped),
  [0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md).
- ADR impact: none.

## Context

Item [0130](../completed/0130_minimax_h3_staged_loading_and_reindexed_8bit.md) took the conditioner,
the AdaLN projections and the VAEs out of resident memory, added `--stream-blocks`, and re-indexed
the 8-bit weights. The items below were found during that work and its three adversarial reviews
and were left out because none of them is needed for the shipped behavior.

## Candidates

1. **Keep several blocks in memory while streaming (owner's suggestion, 2026-11-23; not
   required).** `--stream-blocks` holds one block and reads every block from disk at every step. A
   setting for how many blocks stay resident (or a memory budget the engine fills with blocks)
   would trade memory for fewer reads. It is not needed on the hardware MLX-Gen targets: every
   Apple Silicon Mac has a fast internal SSD, and the measured cost of streaming is 1.5 to 4
   seconds on steps of 115 to 125 seconds, with or without room for the system file cache. It
   becomes relevant only for a package kept on a slow external disk.
2. **Re-index the 8-bit weights of the other families.** The gain measured on MiniMax-H3 comes
   from MLX choosing levels on a float32 grid and storing a bf16 grid, which applies to every
   8-bit package MLX-Gen writes. The other families quantize through `WeightApplier` and
   `nn.quantize`. Measure one family (velocity or image error against its BF16 weights) before
   changing their packages; each would need a re-upload.
3. **Report the read rate of streamed blocks.** After the first step the CLI could print the
   measured GiB per second, so a slow disk is visible in the first minute of a run.
4. **Request preflight.** The estimate anchors on the current footprint plus deferred VAE bytes
   and a per-row term measured when denoising was the peak. Pending adapter tensors and cached
   modulation tables are not counted, and at `960x544` and `1344x768` the decode is now the peak.
   A plan-based estimate (the largest of the encode, denoise and decode phases) would replace the
   single line.
5. **Working memory.** The adapter-wrapped feed-forward input projection holds three full-width
   arrays at once; chunking the feed-forward by rows and folding the adapter scale into its second
   factor would save an estimated 1.4 GiB at `960x544` and 4 GiB at `1344x768`. Not measured.
6. **Source snapshot with `--quantize`.** The streamed conditioner is quantized again at every
   encode, and with `--low-ram` the transformer is quantized again for every seed. Measure the time,
   and decide whether a load-time quantized conditioner should be read unquantized instead (which
   would end the byte-identity between a package and load-time quantization).
7. **Streaming the mixed 4-bit package** and **`--stream-blocks` without `--low-ram` under locked
   RAM** were not measured.
8. **Physical machines.** The 24 to 96 GB results come from one 128 GB Mac with RAM locked; a run
   on a real Mac of each size has not been done.
9. **Hosts.** BlackPixel bundles mlx-gen 0.37, hard-codes a 93 GB peak for MiniMax-H3 and sends
   `--quantize 8`; it needs its own change to use the new figures, `--stream-blocks` and the
   16-bit weights.

## Promotion criteria

Promote item 2 when another family's 8-bit package is next rebuilt. Items 3 to 6 can ride with the
next change to the MiniMax-H3 runtime. Item 1 needs a report of slow streaming from a user first.
