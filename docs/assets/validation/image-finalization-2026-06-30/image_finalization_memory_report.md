# Image Finalization Memory Report

Measured on June 30, 2026 with a dedicated save-phase probe.

## Method
- Synthetic deterministic RGB PNG: 4096x4096
- Sampling window: after image construction, during save/finalization only
- Metrics: sampled process RSS and Darwin physical footprint for the child save process
- Cases: current default save, current embedded metadata save, current sidecar metadata save, simulated legacy default save

## Results

| Case | Save Calls | Reopen Calls | Runtime Snapshots | Peak RSS GB | Avg RSS GB | Peak Physical GB | Avg Physical GB | EXIF | XMP | IPTC | Sidecar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| current-default | 1 | 0 | 0 | 0.192 | 0.192 | 0.171 | 0.171 | no | no | no | no |
| current-embed-metadata | 1 | 0 | 0 | 0.192 | 0.192 | 0.171 | 0.171 | yes | yes | yes | no |
| current-sidecar-metadata | 1 | 0 | 1 | 0.192 | 0.192 | 0.171 | 0.171 | no | no | no | yes |
| legacy-default-simulated | 3 | 2 | 1 | 0.394 | 0.301 | 0.373 | 0.277 | yes | yes | yes | no |

## Interpretation
- Current default versus simulated legacy default: peak RSS -51.2316%, peak physical footprint -54.1496%.
- Current embedded metadata versus simulated legacy default: peak RSS -51.2066%, peak physical footprint -54.1364%.
- Current sidecar metadata versus simulated legacy default: peak RSS -51.2066%, peak physical footprint -54.132%.
- The current embedded-metadata path still preserves EXIF plus PNG XMP/IPTC, but does so with one save call and zero reopen calls.
- The current default path no longer collects runtime-memory metadata and no longer embeds PNG metadata unless explicitly requested.
