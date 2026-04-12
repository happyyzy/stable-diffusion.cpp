<p align="center">
  <img src="./assets/logo.png" width="360x">
</p>

# stable-diffusion.cpp

<div align="center">
<a href="https://trendshift.io/repositories/9714" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9714" alt="leejet%2Fstable-diffusion.cpp | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

Diffusion model(SD,Flux,Wan,...) inference in pure C/C++

***Note that this project is under active development. \
API and command-line option may change frequently.***

## Adreno Optimization Fork

This fork is purpose-built for **Adreno 830 + Q4_0** deployment (FLUX.2-klein / Z-Image / Qwen3-4B), with strict step-by-step numeric validation and reproducible logs/images.

- Full optimization logbook (GOAL-aligned): [`docs/adreno/README.md`](./docs/adreno/README.md)
- Per-step reports and assets: [`docs/adreno/steps/`](./docs/adreno/steps/) + [`docs/adreno/assets/`](./docs/adreno/assets/)
- Runtime/build switch catalog (with accepted presets): [`docs/adreno/flags.md`](./docs/adreno/flags.md)
- Step-tag map: [`docs/adreno/TAGS.md`](./docs/adreno/TAGS.md)

### Fork Branch Model

- `work/main`: full Adreno optimization + debugging history (performance-first engineering branch)
- `pr/main`: upstream-oriented clean branch (minimal patch surface, neutral naming, merge-friendly docs)

### Performance Snapshot

| Case | Baseline | Optimized | Gain |
|---|---:|---:|---:|
| FLUX.2-klein 1024 flash-on (step forward) | 209.81 s/it | 31.256 s/it | 6.71x |
| Z-Image 1024 step1 flash-on | 341.70 s /it| 50.90 s /it| 6.71x |
| FLUX.2-klein 512 full 4-step final gate | 47.81 s total | 38.06 s total | 1.26x |
| FLUX.2-klein 512 edit final gate | 74.96 s total | 67.36 s total | 1.11x |
| FLUX.2-klein 512 edit (2 refs) gate | 100.21 s total | 98.93 s total | pass |

### Before / After (Real Step Artifacts)

| Scenario | Before | After |
|---|---|---|
| FLUX.2-klein 512 final gate | <img src="./docs/adreno/assets/step27/step27_run1_full_cond.png" width="240" /> | <img src="./docs/adreno/assets/step27/step27_run2_cond256_pass.png" width="240" /> |
| FLUX.2-klein 512 edit (2 refs) | <img src="./docs/adreno/assets/step29/step29_base_2ref_resize.png" width="240" /> | <img src="./docs/adreno/assets/step29/step29_pass_2ref_noresize_optmem.png" width="240" /> |
| Z-Image 512 8-step | <img src="./docs/adreno/assets/step25/step25_base_qcomml_t1.png" width="240" /> | <img src="./docs/adreno/assets/step25/step25_outonly_hostattn_step25opt_new.png" width="240" /> |

### Quick Entry (Adreno)

1. Read the frozen presets in [`docs/adreno/flags.md`](./docs/adreno/flags.md).
2. Replay accepted steps from [`docs/adreno/steps/`](./docs/adreno/steps/).
3. Use `work/main` for performance experiments, and `pr/main` for upstream-ready patch preparation.







