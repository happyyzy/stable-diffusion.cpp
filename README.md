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

### Performance Snapshot (verified logs only)

| Scenario | Metric | Baseline (scope) | Optimized | Gain | Record |
|---|---|---:|---:|---:|---|
| FLUX.2-klein 1024 flash-on trunk | sampling (s/step) | 209.81 (native flash, same-source) | 31.256 | 6.71x | `docs/adreno/README.md` Step07 + Step10 |
| Z-Image 1024 flash-on trunk | sampling (s/step) | 341.70 (true-native flash, same-source) | 50.90 | 6.71x | `docs/adreno/steps/step18.md` + `docs/adreno/steps/step19.md` |
| Z-Image 1024 VAE decode-only | decode (s) | 36.88 (ggml conv-direct ref) | 9.25 | 3.99x | `exp_20260216_zimage_q40/step18_1024_flashon_mldrift/run_step18_native_s1_decode_only_oclvae.log` + `docs/adreno/steps/step20.md` |
| Z-Image 1024 VAE decode-only (fastest probe, no-attn) | decode (s) | 36.88 (ggml conv-direct ref) | 8.47 | 4.35x | `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_noattn_t40_o40.log` |
| FLUX.2-klein 1024 VAE decode-only | decode (s) | 32.80 (ggml decode ref) | 5.25 | 6.25x | `docs/adreno/steps/step22.md` + `docs/adreno/steps/step23.md` |
| FLUX.2-klein 512 VAE decode-only | decode (s) | 7.70 (ggml decode-stage ref) | 0.74 | 10.41x | `exp_20260219_localdream_auto/step31_klein_512/repro_server_loop/caseR_t8_convdirect/server_full.log` + `docs/adreno/steps/step26.md` |

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

## 🔥Important News

* **2026/01/18** 🚀 stable-diffusion.cpp now supports **FLUX.2-klein**  
  👉 Details: [PR #1193](https://github.com/leejet/stable-diffusion.cpp/pull/1193)

* **2025/12/01** 🚀 stable-diffusion.cpp now supports **Z-Image**  
  👉 Details: [PR #1020](https://github.com/leejet/stable-diffusion.cpp/pull/1020)

* **2025/11/30** 🚀 stable-diffusion.cpp now supports **FLUX.2-dev**  
  👉 Details: [PR #1016](https://github.com/leejet/stable-diffusion.cpp/pull/1016)

* **2025/10/13** 🚀 stable-diffusion.cpp now supports **Qwen-Image-Edit / Qwen-Image-Edit 2509**  
  👉 Details: [PR #877](https://github.com/leejet/stable-diffusion.cpp/pull/877)

* **2025/10/12** 🚀 stable-diffusion.cpp now supports **Qwen-Image**  
  👉 Details: [PR #851](https://github.com/leejet/stable-diffusion.cpp/pull/851)

* **2025/09/14** 🚀 stable-diffusion.cpp now supports **Wan2.1 Vace**  
  👉 Details: [PR #819](https://github.com/leejet/stable-diffusion.cpp/pull/819)

* **2025/09/06** 🚀 stable-diffusion.cpp now supports **Wan2.1 / Wan2.2**  
  👉 Details: [PR #778](https://github.com/leejet/stable-diffusion.cpp/pull/778)

## Features

- Plain C/C++ implementation based on [ggml](https://github.com/ggml-org/ggml), working in the same way as [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Super lightweight and without external dependencies
- Supported models
  - Image Models
    - SD1.x, SD2.x, [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
    - SDXL, [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)
    - [Some SD1.x and SDXL distilled models](./docs/distilled_sd.md)
    - [SD3/SD3.5](./docs/sd3.md)
    - [FLUX.1-dev/FLUX.1-schnell](./docs/flux.md)
    - [FLUX.2-dev/FLUX.2-klein](./docs/flux2.md)
    - [Chroma](./docs/chroma.md)
    - [Chroma1-Radiance](./docs/chroma_radiance.md)
    - [Qwen Image](./docs/qwen_image.md)
    - [Z-Image](./docs/z_image.md)
    - [Ovis-Image](./docs/ovis_image.md)
  - Image Edit Models
    - [FLUX.1-Kontext-dev](./docs/kontext.md)
    - [Qwen Image Edit series](./docs/qwen_image_edit.md)
  - Video Models
    - [Wan2.1/Wan2.2](./docs/wan.md)
  - [PhotoMaker](https://github.com/TencentARC/PhotoMaker) support.
  - Control Net support with SD 1.5
  - LoRA support, same as [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora)
  - Latent Consistency Models support (LCM/LCM-LoRA)
  - Faster and memory efficient latent decoding with [TAESD](https://github.com/madebyollin/taesd)
  - Upscale images generated with [ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- Supported backends
  - CPU (AVX, AVX2 and AVX512 support for x86 architectures)
  - CUDA
  - Vulkan
  - Metal
  - OpenCL
  - SYCL
- Supported weight formats
  - Pytorch checkpoint (`.ckpt` or `.pth`)
  - Safetensors (`.safetensors`)
  - GGUF (`.gguf`)
- Supported platforms
    - Linux
    - Mac OS
    - Windows
    - Android (via Termux, [Local Diffusion](https://github.com/rmatif/Local-Diffusion))
- Flash Attention for memory usage optimization
- Negative prompt
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) style tokenizer (not all the features, only token weighting for now)
- VAE tiling processing for reduce memory usage
- Sampling method
    - `Euler A`
    - `Euler`
    - `Heun`
    - `DPM2`
    - `DPM++ 2M`
    - [`DPM++ 2M v2`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457)
    - `DPM++ 2S a`
    - [`LCM`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13952)
- Cross-platform reproducibility
    - `--rng cuda`, default, consistent with the `stable-diffusion-webui GPU RNG`
    - `--rng cpu`, consistent with the `comfyui RNG`
- Embedds generation parameters into png output as webui-compatible text string

## Quick Start

### Get the sd executable

- Download pre-built binaries from the [releases page](https://github.com/leejet/stable-diffusion.cpp/releases)
- Or build from source by following the [build guide](./docs/build.md)

### Download model weights

- download weights(.ckpt or .safetensors or .gguf). For example
    - Stable Diffusion v1.5 from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 

    ```sh
    curl -L -O https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
    ```

### Generate an image with just one command

```sh
./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat"
```

***For detailed command-line arguments, check out [cli doc](./examples/cli/README.md).***

## Performance

If you want to improve performance or reduce VRAM/RAM usage, please refer to [performance guide](./docs/performance.md).

## More Guides

- [SD1.x/SD2.x/SDXL](./docs/sd.md)
- [SD3/SD3.5](./docs/sd3.md)
- [FLUX.1-dev/FLUX.1-schnell](./docs/flux.md)
- [FLUX.2-dev/FLUX.2-klein](./docs/flux2.md)
- [FLUX.1-Kontext-dev](./docs/kontext.md)
- [Chroma](./docs/chroma.md)
- [🔥Qwen Image](./docs/qwen_image.md)
- [🔥Qwen Image Edit series](./docs/qwen_image_edit.md)
- [🔥Wan2.1/Wan2.2](./docs/wan.md)
- [🔥Z-Image](./docs/z_image.md)
- [Ovis-Image](./docs/ovis_image.md)
- [LoRA](./docs/lora.md)
- [LCM/LCM-LoRA](./docs/lcm.md)
- [Using PhotoMaker to personalize image generation](./docs/photo_maker.md)
- [Using ESRGAN to upscale results](./docs/esrgan.md)
- [Using TAESD to faster decoding](./docs/taesd.md)
- [Docker](./docs/docker.md)
- [Quantization and GGUF](./docs/quantization_and_gguf.md)
- [Inference acceleration via caching](./docs/caching.md)

## Bindings

These projects wrap `stable-diffusion.cpp` for easier use in other languages/frameworks.

* Golang (non-cgo): [seasonjs/stable-diffusion](https://github.com/seasonjs/stable-diffusion)
* Golang (cgo): [Binozo/GoStableDiffusion](https://github.com/Binozo/GoStableDiffusion)
* C#: [DarthAffe/StableDiffusion.NET](https://github.com/DarthAffe/StableDiffusion.NET)
* Python: [william-murray1204/stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python)
* Rust: [newfla/diffusion-rs](https://github.com/newfla/diffusion-rs)
* Flutter/Dart: [rmatif/Local-Diffusion](https://github.com/rmatif/Local-Diffusion)

## UIs

These projects use `stable-diffusion.cpp` as a backend for their image generation.

- [Jellybox](https://jellybox.com)
- [Stable Diffusion GUI](https://github.com/fszontagh/sd.cpp.gui.wx)
- [Stable Diffusion CLI-GUI](https://github.com/piallai/stable-diffusion.cpp)
- [Local Diffusion](https://github.com/rmatif/Local-Diffusion)
- [sd.cpp-webui](https://github.com/daniandtheweb/sd.cpp-webui)
- [LocalAI](https://github.com/mudler/LocalAI)
- [Neural-Pixel](https://github.com/Luiz-Alcantara/Neural-Pixel)
- [KoboldCpp](https://github.com/LostRuins/koboldcpp)

## Contributors

Thank you to all the people who have already contributed to stable-diffusion.cpp!

[![Contributors](https://contrib.rocks/image?repo=leejet/stable-diffusion.cpp)](https://github.com/leejet/stable-diffusion.cpp/graphs/contributors)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=leejet/stable-diffusion.cpp&type=Date)](https://star-history.com/#leejet/stable-diffusion.cpp&Date)

## References

- [ggml](https://github.com/ggml-org/ggml)
- [diffusers](https://github.com/huggingface/diffusers)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [sd3-ref](https://github.com/Stability-AI/sd3-ref)
- [stable-diffusion-stability-ai](https://github.com/Stability-AI/stablediffusion)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)
- [generative-models](https://github.com/Stability-AI/generative-models/)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
