# Adreno Step Tags

These tags correspond to GOAL step milestones and are used as stable lookup points for debugging and benchmark records.

- `adreno-step01`: bisect to `ggml_cl_scale` minimal fix (bad image -> usable)
- `adreno-step02`: restore thread1 minimal-performance baseline (~8.5s class single-fwd)
- `adreno-step03`: fix Q4 upload race (threads=1/4 numerical consistency)
- `adreno-step04`: repair Qwen3-4B Q4 OpenCL decode correctness
- `adreno-step05`: reconnect Qwen+Klein chain, image path restored
- `adreno-step06`: full-phone tuning (98.16s -> 59.39s total reference)
- `adreno-step07`: 1024 baseline (flash-off OOM, flash-on 209.81s/step)
- `adreno-step08`: transfer 512 debug method to 1024 numeric workflow
- `adreno-step09`: 1024 chain image validation gate
- `adreno-step10`: mldrift attention integrated, 31.256s/step class reached
- `adreno-step11`: mldrift numeric repair (call0 diff major reduction)
- `adreno-step12`: recover speed + step1/2/4 checks
- `adreno-step13`: 1024 4-step chain acceptance
- `adreno-step14`: 512 edit fully on Adreno OpenCL accepted
- `adreno-step15`: long-seq edit attention optimized (~63.40s sampling)
- `adreno-step16`: 768 edit with mldrift accepted
- `adreno-step17`: z-image 512 flash-off repair accepted
- `adreno-step18`: z-image 1024 flash-on + 4096+128 attention path integrated

See full detail: `docs/adreno/README.md`.
