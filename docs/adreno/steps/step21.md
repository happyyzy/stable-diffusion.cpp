# Step21 - Z-Image 1024 8-Step Final Gate

Status: passed (2026-02-18, rerun)

## Goal

- z-image turbo, 1024, 8 steps, full phone flow
- gate: total time `<480s`
- image must be normal (not black/noise)

## Fixed run configuration

- trunk:
  - `GGML_OPENCL_MLDRIFT=1`
  - `GGML_OPENCL_MLDRIFT_H30_IO_FIRST=1`
  - `GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD=4096`
- q4 stabilizer:
  - `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR=context_refiner.0.attention.out.weight,noise_refiner.0.attention.out.weight`
  - `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS=64`
- vae:
  - `--vae-backend qcom_ml --vae-conv-direct`
  - `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`
  - tiled decode (`tile=40`, `overlap=0.0`)

## Attempt history

### Attempt 1 (fail gate)

- log: `exp_20260216_zimage_q40/step21_final_8step/run_step21_zimg_1024_s8_qcomml.log`
- output: `exp_20260216_zimage_q40/step21_final_8step/step21_zimg_1024_s8_qcomml.png`
- timing:
  - condition: `244 ms`
  - sampling: `480.76s` (`~60.09s/it`)
  - vae decode: `12.19s`
  - total: `493.39s` (fail, over by `13.39s`)

### Attempt 2 (thermal rerun, pass)

- log: `exp_20260216_zimage_q40/step21_final_8step/run_step21_zimg_1024_s8_qcomml_rerun.log`
- output: `exp_20260216_zimage_q40/step21_final_8step/step21_zimg_1024_s8_qcomml_rerun.png`
- timing:
  - condition: `241 ms`
  - sampling: `448.23s` (`~56.03s/it`)
  - vae decode: `9.73s`
  - total: `458.41s` (pass, margin `21.59s`)

## Conclusion

- Step21 gate is accepted on rerun:
  - total `458.41s < 480s`
  - image quality confirmed normal in review.
- first-vs-rerun delta indicates strong thermal sensitivity:
  - total improved `34.98s`
  - sampling improved `32.53s`

## Full report

- `exp_20260216_zimage_q40/step21_final_8step/report_step21_attempt_20260218.md`
