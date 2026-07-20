# T4 GPU Benchmarks — colab-llm-server

Tested on Google Colab free-tier Tesla T4 (15GB VRAM, CUDA 7.5).
llama-cpp-python 0.3.33, CUDA backend, 150-token generation, 4096 ctx (8192 for MoE).

## Results

| Model | Quant | Size | Load Time | VRAM Used | Speed (tok/s) | Status |
|-------|-------|------|-----------|-----------|----------------|--------|
| Qwen3-8B | Q4_K_M | 5.03 GB | 2.8s | 5,443 MiB | 31.3 | OK |
| Qwen3.6-35B-A3B APEX Nano | I-Nano | 11.69 GB | 74.1s | 11,811 MiB | 43.9 | OK |
| Gemma4-26B APEX I-Mini | I-Mini | 12.27 GB | 83.4s | 13,257 MiB | 37.6 | OK |
| Darwin Reason 27B | Q3_K_M | 13.30 GB | 77.2s | 13,185 MiB | 8.0 | OK |
| Darwin Code 27B | Q3_K_M | 13.50 GB | 82.8s | 13,375 MiB | 8.0 | OK |
| Carnice V2 27B | IQ3_M | 13.66 GB | 80.5s | 13,533 MiB | 7.9 | OK |
| Qwopus v2 27B | Q3_K_M | 13.50 GB | — | — | — | Download stalled |
| **Ternary Bonsai 27B** | **Ternary Q2_0** | **~3.5 GB** | **—** | **—** | **—** | **Not yet benchmarked** |

## Analysis

### Speed tiers
- **Fast (30+ tok/s):** Qwen3-8B, Qwen3.6-35B-A3B MoE, Gemma4-26B MoE
  - MoE models punch above weight class — only active experts loaded per token
  - 8B dense model fastest load (2.8s) and lowest VRAM (5.4 GB)
- **Slow (~8 tok/s):** 27B dense models (Darwin Reason, Darwin Code, Carnice V2)
  - All 27B Q3 models fit under 15GB but barely — 13.2-13.5 GB used
  - ~2 GB headroom remains for context

### VRAM efficiency
- MoE models best bang-per-VRAM: 35B params in 11.8 GB, 43.9 tok/s
- 27B dense at Q3_K_M/IQ3_M near ceiling — 87-88% VRAM utilization
- 8B Q4_K_M uses only 35% VRAM — room for larger context or parallel instances

### Memory optimizations applied
1. **Removed `torch` import** from setup notebook → saves ~2 GB system RAM
2. **Eliminated double model load** in API server → saves 5-12 GB VRAM
3. **GPU check via `llama_cpp.llama_supports_gpu_offload()`** instead of `torch.cuda.is_available()`
4. **VRAM validation via `nvidia-smi` subprocess** instead of loading model into kernel
5. **HF cache clearing between benchmarks** → prevents disk exhaustion on 100GB Colab disk

### Recommended configs for T4
- **Best speed:** Qwen3.6-35B-A3B APEX Nano (43.9 tok/s, 11.8 GB)
- **Best quality/size:** Gemma4-26B APEX I-Mini (37.6 tok/s, 13.3 GB)
- **Best for coding:** Darwin Code 27B Q3_K_M (8.0 tok/s, 13.4 GB)
- **Lightest:** Qwen3-8B Q4_K_M (31.3 tok/s, 5.4 GB)
- **Most VRAM-efficient:** Ternary Bonsai 27B (~3.5 GB, 1.71-bit, custom fork)

## Models pending benchmark
The following models from the expanded list have been added to `generate_notebooks.py`
but not yet benchmarked due to Colab session/disk limits:
- Qwopus v2 27B (download stalled)
- Qwable 27B, ablit. Qwable 27B
- Darwin APEX 35B, Carnice APEX 35B, Carwin Nano 35B, Holo 3.1 35B
- Nemotron Nano 30B, Orinth 35B, Orinth 9B, GSM 27B, Carwin 27B
- Senter SFT 8B

All are configured with appropriate quantizations (Q3_K_M / IQ3_M / I-Nano / I-Mini)
to fit within the 15 GB T4 VRAM limit.
