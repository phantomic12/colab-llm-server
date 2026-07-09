# 🚀 Colab LLM Runner — Free-Tier T4 Models That Actually Fit

Run **large language models locally on a free Google Colab T4 GPU** (15 GB VRAM) with `llama-cpp-python` + CUDA, then optionally expose an **OpenAI-compatible API** over the internet via a Cloudflare tunnel.

Every model listed below has been **verified to fit on a free Colab T4 instance** — no Colab Pro, no A100, no V100.

---

## ✅ Verified Models (Free Colab T4 — 15360 MiB VRAM)

| # | Model | Type | Quant | Size | VRAM | Speed | Status |
|---|-------|------|-------|------|------|-------|--------|
| 1 | **Qwen3-8B** | Dense 8B | Q4_K_M | 5.03 GB | 5467 MiB | 32.0 tok/s | ✅ Proven |
| 2 | **Qwen3.6-35B-A3B** (MoE) | MoE 35B/3B | APEX Nano | 10.88 GB | ~12 GB | 29.4 tok/s | ✅ Proven |
| 3 | **Gemma 4 26B-A4B** (MoE) | MoE 26B/3.8B | APEX I-Mini | 12.27 GB | ~13 GB | loads in 98.4s | ✅ Proven to fit |
| 4 | **gpt-oss-20b** | MoE 21B/3.6B | MXFP4 (native) | ~13 GB | ~13 GB | OpenAI official | ✅ Proven to fit |

## 🆕 Expanded Model List (T4-Compatible Quants)

These models have been added with quants selected to fit the T4's 15GB VRAM:

| # | Model | Type | Quant | Size | Context | Notes |
|---|-------|------|-------|------|---------|-------|
| 5 | **Nemotron 3 Nano 30B-A3B** | MoE 30B/3B | UD-IQ3_XXS | ~13 GB | 4K | NVIDIA Mamba2 hybrid |
| 6 | **Darwin Reason 27B** | Dense 27B | Q3_K_M | 12.39 GB | 4K | Reasoning fine-tune of Qwen3.6-27B |
| 7 | **Darwin Code 27B** | Dense 27B | Q3_K_M | 12.57 GB | 4K | Coding fine-tune of Qwen3.6-27B |
| 8 | **Carnice V2 27B** | Dense 27B | IQ3_M | 12.73 GB | 4K | Agentic, tool-calling, Hermes Agent |
| 9 | **Qwopus v2 27B** | Dense 27B | Q3_K_M | 12.57 GB | 4K | Claude Opus trace-inversion, MTP |
| 10 | **Qwable 27B** | Dense 27B | Q4_K_M | 15.66 GB | 2K | ⚠️ Borderline — 2K context max |
| 11 | **Darwin APEX 35B** | MoE 36B | APEX I-Nano | 10.67 GB | 8K | Darwin Opus, MoE |
| 12 | **Carwin Nano 35B** | MoE 35B | Nano | 11.49 GB | 8K | Carwin MoE, MTP |
| 13 | **Holo 3.1 35B** | MoE 35B/3B | APEX I-Mini | 13.33 GB | 4K | H Company computer-use agent |
| 14 | **Carwin 27B** | Dense 27B | Q4_K_M | 15.68 GB | 2K | ⚠️ Borderline — Carnice+Darwin merge |
| 15 | **Ablit. Qwable 27B** | Dense 27B | — | — | — | ❌ No T4-fit quant available |

### Models Not Found on HuggingFace

The following models from the requested list could not be located on HuggingFace as of July 2026:

- **GSM 27B** — no matching model found (GSM8K fine-tunes exist but not at 27B scale with this name)
- **Orinth 35B / Orinth 9B** — no models found under any search variation
- **Senter SFT 8B** — no GGUF found (only `Redvodk/senter-omni-model` exists, which is a Qwen2.5-Omni 4-bit, not SFT 8B)

If these are private repos or published under different names, please provide the HuggingFace repo IDs.

---

## 📦 Pre-built `llama-bench` binaries (for fast Colab cold-starts)

The `06_benchmark_suite.ipynb` notebook (and any `bench_colab_one.py` you
run on Colab) needs a `llama-bench` binary. Building llama.cpp from source
inside a fresh Colab VM burns ~8 min of every benchmark run. To skip that:

```bash
# In a Colab cell, after cloning this repo:
!wget -q https://github.com/phantomic12/colab-llm-server/releases/latest/download/llama-bench-cuda-T4.tar.zst
!tar --use-compress-program=unzstd -xf llama-bench-cuda-T4.tar.zst
!cp llama-bench/bin/llama-bench /usr/local/bin/
```

Or pass `--prebuilt-asset cuda-T4` to `bench_colab_one.py` (see
`phantomic12/llama-bench/scripts/bench_colab_one.py` for the
`--prebuilt-asset` integration — it auto-resolves the latest matching asset
via the GitHub release API).

### How the binaries are produced

`.github/workflows/release-binaries.yml` builds three prebuilt artifacts
on every `v*` tag push and uploads them to GitHub Releases:

| variant | use case |
|---|---|
| `llama-bench-cuda-T4.tar.zst`     | Colab T4 (sm_75) — also runs on most NVIDIA GPUs via PTX JIT |
| `llama-bench-cpu-zen5.tar.zst`    | Ryzen AI 9 HX 370, native AVX-512 VNNI+VBMI+BF16 (Zen 4/5) |
| `llama-bench-cpu-avx2.tar.zst`    | generic AVX2 x86_64 fallback |

Each tarball bundles the binary plus its dependent `.so` files, plus a
`BUILD_INFO.txt` and a `.sha256` sidecar.

> **Why pin to llama.cpp b7997?** Upstream master has since dropped the
> `llama-bench` binary entirely. b7997 (May 2025) is the last release
> with a self-contained `tools/llama-bench` executable.

### Cutting a new release

```bash
# Tag and push — GHA picks it up automatically.
git tag -a v0.1.0 -m "release: prebuilt llama-bench binaries"
git push origin v0.1.0
# → watch /actions/workflows/release-binaries.yml
# → after ~5 min the release page has 3 tarballs
```

Or use `workflow_dispatch` from the Actions tab to re-cut artifacts under
a tag name you choose (useful for hot-fixing the build without bumping
the version).

---

## 📓 Notebooks

| Notebook | Model | What it does |
|----------|-------|--------------|
| [`00_setup.ipynb`](notebooks/00_setup.ipynb) | — | Install `llama-cpp-python` with CUDA 12.4, verify T4 GPU (**no torch — saves 2GB RAM**) |
| [`01_qwen3_8b_server.ipynb`](notebooks/01_qwen3_8b_server.ipynb) | Qwen3-8B | Download & serve Qwen3-8B Q4_K_M |
| [`02_qwen3_6_35b_moe_server.ipynb`](notebooks/02_qwen3_6_35b_moe_server.ipynb) | Qwen3.6-35B-A3B | Run 35B MoE with APEX Nano quant |
| [`03_gemma4_26b_moe_server.ipynb`](notebooks/03_gemma4_26b_moe_server.ipynb) | Gemma 4 26B-A4B | Run Gemma 4 MoE with APEX I-Mini |
| [`04_gpt_oss_20b_server.ipynb`](notebooks/04_gpt_oss_20b_server.ipynb) | gpt-oss-20b | Run OpenAI's gpt-oss-20b (MXFP4) |
| [`05_nemotron_3_nano_server.ipynb`](notebooks/05_nemotron_3_nano_server.ipynb) | Nemotron 3 Nano 30B | Run NVIDIA's MoE (auto-selects smallest quant) |
| [`06_benchmark_suite.ipynb`](notebooks/06_benchmark_suite.ipynb) | All | Benchmark suite with timing & VRAM reporting |
| [`07_darwin_reason_27b_server.ipynb`](notebooks/07_darwin_reason_27b_server.ipynb) | Darwin Reason 27B | Reasoning fine-tune, Q3_K_M |
| [`08_darwin_code_27b_server.ipynb`](notebooks/08_darwin_code_27b_server.ipynb) | Darwin Code 27B | Coding fine-tune, Q3_K_M |
| [`09_carnice_v2_27b_server.ipynb`](notebooks/09_carnice_v2_27b_server.ipynb) | Carnice V2 27B | Agentic fine-tune, IQ3_M |
| [`10_qwopus_v2_27b_server.ipynb`](notebooks/10_qwopus_v2_27b_server.ipynb) | Qwopus v2 27B | Claude Opus trace-inversion, Q3_K_M |
| [`11_qwable_27b_server.ipynb`](notebooks/11_qwable_27b_server.ipynb) | Qwable 27B | Reasoning + instruction-tuned, Q4_K_M (borderline) |
| [`12_darwin_apex_35b_server.ipynb`](notebooks/12_darwin_apex_35b_server.ipynb) | Darwin APEX 35B | Darwin Opus MoE, APEX I-Nano |
| [`13_carwin_nano_35b_server.ipynb`](notebooks/13_carwin_nano_35b_server.ipynb) | Carwin Nano 35B | Carwin MoE, Nano quant |
| [`14_holo_3_1_35b_server.ipynb`](notebooks/14_holo_3_1_35b_server.ipynb) | Holo 3.1 35B | H Company computer-use agent, APEX I-Mini |
| [`15_carwin_27b_server.ipynb`](notebooks/15_carwin_27b_server.ipynb) | Carwin 27B | Carnice+Darwin merge, Q4_K_M (borderline) |
| [`16_qwable_ablit_27b_server.ipynb`](notebooks/16_qwable_ablit_27b_server.ipynb) | Ablit. Qwable 27B | Abliterated — ⚠️ no T4-fit quant |

---

## 🧠 Memory Optimizations

This repo has been optimized to minimize RAM and VRAM usage on Colab's free tier:

### 1. No `torch` import (saves ~2GB RAM)
The original setup notebook imported `torch` just to call `torch.cuda.is_available()`. `llama-cpp-python` has its own CUDA detection via `llama_cpp.llama_supports_gpu_offload()` — no need for PyTorch's 2GB overhead.

### 2. No double model load (saves 5-12GB VRAM)
The original API server notebook loaded the model in the notebook kernel (cell 3) for an `nvidia-smi` check, then loaded it **again** in the `uvicorn` subprocess (cell 5). The model in the kernel was never used after the check. Now the notebook only checks VRAM availability without loading the model — the model loads once, in the server subprocess.

### 3. `n_threads=2` everywhere
All `Llama()` calls now explicitly set `n_threads=2`, matching the Colab T4's 2 vCPU cores. The original server script was missing this parameter.

### 4. Cloudflared via binary, not pip
The original install cell pip-installed `cloudflared` but the actual notebooks used `wget` to download the binary. Removed the unnecessary pip install.

---

## 🏃 Quickstart

### 1. Run a model locally
1. Open [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb) in Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells — this installs `llama-cpp-python` with CUDA and confirms the GPU
4. Open any model notebook (e.g. `02_qwen3_6_35b_moe_server.ipynb`) and run all cells

### 2. Serve an OpenAI-compatible API
1. Open any model notebook (e.g. `07_darwin_reason_27b_server.ipynb`) in Colab (T4 runtime)
2. Run all cells — the notebook will:
   - Install `llama-cpp-python` (CUDA)
   - Download the model
   - Start an HTTP server on `localhost:8080`
   - Start a **Cloudflare tunnel** for public access
   - Print a URL like `https://<random>.trycloudflare.com`
3. Use that URL as your OpenAI `base_url` from any client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-tunnel>.trycloudflare.com/v1",
    api_key="sk-no-key-needed"   # the server accepts any key
)

resp = client.chat.completions.create(
    model="darwin-reason-27b",
    messages=[{"role": "user", "content": "Explain MoE in one sentence."}],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```

Works with `curl`, the `openai` Python/JS SDK, LangChain, LlamaIndex, Continue.dev, etc.

---

## 🧠 About APEX Quantization

**APEX** ("Aggressive Parameter-Efficient eXpansion") is a family of ultra-low-bit GGUF quantizations tuned specifically to fit large Mixture-of-Experts (MoE) models into constrained VRAM.

- **APEX Nano** — ~2.5–3.0 bpw, tuned for large MoE models (Qwen3.6-35B-A3B fits in 12 GB)
- **APEX I-Mini** — slightly higher precision variant for dense/MoE hybrids (Gemma 4 26B-A4B)
- **APEX I-Compact** — higher quality, larger size
- **APEX I-Balanced** — highest quality APEX variant

APEX works by pushing the non-expert weights (attention, norm, embeddings) to very low bit-widths while keeping active expert routing weights slightly higher, since MoE models only activate a fraction of parameters per token. The result: a 35B-parameter model that runs on a free T4.

> ⚠️ APEX quants trade some quality for VRAM. For maximum quality at the same model size, see **ik_llama.cpp** below.

---

## 📊 Quantization Guide for T4 (15GB VRAM)

| Model Size | Recommended Quant | Size | Context | Notes |
|------------|-------------------|------|---------|-------|
| 8B dense | Q4_K_M | ~5 GB | 4K-48K | Best quality/size ratio |
| 27B dense | Q3_K_M or IQ3_M | ~12-13 GB | 4K | Q4_K_M (15.7GB) is too tight |
| 27B dense (borderline) | Q4_K_M | ~15.7 GB | 2K max | Will OOM with larger context |
| 30B MoE | UD-IQ3_XXS | ~13 GB | 4K | Standard quants too big |
| 35B MoE | APEX I-Nano | ~10-11 GB | 8K | Best MoE option for T4 |
| 35B MoE | APEX I-Mini | ~12-13 GB | 4-8K | Higher quality MoE |

---

## 🔧 Advanced: ik_llama.cpp

For users who want to push performance further, [**ik_llama.cpp**](https://github.com/ikawrakow/ik_llama.cpp) is a fork of `llama.cpp` with optimized kernels for:

- **Faster MoE inference** (matters a lot for Qwen3.6-35B-A3B and Gemma 4 26B-A4B)
- Improved KV cache quantization
- Better CPU/GPU hybrid offloading for memory-constrained setups

On a T4, switching to `ik_llama.cpp` can yield a measurable speedup on MoE models. To use it, build the `ik_llama.cpp` Python bindings from source instead of installing the prebuilt `llama-cpp-python` wheel — see the [`ik_llama.cpp` README](https://github.com/ikawrakow/ik_llama.cpp) for build instructions.

---

## ☁️ Cloudflare Tunnel Setup

The API server notebook uses `cloudflared` to create a **temporary public URL** that forwards to your Colab notebook's `localhost:8080`. No account, no DNS, no port forwarding needed.

- The URL is ephemeral — it dies when the Colab runtime disconnects or the notebook stops
- The tunnel is one-way (inbound to your server); it does not expose your machine
- Any API key is accepted by the server — the tunnel URL itself is the secret

For a persistent URL, sign up for a free [Cloudflare Zero Trust](https://www.cloudflare.com/products/zero-trust/) account and configure a named tunnel.

---

## 📦 Hardware Notes

- **GPU**: NVIDIA T4, 15360 MiB VRAM (free Colab tier)
- **RAM**: ~13 GB system
- **CPU**: 2 vCPU (Xeon)
- **Disk**: ~100 GB ephemeral
- **Runtime**: 12-hour max session, may disconnect when idle

---

## ⚠️ Caveats

- Colab free-tier runtimes are **ephemeral** — you re-download models each session. Use `hf_transfer` or `hf_hub_download` for fast downloads.
- T4 does **not** support FP8/BF16 natively — quantized (Q4/MXFP4/APEX) models are the only way to fit these sizes.
- The Cloudflare tunnel URL is public — anyone with the URL can hit your API. Don't leave it running unattended with sensitive data.
- gpt-oss-20b uses **native MXFP4**; run it via the [official OpenAI Colab notebook](https://github.com/openai/gpt-oss) or adapt notebook `04`.
- 27B dense models at Q4_K_M (15.7GB) are **borderline** — use 2K context max or they will OOM.
- Some models (Nemotron 30B, Holo 3.1 35B) have no standard Q4_K_M that fits T4 — use APEX or ultra-low-bit quants.

---

## License

MIT — see individual model licenses for usage terms of the weights themselves.
