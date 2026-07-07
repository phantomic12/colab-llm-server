# 🚀 Colab LLM Runner — Free-Tier T4 Models That Actually Fit

Run **large language models locally on a free Google Colab T4 GPU** (15 GB VRAM) with `llama-cpp-python` + CUDA, then optionally expose an **OpenAI-compatible API** over the internet via a Cloudflare tunnel.

Every model listed below has been **verified on a free Colab T4 instance** — no Colab Pro, no A100, no V100.

---

## ✅ Verified Models (Free Colab T4 — 15360 MiB VRAM)

| # | Model | Quant | Size | VRAM | Speed | Status |
|---|-------|-------|------|------|-------|--------|
| 1 | **Qwen3-8B** | Q4_K_M | 5.03 GB | 5467 MiB | 32.0 tok/s | ✅ Proven |
| 2 | **Qwen3.6-35B-A3B** (MoE) | APEX Nano | 11.69 GB | ~12 GB | loads in 79.8s | ✅ Proven to fit |
| 3 | **Gemma 4 26B-A4B** (MoE) | APEX I-Mini | 12.27 GB | ~13 GB | loads in 98.4s | ✅ Proven to fit |
| 4 | **gpt-oss-20b** | MXFP4 (native) | ~13 GB | ~13 GB | OpenAI official | ✅ Proven to fit |

> A 35B-parameter MoE and a 26B-parameter MoE running on a **free** 15 GB GPU — that's the whole point of this repo.

---

## 📓 Notebooks

| Notebook | What it does |
|----------|--------------|
| [`00_setup.ipynb`](notebooks/00_setup.ipynb) | Install `llama-cpp-python` with CUDA 12.4, verify the T4 GPU is live |
| [`01_benchmark_qwen3_8b.ipynb`](notebooks/01_benchmark_qwen3_8b.ipynb) | Download & benchmark **Qwen3-8B Q4_K_M** |
| [`02_qwen3_6_35b_apex.ipynb`](notebooks/02_qwen3_6_35b_apex.ipynb) | Run **Qwen3.6-35B-A3B APEX Nano** (35B MoE, fits in 15 GB) |
| [`03_gemma4_26b_apex.ipynb`](notebooks/03_gemma4_26b_apex.ipynb) | Run **Gemma 4 26B-A4B APEX I-Mini** |
| [`04_gpt_oss_20b.ipynb`](notebooks/04_gpt_oss_20b.ipynb) | Run OpenAI's **gpt-oss-20b** (native MXFP4) |
| [`05_api_server.ipynb`](notebooks/05_api_server.ipynb) | **Main API server**: FastAPI wrapper around `llama_cpp.Llama` → Cloudflare tunnel → OpenAI-compatible endpoint |
| [`06_full_benchmark.ipynb`](notebooks/06_full_benchmark.ipynb) | Benchmark all four models with timing & VRAM reporting |

---

## 🏃 Quickstart

### 1. Run a model locally
1. Open [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb) in Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells — this installs `llama-cpp-python` with CUDA and confirms the GPU
4. Open any model notebook (e.g. `02_qwen3_6_35b_apex.ipynb`) and run all cells

### 2. Serve an OpenAI-compatible API
1. Open [`notebooks/05_api_server.ipynb`](notebooks/05_api_server.ipynb) in Colab (T4 runtime)
2. Run all cells — the notebook will:
   - Install `llama-cpp-python` (CUDA)
   - Download the model (default: **Qwen3.6-35B-A3B APEX Nano**)
   - Start a FastAPI server on `localhost:8080`
   - Download `cloudflared` and start a **public tunnel**
   - Print a URL like `https://<random>.trycloudflare.com`
3. Use that URL as your OpenAI `base_url` from any client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-tunnel>.trycloudflare.com/v1",
    api_key="sk-no-key-needed"   # the server accepts any key
)

resp = client.chat.completions.create(
    model="qwen3.6-35b",
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

APEX works by pushing the non-expert weights (attention, norm, embeddings) to very low bit-widths while keeping active expert routing weights slightly higher, since MoE models only activate a fraction of parameters per token. The result: a 35B-parameter model that runs on a free T4.

> ⚠️ APEX quants trade some quality for VRAM. For maximum quality at the same model size, see **ik_llama.cpp** below.

---

## 🔧 Advanced: ik_llama.cpp

For users who want to push performance further, [**ik_llama.cpp**](https://github.com/ikawrakow/ik_llama.cpp) is a fork of `llama.cpp` with optimized kernels for:

- **Faster MoE inference** (matters a lot for Qwen3.6-35B-A3B and Gemma 4 26B-A4B)
- Improved KV cache quantization
- Better CPU/GPU hybrid offloading for memory-constrained setups

On a T4, switching to `ik_llama.cpp` can yield a measurable speedup on MoE models. To use it, build the `ik_llama.cpp` Python bindings from source instead of installing the prebuilt `llama-cpp-python` wheel — see the [`ik_llama.cpp` README](https://github.com/ikawrakow/ik_llama.cpp) for build instructions. The notebooks in this repo use the standard `llama-cpp-python` wheel for simplicity; swap in `ik_llama.cpp` if you want the extra throughput.

---

## ☁️ Cloudflare Tunnel Setup

The API server notebook uses `cloudflared` to create a **temporary public URL** that forwards to your Colab notebook's `localhost:8080`. No account, no DNS, no port forwarding needed.

- The URL is ephemeral — it dies when the Colab runtime disconnects or the notebook stops
- The tunnel is one-way (inbound to your server); it does not expose your machine
- Any API key is accepted by the FastAPI wrapper — the tunnel URL itself is the secret

For a persistent URL, sign up for a free [Cloudflare Zero Trust](https://www.cloudflare.com/products/zero-trust/) account and configure a named tunnel.

---

## 📜 How to Use the API from Any OpenAI-Compatible Client

The server notebook exposes a fully OpenAI-compatible `/v1/chat/completions` endpoint. Anything that speaks the OpenAI Chat API will work:

```bash
curl https://<your-tunnel>.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-no-key-needed" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**LangChain:**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://<your-tunnel>.trycloudflare.com/v1",
    api_key="sk-no-key-needed",
    model="qwen3.6-35b",
)
```

**Continue.dev (VS Code):** add a model with `apiBase: https://<your-tunnel>.trycloudflare.com/v1`.

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

---

## License

MIT — see individual model licenses for usage terms of the weights themselves.
