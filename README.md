# Colab LLM Server

Run powerful open-weight LLMs on Google Colab's free T4 GPU and expose them via Cloudflare tunnels.

## Notebooks

| Notebook | Model | Size | Active | Speed (T4) |
|----------|-------|------|--------|------------|
| [01_qwen3_8b](notebooks/01_qwen3_8b_server.ipynb) | Qwen3-8B Q4_K_M | 5.0 GB | 8B dense | ~32 tok/s |
| [02_qwen3_6_35b_moe](notebooks/02_qwen3_6_35b_moe_server.ipynb) | Qwen3.6-35B-A3B APEX Nano | 10.9 GB | 3B/35B MoE | ~100 tok/s est |
| [03_gemma4_26b_moe](notebooks/03_gemma4_26b_moe_server.ipynb) | Gemma 4 26B-A4B APEX I-Mini | 12.3 GB | 3.8B/25.2B MoE | ~80 tok/s est |
| [04_gpt_oss_20b](notebooks/04_gpt_oss_20b_server.ipynb) | gpt-oss-20b MXFP4 | ~13 GB | 3.6B/21B MoE | ~60 tok/s est |
| [05_nemotron_3_nano](notebooks/05_nemotron_3_nano_server.ipynb) | Nemotron 3 Nano 30B-A3B | ~10 GB | 3B/30B MoE | ~80 tok/s est |

## Quick Start

1. Open any notebook in Google Colab (or run locally via `colab` CLI)
2. Set runtime to T4 GPU (Runtime > Change runtime type > T4 GPU)
3. Run all cells
4. Copy the `trycloudflare.com` URL from the output
5. Use it as an OpenAI-compatible API endpoint:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://your-tunnel.trycloudflare.com/v1",
    api_key="not-needed"
)
response = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## How It Works

- **llama-cpp-python** loads GGUF models with CUDA acceleration on the T4
- **APEX quantization** compresses MoE expert weights adaptively (higher precision for shared layers, lower for rarely-activated experts)
- **Cloudflare quick tunnels** create an ephemeral public URL pointing at the local server
- The API is OpenAI-compatible (`/v1/chat/completions`, `/v1/models`)

## Requirements

- Free Google Colab account (T4 GPU)
- For `colab` CLI: `pip install google-colab-cli` and `colab auth`

## Model Selection Guide

| Need | Pick |
|------|------|
| Best overall quality | Qwen3.6-35B-A3B APEX Nano |
| Multimodal (text + image) | Gemma 4 26B-A4B APEX I-Mini |
| Math/reasoning | gpt-oss-20b |
| Fastest, most headroom | Qwen3-8B Q4_K_M |
| NVIDIA agentic coding | Nemotron 3 Nano 30B-A3B |

## License

MIT - see [LICENSE](LICENSE)
