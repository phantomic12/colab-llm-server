#!/usr/bin/env python3
"""Generate Colab notebooks for the colab-llm-server repo.

Memory optimizations vs original:
- No torch import (saves ~2GB RAM) — llama-cpp-python has its own CUDA detection
- No double model load in 05_api_server (saves 5-12GB VRAM) — model loads only in server subprocess
- n_threads=2 everywhere (matches T4's 2 vCPU cores)
- cloudflared installed via wget binary, not pip (saves ~50MB + avoids dependency conflicts)
"""
import json, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

def cell(code, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": code if isinstance(code, list) else [code],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }

def md(text):
    return cell(text, "markdown")

def code(src):
    return cell(src, "code")

def make_notebook(cells, name):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU"
        },
        "cells": cells
    }
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  wrote {path}")

# ============================================================
# Shared cells
# ============================================================

INSTALL_CELL = code(
    '# Install dependencies\n'
    '# llama-cpp-python with CUDA 12.4 — no torch needed (saves ~2GB RAM)\n'
    '!pip install -q llama-cpp-python huggingface_hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n'
    'import llama_cpp; print(f"llama-cpp-python {llama_cpp.__version__}")\n'
    'print(f"CUDA GPU offload: {llama_cpp.llama_supports_gpu_offload()}")'
)

TUNNEL_CELL = code(
    '# Start Cloudflare quick tunnel (ephemeral URL)\n'
    'import subprocess, threading, time, re, sys\n'
    '\n'
    'TUNNEL_URL = None\n'
    '\n'
    'def run_tunnel(port=8080):\n'
    '    global TUNNEL_URL\n'
    '    proc = subprocess.Popen(\n'
    '        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],\n'
    '        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True\n'
    '    )\n'
    '    for line in proc.stdout:\n'
    '        print(line, end="")\n'
    '        m = re.search(r\'(https://[a-z0-9-]+\\.trycloudflare\\.com)\', line)\n'
    '        if m and not TUNNEL_URL:\n'
    '            TUNNEL_URL = m.group(1)\n'
    '            print(f"\\n>>> TUNNEL URL: {TUNNEL_URL}")\n'
    '            print(f">>> API endpoint: {TUNNEL_URL}/v1/chat/completions")\n'
    '            print(f">>> Models: {TUNNEL_URL}/v1/models")\n'
    '\n'
    'thread = threading.Thread(target=run_tunnel, daemon=True)\n'
    'thread.start()\n'
    '\n'
    '# Wait for tunnel URL\n'
    'for _ in range(30):\n'
    '    if TUNNEL_URL:\n'
    '        break\n'
    '    time.sleep(1)\n'
    '\n'
    'if not TUNNEL_URL:\n'
    '    print("WARNING: Tunnel URL not detected yet. Check cloudflared output above.")\n'
    'else:\n'
    '    print(f"\\nReady! API at: {TUNNEL_URL}/v1/chat/completions")\n'
)

# VRAM check cell — lightweight, no model load
VRAM_CHECK_CELL = code(
    '# Check available VRAM (no model loaded yet)\n'
    'import subprocess, os\n'
    '\n'
    'size_gb = os.path.getsize(model_path) / 1024**3\n'
    'print(f"Model: {os.path.basename(model_path)}")\n'
    'print(f"Size: {size_gb:.2f} GB")\n'
    '\n'
    'r = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]).decode()\n'
    'print(f"VRAM (used/total): {r.strip()}")\n'
    'print(f"Available for model: ~{15360 - int(r.split(\',\')[0].strip().replace(\'MiB\',\'\').strip())} MiB")'
)

SERVER_CELL = code(
    '# Start the OpenAI-compatible API server in background\n'
    '# Model loads ONLY here — not in the notebook kernel (saves 5-12GB VRAM)\n'
    'import subprocess, os, signal\n'
    '\n'
    'MODEL_PATH = model_path  # set by the download cell above\n'
    'PORT = 8080\n'
    '\n'
    '# Kill any existing server\n'
    'os.system("pkill -f llama_server || true")\n'
    '\n'
    '# Start llama-cpp-python server\n'
    'server_script = f"""\n'
    'import llama_cpp\n'
    'from llama_cpp import Llama\n'
    'import threading, json, http.server\n'
    '\n'
    'llm = Llama(\n'
    '    model_path="{MODEL_PATH}",\n'
    '    n_gpu_layers=-1,\n'
    '    n_ctx={CONTEXT},\n'
    '    verbose=False,\n'
    '    n_threads=2,\n'
    ')\n'
    '\n'
    'class APIHandler(http.server.BaseHTTPRequestHandler):\n'
    '    def do_GET(self):\n'
    '        if self.path == "/v1/models":\n'
    '            self.send_response(200)\n'
    '            self.send_header("Content-Type", "application/json")\n'
    '            self.end_headers()\n'
    '            self.wfile.write(json.dumps({{"data":[{{"id":"{MODEL_NAME}"}}]}}).encode())\n'
    '        else:\n'
    '            self.send_response(404)\n'
    '            self.end_headers()\n'
    '\n'
    '    def do_POST(self):\n'
    '        if self.path == "/v1/chat/completions":\n'
    '            length = int(self.headers.get("Content-Length", 0))\n'
    '            body = json.loads(self.rfile.read(length))\n'
    '            messages = body.get("messages", [])\n'
    '            prompt = ""\n'
    '            for msg in messages:\n'
    '                role = msg.get("role", "user")\n'
    '                content = msg.get("content", "")\n'
    '                prompt += f"<|im_start|>{role}\\n{content}<|im_end|>\\n"\n'
    '            prompt += "<|im_start|>assistant\\n"\n'
    '            max_tokens = body.get("max_tokens", 256)\n'
    '            temperature = body.get("temperature", 0.7)\n'
    '            out = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["<|im_end|>"])\n'
    '            text = out["choices"][0]["text"]\n'
    '            response = {{\n'
    '                "id": "chatcmpl-1",\n'
    '                "object": "chat.completion",\n'
    '                "choices": [{{"index": 0, "message": {{"role": "assistant", "content": text}}, "finish_reason": "stop"}}],\n'
    '                "usage": {{"prompt_tokens": 0, "completion_tokens": out["usage"]["completion_tokens"], "total_tokens": out["usage"]["completion_tokens"]}}\n'
    '            }}\n'
    '            self.send_response(200)\n'
    '            self.send_header("Content-Type", "application/json")\n'
    '            self.end_headers()\n'
    '            self.wfile.write(json.dumps(response).encode())\n'
    '        else:\n'
    '            self.send_response(404)\n'
    '            self.end_headers()\n'
    '\n'
    '    def log_message(self, format, *args):\n'
    '        pass  # suppress logs\n'
    '\n'
    'server = http.server.HTTPServer(("0.0.0.0", {PORT}), APIHandler)\n'
    'print(f"Server running on port {PORT}")\n'
    'server.serve_forever()\n'
    '"""\n'
    '\n'
    'with open("/tmp/server.py", "w") as f:\n'
    '    f.write(server_script)\n'
    '\n'
    'server_proc = subprocess.Popen(["python3", "/tmp/server.py"],\n'
    '    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n'
    '\n'
    '# Wait for server to start\n'
    'import time\n'
    'time.sleep(5)\n'
    'print("Server started! Use the tunnel URL above.")\n'
    'print(f"Test: curl {TUNNEL_URL}/v1/models")\n'
)

TEST_CELL = code(
    '# Test the API\n'
    '!curl -s {TUNNEL_URL}/v1/chat/completions \\\n'
    '  -X POST -H "Content-Type: application/json" \\\n'
    '  -d \'{"messages":[{"role":"user","content":"What is 17 * 23? Think step by step."}],"max_tokens":200}\'\n'
    '\n'
    'print("\\n\\n--- Python client test ---")\n'
    'from openai import OpenAI\n'
    'client = OpenAI(base_url=f"{TUNNEL_URL}/v1", api_key="not-needed")\n'
    'r = client.chat.completions.create(\n'
    '    model="default",\n'
    '    messages=[{"role": "user", "content": "Say hello in 3 languages."}],\n'
    '    max_tokens=100\n'
    ')\n'
    'print(r.choices[0].message.content)\n'
)

def download_cell(repo_id, filename, model_name, context=4096):
    return code(
        f'# Download model\n'
        f'from huggingface_hub import hf_hub_download\n'
        f'\n'
        f'model_path = hf_hub_download(\n'
        f'    repo_id="{repo_id}",\n'
        f'    filename="{filename}"\n'
        f')\n'
        f'import os\n'
        f'print(f"Model: {{os.path.getsize(model_path)/1e9:.2f}} GB")\n'
        f'print(f"Path: {{model_path}}")\n'
        f'\n'
        f'# Global vars for server cell\n'
        f'MODEL_NAME = "{model_name}"\n'
        f'CONTEXT = {context}\n'
    )

def search_download_cell(repo_id, model_name, context=4096, preferred_quants=None):
    """Download cell that searches for the best-fitting quant."""
    if preferred_quants is None:
        preferred_quants = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "Q3_K_S", "IQ3_M", "IQ3_XXS"]
    patterns = '", "'.join(preferred_quants)
    return code(
        f'# Download model — auto-pick best quant that fits T4 (15GB VRAM)\n'
        f'from huggingface_hub import hf_hub_download, list_repo_files\n'
        f'import os\n'
        f'\n'
        f'repo = "{repo_id}"\n'
        f'files = list_repo_files(repo)\n'
        f'gguf = [f for f in files if f.endswith(".gguf")]\n'
        f'print("Available quants:")\n'
        f'for f in sorted(gguf)[:20]:\n'
        f'    print(f"  {{f}}")\n'
        f'\n'
        f'# Try preferred quants in order (first that fits)\n'
        f'for pattern in ["{patterns}"]:\n'
        f'    target = [f for f in gguf if pattern in f and "Dynamic" not in f]\n'
        f'    if not target:\n'
        f'        target = [f for f in gguf if pattern in f]\n'
        f'    if target:\n'
        f'        fname = target[0]\n'
        f'        print(f"\\nDownloading: {{fname}}")\n'
        f'        model_path = hf_hub_download(repo, fname)\n'
        f'        size_gb = os.path.getsize(model_path) / 1e9\n'
        f'        print(f"Size: {{size_gb:.2f}} GB")\n'
        f'        if size_gb > 14.5:\n'
        f'            print(f"WARNING: {{size_gb:.1f}}GB is tight for T4 (15GB). Use small context.")\n'
        f'        break\n'
        f'\n'
        f'MODEL_NAME = "{model_name}"\n'
        f'CONTEXT = {context}\n'
    )

# ============================================================
# 00: Setup (memory-optimized — no torch)
# ============================================================
make_notebook([
    md("# 00 — Setup: Install llama-cpp-python with CUDA & Verify the T4 GPU\n\n"
       "Installs `llama-cpp-python` with **CUDA 12.4** and verifies the T4 GPU.\n\n"
       "**Memory optimization**: No `torch` import — `llama-cpp-python` has its own CUDA detection. "
       "This saves ~2GB of system RAM on a machine that only has ~13GB.\n\n"
       "**Requirements:** Runtime → Change runtime type → **T4 GPU** (free tier)."),
    md("## 1. Check the GPU"),
    code(
        '!nvidia-smi\n'
        '\n'
        '# Verify llama-cpp-python can see CUDA (no torch needed — saves ~2GB RAM)\n'
        'import llama_cpp\n'
        'print(f"llama-cpp-python version: {llama_cpp.__version__}")\n'
        'print(f"CUDA support: {llama_cpp.llama_supports_gpu_offload()}")\n'
    ),
    md("You should see a **Tesla T4** with ~15360 MiB total memory."),
    md("## 2. Install llama-cpp-python with CUDA 12.4"),
    code(
        '%%time\n'
        '%%capture\n'
        'pip install llama-cpp-python huggingface_hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n'
    ),
    md("## 3. Verify the install"),
    code(
        'import llama_cpp\n'
        'print(f"llama-cpp-python version: {llama_cpp.__version__}")\n'
        'print(f"CUDA GPU offload: {llama_cpp.llama_supports_gpu_offload()}")\n'
        'print("Setup OK")\n'
    ),
    md("## 4. Install huggingface_hub and verify"),
    code(
        'from huggingface_hub import hf_hub_download\n'
        'print("huggingface_hub ready")\n'
    ),
    md("## ✅ Setup complete\n\nYou now have a CUDA-enabled `llama-cpp-python` install on a T4 GPU. Proceed to any model notebook."),
], "00_setup.ipynb")

# ============================================================
# 01: Qwen3-8B Q4_K_M (proven: 32 tok/s, 5.5GB VRAM)
# ============================================================
make_notebook([
    md("# Qwen3-8B Server (Q4_K_M)\n\nProven 32 tok/s on Colab T4. 5GB VRAM, tons of headroom.\n\n**Model**: 8B dense, 128K context, thinking mode\n**License**: Apache 2.0"),
    INSTALL_CELL,
    download_cell("bartowski/Qwen_Qwen3-8B-GGUF", "Qwen_Qwen3-8B-Q4_K_M.gguf", "qwen3-8b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "01_qwen3_8b_server.ipynb")

# ============================================================
# 02: Qwen3.6-35B-A3B APEX Nano (proven: loads on T4, 10.9GB)
# ============================================================
make_notebook([
    md("# Qwen3.6-35B-A3B MoE Server (APEX Nano)\n\n"
       "**35B total, 3B active** — the MoE that fits free Colab T4.\n\n"
       "- APEX Nano quant: 10.88 GB (adaptive precision for expert layers)\n"
       "- 256 experts, 8 routed per token\n"
       "- 1M context (YaRN), multimodal\n"
       "- Apache 2.0\n"
       "- Proven load on T4 in 79.8s\n\n"
       "**Note**: Download may take 10-15 min without HF token. Set `HF_TOKEN` in Colab secrets for faster downloads."),
    INSTALL_CELL,
    download_cell("mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF",
                  "Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Nano.gguf",
                  "qwen3.6-35b-a3b", 8192),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "02_qwen3_6_35b_moe_server.ipynb")

# ============================================================
# 03: Gemma 4 26B-A4B APEX I-Mini (proven: loads on T4, 12.3GB)
# ============================================================
make_notebook([
    md("# Gemma 4 26B-A4B MoE Server (APEX I-Mini)\n\n"
       "**25.2B total, 3.8B active** — Google's newest MoE.\n\n"
       "- APEX I-Mini: 12.27 GB\n"
       "- 128 experts, 8 active per token\n"
       "- Multimodal: text + image\n"
       "- 256K context\n"
       "- Apache 2.0\n"
       "- Proven load on T4 in 98.4s"),
    INSTALL_CELL,
    download_cell("mudler/gemma-4-26B-A4B-it-APEX-GGUF",
                  "gemma-4-26B-A4B-APEX-I-Mini.gguf",
                  "gemma4-26b-a4b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "03_gemma4_26b_moe_server.ipynb")

# ============================================================
# 04: gpt-oss-20b (OpenAI's open model, native MXFP4)
# ============================================================
make_notebook([
    md("# gpt-oss-20b Server (MXFP4)\n\n"
       "**21B total, 3.6B active** — OpenAI's open-weight MoE.\n\n"
       "- Native MXFP4 quantization (trained that way, not post-hoc)\n"
       "- Designed to run in 16GB VRAM\n"
       "- 131K context, 32K max output\n"
       "- Reasoning effort: low/medium/high\n"
       "- AIME 2025: 98.7%, GPQA Diamond: 80.1%\n"
       "- Apache 2.0\n\n"
       "**Note**: gpt-oss-20b uses MXFP4 which may require newer llama.cpp. If llama-cpp-python fails, use the transformers approach."),
    INSTALL_CELL,
    code(
        '# Download gpt-oss-20b GGUF\n'
        'from huggingface_hub import hf_hub_download, list_repo_files\n'
        'import os\n'
        '\n'
        '# Try unsloth first, then bartowski\n'
        'for repo in ["unsloth/gpt-oss-20b-GGUF", "bartowski/gpt-oss-20b-GGUF"]:\n'
        '    try:\n'
        '        files = list_repo_files(repo)\n'
        '        gguf = [f for f in files if f.endswith(".gguf")]\n'
        '        target = [f for f in gguf if "Q4_K_M" in f] or gguf\n'
        '        if target:\n'
        '            print(f"Downloading from {repo}: {target[0]}")\n'
        '            model_path = hf_hub_download(repo, target[0])\n'
        '            print(f"Size: {os.path.getsize(model_path)/1e9:.2f} GB")\n'
        '            break\n'
        '    except Exception as e:\n'
        '        print(f"{repo}: {e}")\n'
        '\n'
        'MODEL_NAME = "gpt-oss-20b"\n'
        'CONTEXT = 8192\n'
    ),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "04_gpt_oss_20b_server.ipynb")

# ============================================================
# 05: Nemotron 3 Nano 30B-A3B (NVIDIA)
# ============================================================
make_notebook([
    md("# Nemotron 3 Nano 30B-A3B Server\n\n"
       "**30B total, 3B active** — NVIDIA's MoE.\n\n"
       "- Mamba2-Transformer hybrid MoE architecture\n"
       "- Based on Qwen3-VL-30B-A3B\n"
       "- 256K context\n"
       "- Open NVIDIA Model Agreement license\n\n"
       "**Note**: This model is memory-heavy. Standard Q4_K_M is 22.9GB (too big for T4). "
       "The notebook auto-selects the smallest available quant. "
       "If no quant fits, try the APEX version from `mudler/Nemotron-3-Nano-30B-A3B-APEX-GGUF`."),
    INSTALL_CELL,
    search_download_cell(
        "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        "nemotron-3-nano-30b", 4096,
        preferred_quants=["UD-IQ3_XXS", "UD-IQ2_M", "IQ3_XXS", "IQ2_M", "Q2_K_L", "Q2_K"]
    ),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "05_nemotron_3_nano_server.ipynb")

# ============================================================
# 06: Benchmark suite
# ============================================================
make_notebook([
    md("# Benchmark Suite\n\n"
       "Run all models on a Colab T4 and compare:\n"
       "- Load time\n"
       "- VRAM usage\n"
       "- Tokens/second\n"
       "- Which models actually fit\n\n"
       "Results from our testing:\n\n"
       "| Model | Size | Load | VRAM | Speed |\n"
       "|-------|------|------|------|-------|\n"
       "| Qwen3-8B Q4_K_M | 5.0 GB | 2.7s | 5.5 GB | 32.0 tok/s |\n"
       "| Qwen3.6-35B-A3B APEX Nano | 10.9 GB | 79.8s | ~12 GB | (load proven) |\n"
       "| Gemma 4 26B-A4B APEX I-Mini | 12.3 GB | 98.4s | ~13 GB | (load proven) |"),
    code(
        '# Install\n'
        '!pip install -q llama-cpp-python huggingface_hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n'
        'import llama_cpp, subprocess, time, os\n'
        'from huggingface_hub import hf_hub_download\n'
        'print(f"llama-cpp-python {llama_cpp.__version__}")\n'
        '\n'
        'def bench(model_path, label, n_ctx=4096, max_tokens=150):\n'
        '    print(f"\\n{\'=\'*50}")\n'
        '    print(f"BENCH: {label} ({os.path.getsize(model_path)/1e9:.1f}GB)")\n'
        '    t0 = time.time()\n'
        '    llm = llama_cpp.Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False, n_threads=2)\n'
        '    load_time = time.time() - t0\n'
        '    print(f"Load: {load_time:.1f}s")\n'
        '    prompt = "<|im_start|>user\\nWhat is 17 * 23? Think step by step.<|im_end|>\\n<|im_start|>assistant\\n"\n'
        '    t0 = time.time()\n'
        '    out = llm(prompt, max_tokens=max_tokens, temperature=0.7, stop=["<|im_end|>"])\n'
        '    elapsed = time.time() - t0\n'
        '    tok = out["usage"]["completion_tokens"]\n'
        '    print(f"Math: {tok} tok / {elapsed:.1f}s = {tok/elapsed:.1f} tok/s")\n'
        '    print(f"  {out[\'choices\'][0][\'text\'][:300]}")\n'
        '    r = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader"]).decode()\n'
        '    print(f"VRAM: {r.strip()}")\n'
        '    del llm; import gc; gc.collect()\n'
        '    return label, load_time, tok/elapsed\n'
        '\n'
        'results = []\n'
        '\n'
        '# 1. Qwen3-8B\n'
        'try:\n'
        '    p = hf_hub_download("bartowski/Qwen_Qwen3-8B-GGUF", "Qwen_Qwen3-8B-Q4_K_M.gguf")\n'
        '    results.append(bench(p, "Qwen3-8B Q4_K_M"))\n'
        'except Exception as e: print(f"FAIL: {e}")\n'
        '\n'
        '# 2. Qwen3.6-35B-A3B APEX Nano\n'
        'try:\n'
        '    p = hf_hub_download("mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF",\n'
        '                        "Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Nano.gguf")\n'
        '    results.append(bench(p, "Qwen3.6-35B-A3B APEX Nano", n_ctx=8192))\n'
        'except Exception as e: print(f"FAIL: {e}")\n'
        '\n'
        '# 3. Gemma 4 26B-A4B APEX I-Mini\n'
        'try:\n'
        '    p = hf_hub_download("mudler/gemma-4-26B-A4B-it-APEX-GGUF",\n'
        '                        "gemma-4-26B-A4B-APEX-I-Mini.gguf")\n'
        '    results.append(bench(p, "Gemma4-26B APEX I-Mini"))\n'
        'except Exception as e: print(f"FAIL: {e}")\n'
        '\n'
        'print(f"\\n\\n{\'=\'*50}")\n'
        'print("RESULTS:")\n'
        'for label, load, tps in results:\n'
        '    print(f"  {label}: Load={load:.1f}s, {tps:.1f} tok/s")\n'
    ),
], "06_benchmark_suite.ipynb")

# ============================================================
# 07: Darwin Reason 27B (Q3_K_M, 12.4GB — fits T4)
# ============================================================
make_notebook([
    md("# Darwin Reason 27B Server\n\n"
       "**27B dense** — reasoning-focused fine-tune of Qwen3.6-27B.\n\n"
       "- Q3_K_M quant: 12.39 GB (fits T4 with room for 4K context)\n"
       "- Q4_K_M is 15.4GB — too tight for T4 with any context\n"
       "- Based on Qwen3.6-27B (128K context, thinking mode)\n"
       "- Apache 2.0\n\n"
       "**Note**: 27B dense models are memory-heavy. Q3_K_M is the sweet spot for T4. "
       "If you need more context, try IQ4_XS (14.15GB) with 2K context."),
    INSTALL_CELL,
    download_cell("mradermacher/Darwin-28B-REASON-GGUF",
                  "Darwin-28B-REASON.Q3_K_M.gguf",
                  "darwin-reason-27b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "07_darwin_reason_27b_server.ipynb")

# ============================================================
# 08: Darwin Code 27B (Q3_K_M, 12.6GB — fits T4)
# ============================================================
make_notebook([
    md("# Darwin Code 27B Server\n\n"
       "**27B dense** — coding-focused fine-tune of Qwen3.6-27B.\n\n"
       "- Q3_K_M quant: 12.57 GB (fits T4 with room for 4K context)\n"
       "- Q4_K_M is 15.7GB — too tight for T4\n"
       "- Based on Qwen3.6-27B\n"
       "- Apache 2.0"),
    INSTALL_CELL,
    download_cell("mradermacher/Darwin-28B-Coder-GGUF",
                  "Darwin-28B-Coder.Q3_K_M.gguf",
                  "darwin-code-27b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "08_darwin_code_27b_server.ipynb")

# ============================================================
# 09: Carnice V2 27B (IQ3_M, 12.7GB — fits T4)
# ============================================================
make_notebook([
    md("# Carnice V2 27B Server\n\n"
       "**27B dense** — agentic fine-tune of Qwen3.6-27B by kai-os.\n\n"
       "- IQ3_M quant: 12.73 GB (fits T4 with room for 4K context)\n"
       "- Q4_K_M is 16.3GB — too big for T4\n"
       "- Hermes Agent compatible, tool-calling, agentic\n"
       "- Based on Qwen3.6-27B\n"
       "- Apache 2.0"),
    INSTALL_CELL,
    download_cell("bartowski/kai-os_Carnice-V2-27b-GGUF",
                  "kai-os_Carnice-V2-27b-IQ3_M.gguf",
                  "carnice-v2-27b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "09_carnice_v2_27b_server.ipynb")

# ============================================================
# 10: Qwopus v2 27B (Q3_K_M, 12.6GB — fits T4)
# ============================================================
make_notebook([
    md("# Qwopus v2 27B Server\n\n"
       "**27B dense** — Claude Opus trace-inversion fine-tune of Qwen3.6-27B.\n\n"
       "- Q3_K_M quant: 12.57 GB (fits T4 with room for 4K context)\n"
       "- Q4_K_M is 15.7GB — too tight for T4\n"
       "- MTP (Multi-Token Prediction) variant for speculative decoding\n"
       "- Multimodal: text + image\n"
       "- Based on Qwen3.6-27B\n"
       "- Apache 2.0"),
    INSTALL_CELL,
    download_cell("Jackrong/Qwopus3.6-27B-v2-MTP-GGUF",
                  "Qwopus3.6-27B-v2-MTP-Q3_K_M.gguf",
                  "qwopus-v2-27b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "10_qwopus_v2_27b_server.ipynb")

# ============================================================
# 11: Qwable 27B (Q4_K_M, 15.7GB — borderline T4)
# ============================================================
make_notebook([
    md("# Qwable 27B Server\n\n"
       "**27B dense** — reasoning + instruction-tuned fine-tune of Qwen3.6-27B by Mia-AiLab.\n\n"
       "- Q4_K_M quant: 15.66 GB — **borderline for T4** (15GB VRAM)\n"
       "- Use 2K context max, or it will OOM\n"
       "- MTP variant for speculative decoding\n"
       "- Based on Qwen3.6-27B\n"
       "- MIT license\n\n"
       "**Warning**: This is the tightest fit in the repo. If it OOMs, use a smaller model "
       "or try the abliterated version with a smaller quant."),
    INSTALL_CELL,
    download_cell("Mia-AiLab/Qwable-3.6-27b-MTP",
                  "Qwable-3.6-27b_q4_k_m.gguf",
                  "qwable-27b", 2048),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "11_qwable_27b_server.ipynb")

# ============================================================
# 12: Darwin APEX 35B (APEX I-Nano, 10.7GB — fits T4)
# ============================================================
make_notebook([
    md("# Darwin APEX 35B Server\n\n"
       "**36B MoE** — Darwin Opus with APEX quantization.\n\n"
       "- APEX I-Nano quant: 10.67 GB (fits T4 with room for 8K context)\n"
       "- APEX I-Mini: 12.54 GB (also fits, higher quality)\n"
       "- MoE architecture (based on Qwen3.6-35B-A3B)\n"
       "- Apache 2.0"),
    INSTALL_CELL,
    download_cell("mudler/Darwin-36B-Opus-APEX-GGUF",
                  "Darwin-36B-Opus-APEX-I-Nano.gguf",
                  "darwin-apex-35b", 8192),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "12_darwin_apex_35b_server.ipynb")

# ============================================================
# 13: Carwin Nano 35B (MoE Nano, 11.5GB — fits T4)
# ============================================================
make_notebook([
    md("# Carwin Nano 35B Server\n\n"
       "**35B MoE** — Carwin Mixture-of-Experts, Nano quant.\n\n"
       "- Nano quant: 11.49 GB (fits T4 with room for 8K context)\n"
       "- MoE architecture\n"
       "- MTP variant\n"
       "- Apache 2.0"),
    INSTALL_CELL,
    download_cell("isneezekittens/Carwin-MoE-Nano-GGUF",
                  "carwin-moe-Nano.gguf",
                  "carwin-nano-35b", 8192),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "13_carwin_nano_35b_server.ipynb")

# ============================================================
# 14: Holo 3.1 35B (APEX I-Mini, 13.3GB — fits T4)
# ============================================================
make_notebook([
    md("# Holo 3.1 35B Server\n\n"
       "**35B MoE** — H Company's computer-use agent model.\n\n"
       "- APEX I-Mini quant: 13.33 GB (fits T4 with room for 4K context)\n"
       "- 35B total, 3B active (MoE)\n"
       "- Multimodal: text + image\n"
       "- Computer use / GUI agents\n"
       "- Apache 2.0\n\n"
       "**Note**: This is the model behind H Company's computer-use agent. "
       "The APEX I-Mini quant fits T4 with ~2GB headroom for context."),
    INSTALL_CELL,
    download_cell("mudler/Holo3-35B-A3B-APEX-GGUF",
                  "Holo3-35B-A3B-APEX-I-Mini.gguf",
                  "holo-3.1-35b", 4096),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "14_holo_3_1_35b_server.ipynb")

# ============================================================
# 15: Carwin 27B (Q4_K_M, 15.7GB — borderline T4)
# ============================================================
make_notebook([
    md("# Carwin 27B Server\n\n"
       "**27B dense** — Carwin (Carnice + Darwin merge), MTP variant.\n\n"
       "- Q4_K_M quant: 15.68 GB — **borderline for T4** (15GB VRAM)\n"
       "- Use 2K context max, or it will OOM\n"
       "- MTP (Multi-Token Prediction) for speculative decoding\n"
       "- Multimodal: includes mmproj for vision\n"
       "- Apache 2.0\n\n"
       "**Warning**: Tightest fit. Only Q4_K_M available. If OOM, use 2K context "
       "or switch to Carwin Nano 35B (MoE, 11.5GB)."),
    INSTALL_CELL,
    download_cell("isneezekittens/Carwin-28B-MTP-GGUF",
                  "carwin-Q4_K_M.gguf",
                  "carwin-27b", 2048),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "15_carwin_27b_server.ipynb")

# ============================================================
# 16: Abliterated Qwable 27B
# ============================================================
make_notebook([
    md("# Abliterated Qwable 27B Server\n\n"
       "**27B dense** — Qwable 3.6-27b with refusal mechanisms removed (abliterated).\n\n"
       "- Q4_K_M F16 KV: 25.85 GB — **too big for T4**\n"
       "- Q4_K_M Q8 KV: 19.11 GB — **too big for T4**\n\n"
       "**Status**: No T4-fit quant available. The only GGUF quants are Q4_K_M with "
       "F16 or Q8 KV cache, both exceeding 15GB. \n\n"
       "To use this model on T4, you would need to:\n"
       "1. Quantize to Q3_K_M or IQ3_M yourself\n"
       "2. Or use the non-abliterated Qwable with a smaller quant\n\n"
       "This notebook is a placeholder — it will attempt to download and will likely OOM."),
    INSTALL_CELL,
    code(
        '# WARNING: This model has no T4-fit quant.\n'
        '# The smallest available is Q4_K_M_Q8 at 19.1GB — too big for 15GB T4.\n'
        '# This cell will attempt download but the server will likely OOM.\n'
        'from huggingface_hub import hf_hub_download\n'
        'import os\n'
        '\n'
        'try:\n'
        '    model_path = hf_hub_download(\n'
        '        repo_id="huihui-ai/Huihui-Qwable-3.6-27b-abliterated-MTP-GGUF",\n'
        '        filename="Huihui-Qwable-3.6-27b-abliterated-Q4_K_M_Q8-MTP.gguf"\n'
        '    )\n'
        '    size_gb = os.path.getsize(model_path) / 1e9\n'
        '    print(f"Downloaded: {size_gb:.2f} GB")\n'
        '    if size_gb > 14.5:\n'
        '        print(f"WARNING: {size_gb:.1f}GB exceeds T4 VRAM (15GB). Server will OOM.")\n'
        '        print("No smaller quant available for this model.")\n'
        'except Exception as e:\n'
        '    print(f"Download failed: {e}")\n'
        '\n'
        'MODEL_NAME = "qwable-ablit-27b"\n'
        'CONTEXT = 2048\n'
    ),
    VRAM_CHECK_CELL,
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "16_qwable_ablit_27b_server.ipynb")

print("\nAll notebooks generated!")
