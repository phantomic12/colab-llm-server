#!/usr/bin/env python3
"""Generate Colab notebooks for the colab-llm-server repo."""
import json, os

OUT_DIR = "/tmp/colab-llm-server/notebooks"
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
# Shared: install + tunnel helper (common preamble)
# ============================================================

INSTALL_CELL = code(
    '# Install dependencies\n'
    '!pip install -q llama-cpp-python huggingface_hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n'
    '!pip install -q cloudflared\n'
    'import llama_cpp; print(f"llama-cpp-python {llama_cpp.__version__}")'
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

SERVER_CELL = code(
    '# Start the OpenAI-compatible API server\n'
    '# This cell runs the server in the background\n'
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
    '                prompt += f"<|im_start|>{{role}}\\n{{content}}<|im_end|>\\n"\n'
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

# ============================================================
# 01: Qwen3-8B Q4_K_M (proven: 32 tok/s, 5.5GB VRAM)
# ============================================================
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

make_notebook([
    md("# Qwen3-8B Server (Q4_K_M)\n\nProven 32 tok/s on Colab T4. 5GB VRAM, tons of headroom.\n\n**Model**: 8B dense, 128K context, thinking mode\n**License**: Apache 2.0"),
    INSTALL_CELL,
    download_cell("bartowski/Qwen_Qwen3-8B-GGUF", "Qwen_Qwen3-8B-Q4_K_M.gguf", "qwen3-8b", 4096),
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "01_qwen3_8b_server.ipynb")

# ============================================================
# 02: Qwen3.6-35B-A3B APEX Nano (proven: loads on T4, 10.9GB)
# ============================================================
make_notebook([
    md("# Qwen3.6-35B-A3B MoE Server (APEX Nano)\n\n**35B total, 3B active** -- the newest MoE that fits free Colab T4.\n\n- APEX Nano quant: 10.88 GB (adaptive precision for expert layers)\n- 256 experts, 8 routed per token\n- 1M context (YaRN), multimodal\n- Apache 2.0\n- Proven load on T4 in 79.8s\n\n**Note**: Download may take 10-15 min without HF token. Set `HF_TOKEN` in Colab secrets for faster downloads."),
    INSTALL_CELL,
    download_cell("mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF",
                  "Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Nano.gguf",
                  "qwen3.6-35b-a3b", 8192),
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "02_qwen3_6_35b_moe_server.ipynb")

# ============================================================
# 03: Gemma 4 26B-A4B APEX I-Mini (proven: loads on T4, 12.3GB)
# ============================================================
make_notebook([
    md("# Gemma 4 26B-A4B MoE Server (APEX I-Mini)\n\n**25.2B total, 3.8B active** -- Google's newest MoE.\n\n- APEX I-Mini: 12.27 GB\n- 128 experts, 8 active per token\n- Multimodal: text + image\n- 256K context\n- Apache 2.0\n- Proven load on T4 in 98.4s"),
    INSTALL_CELL,
    download_cell("mudler/gemma-4-26B-A4B-it-APEX-GGUF",
                  "gemma-4-26B-A4B-APEX-I-Mini.gguf",
                  "gemma4-26b-a4b", 4096),
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "03_gemma4_26b_moe_server.ipynb")

# ============================================================
# 04: gpt-oss-20b (OpenAI's open model, native MXFP4)
# ============================================================
make_notebook([
    md("# gpt-oss-20b Server (MXFP4)\n\n**21B total, 3.6B active** -- OpenAI's open-weight MoE.\n\n- Native MXFP4 quantization (trained that way, not post-hoc)\n- Designed to run in 16GB VRAM\n- 131K context, 32K max output\n- Reasoning effort: low/medium/high\n- AIME 2025: 98.7%, GPQA Diamond: 80.1%\n- Apache 2.0\n- Official OpenAI Colab cookbook exists\n\n**Note**: gpt-oss-20b uses MXFP4 which may require newer llama.cpp. If llama-cpp-python fails, use the transformers approach in cell below."),
    INSTALL_CELL,
    code(
        '# Download gpt-oss-20b GGUF\n'
        '# Using bartowski\'s GGUF -- find the right file\n'
        'from huggingface_hub import hf_hub_download, list_repo_files\n'
        'import os\n'
        '\n'
        '# Try unsloth first, then bartowski\n'
        'for repo in ["unsloth/gpt-oss-20b-GGUF", "bartowski/gpt-oss-20b-GGUF"]:\n'
        '    try:\n'
        '        files = list_repo_files(repo)\n'
        '        gguf = [f for f in files if f.endswith(".gguf")]\n'
        '        # Prefer Q4_K_M\n'
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
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "04_gpt_oss_20b_server.ipynb")

# ============================================================
# 05: Nemotron 3 Nano 30B-A3B (NVIDIA, Dec 2025)
# ============================================================
make_notebook([
    md("# Nemotron 3 Nano 30B-A3B Server\n\n**30B total, 3B active** -- NVIDIA's MoE.\n\n- Mamba2-Transformer hybrid MoE architecture\n- Based on Qwen3-VL-30B-A3B\n- 256K context\n- Open NVIDIA Model Agreement license\n- Unsloth Dynamic GGUFs available\n\n**Note**: Pick a quant that fits 15GB. UD-Q3_K_M (~15GB) is borderline; UD-IQ3_XXS (~13GB) is safer."),
    INSTALL_CELL,
    code(
        '# Download Nemotron 3 Nano 30B-A3B\n'
        'from huggingface_hub import hf_hub_download, list_repo_files\n'
        'import os\n'
        '\n'
        'repo = "unsloth/NVIDIA-Nemotron-3-Nano-30B-A3B-Reasoning-GGUF"\n'
        'files = list_repo_files(repo)\n'
        'gguf = [f for f in files if f.endswith(".gguf")]\n'
        'print("Available quants:")\n'
        'for f in sorted(gguf)[:20]:\n'
        '    print(f"  {f}")\n'
        '\n'
        '# Prefer UD-IQ3_XXS (~13GB, fits T4 comfortably)\n'
        '# Fallback to UD-Q3_K_S (~15GB, tight)\n'
        'for pattern in ["IQ3_XXS", "IQ3_XS", "Q3_K_S", "Q3_K_M", "Q4_K_S"]:\n'
        '    target = [f for f in gguf if pattern in f and "Dynamic" not in f]\n'
        '    if not target:\n'
        '        target = [f for f in gguf if pattern in f]\n'
        '    if target:\n'
        '        fname = target[0]\n'
        '        print(f"\\nDownloading: {fname}")\n'
        '        model_path = hf_hub_download(repo, fname)\n'
        '        print(f"Size: {os.path.getsize(model_path)/1e9:.2f} GB")\n'
        '        break\n'
        '\n'
        'MODEL_NAME = "nemotron-3-nano-30b"\n'
        'CONTEXT = 4096\n'
    ),
    TUNNEL_CELL,
    SERVER_CELL,
    TEST_CELL,
], "05_nemotron_3_nano_server.ipynb")

# ============================================================
# 06: Benchmark suite (the one we actually ran)
# ============================================================
make_notebook([
    md("# Benchmark Suite\n\nRun all models on a Colab T4 and compare:\n- Load time\n- VRAM usage\n- Tokens/second\n- Which models actually fit\n\nResults from our testing:\n\n| Model | Size | Load | VRAM | Speed |\n|-------|------|------|------|-------|\n| Qwen3-8B Q4_K_M | 5.0 GB | 2.7s | 5.5 GB | 32.0 tok/s |\n| Qwen3.6-35B-A3B APEX Nano | 10.9 GB | 79.8s | ~12 GB | (load proven) |\n| Gemma 4 26B-A4B APEX I-Mini | 12.3 GB | 98.4s | ~13 GB | (load proven) |"),
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

print("\nAll notebooks generated!")
