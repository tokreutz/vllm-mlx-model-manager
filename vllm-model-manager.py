#!/usr/bin/env python3
"""
vLLM-MLX Model Manager - Dynamic model loading wrapper
Provides LM Studio-like API with on-demand model loading for vLLM-MLX backend
"""

import subprocess
import time
import requests
import json
from typing import Optional, Dict
from flask import Flask, request, jsonify, Response
import threading
import signal
import sys
import os

app = Flask(__name__)

class VLLMModelManager:
    def __init__(self, port=8001, cache_dir="~/.lmstudio/models"):
        self.port = port
        self.cache_dir = os.path.expanduser(cache_dir)
        self.current_model: Optional[str] = None
        self.vllm_process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        
    def is_model_loaded(self) -> bool:
        """Check if vLLM server is responding"""
        try:
            resp = requests.get(f"http://localhost:{self.port}/health", timeout=1)
            return resp.status_code == 200
        except:
            return False
    
    def stop_current_model(self):
        """Stop currently running vLLM server"""
        if self.vllm_process:
            print(f"Stopping model: {self.current_model}")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
            self.vllm_process = None
            self.current_model = None
            time.sleep(1)  # Let port free up
    
    def load_model(self, model_name: str, continuous_batching: bool = False):
        """Load a model via vLLM-MLX"""
        with self.lock:
            # Already loaded?
            if self.current_model == model_name and self.is_model_loaded():
                print(f"Model already loaded: {model_name}")
                return
            
            # Stop existing model
            if self.vllm_process:
                self.stop_current_model()
            
            # Start new model
            print(f"Loading model: {model_name}")
            
            # Check if model exists in LM Studio cache (case-insensitive fuzzy match)
            local_path = None
            if "/" in model_name:
                org, model = model_name.split("/", 1)
                org_dir = os.path.join(self.cache_dir, org)
                if os.path.exists(org_dir):
                    # Try exact match first
                    exact_path = os.path.join(org_dir, model)
                    if os.path.exists(exact_path):
                        local_path = exact_path
                    else:
                        # Try case-insensitive fuzzy match
                        model_lower = model.lower()
                        for item in os.listdir(org_dir):
                            if item.lower().startswith(model_lower.replace("-", "")):
                                local_path = os.path.join(org_dir, item)
                                print(f"Fuzzy matched: {model} → {item}")
                                break
            
            if local_path and os.path.exists(local_path):
                print(f"Found model in LM Studio cache: {local_path}")
                model_arg = local_path
            else:
                print(f"Model not in cache, will download from HuggingFace")
                model_arg = model_name
            
            # Don't set HF_HOME if using local path (avoids confusion)
            env = os.environ.copy()
            
            cmd = [
                "vllm-mlx", "serve", model_arg,
                "--port", str(self.port)
            ]
            if continuous_batching:
                cmd.append("--continuous-batching")
            
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr to stdout
                text=True,
                env=env,
                bufsize=1  # Line buffered
            )
            
            # Stream output in background thread
            def log_output():
                for line in self.vllm_process.stdout:
                    print(f"[vLLM] {line.rstrip()}")
            
            log_thread = threading.Thread(target=log_output, daemon=True)
            log_thread.start()
            
            self.current_model = model_name
            
            # Wait for server to be ready
            max_wait = 120  # 2 minutes
            start = time.time()
            while time.time() - start < max_wait:
                if self.is_model_loaded():
                    print(f"Model ready: {model_name}")
                    return
                time.sleep(1)
            
            raise RuntimeError(f"Model failed to load within {max_wait}s")
    
    def proxy_request(self, path: str, method: str, data: dict) -> Response:
        """Forward request to vLLM server"""
        url = f"http://localhost:{self.port}{path}"
        
        if method == "POST":
            resp = requests.post(url, json=data, stream=True)
        else:
            resp = requests.get(url)
        
        # Stream response if it's SSE
        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            def generate():
                for chunk in resp.iter_content(chunk_size=None):
                    yield chunk
            return Response(generate(), content_type=resp.headers["content-type"])
        
        return jsonify(resp.json())

manager = VLLMModelManager()

@app.route("/v1/models", methods=["GET"])
def list_models():
    """Return currently loaded model"""
    models = []
    if manager.current_model:
        models.append({
            "id": manager.current_model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vllm-mlx"
        })
    return jsonify({"object": "list", "data": models})

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completion requests with auto-loading"""
    data = request.get_json()
    model = data.get("model", "default")
    
    # If model specified and different from current, load it
    if model != "default" and model != manager.current_model:
        try:
            manager.load_model(model)
        except Exception as e:
            return jsonify({"error": {"message": f"Failed to load model: {e}"}}), 500
    
    # Ensure we have a loaded model
    if not manager.current_model:
        return jsonify({"error": {"message": "No model loaded"}}), 400
    
    return manager.proxy_request("/v1/chat/completions", "POST", data)

@app.route("/api/v1/chat", methods=["POST"])
def native_chat():
    """Handle native API requests (LM Studio compatible)"""
    data = request.get_json()
    model = data.get("model", "default")
    
    if model != "default" and model != manager.current_model:
        try:
            manager.load_model(model)
        except Exception as e:
            return jsonify({"error": {"message": f"Failed to load model: {e}"}}), 500
    
    if not manager.current_model:
        return jsonify({"error": {"message": "No model loaded"}}), 400
    
    # vLLM-MLX uses OpenAI format, convert if needed
    # For now, proxy directly
    return manager.proxy_request("/v1/chat/completions", "POST", {
        "model": manager.current_model,
        "messages": [{"role": "user", "content": data.get("input", "")}]
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "current_model": manager.current_model,
        "model_loaded": manager.is_model_loaded()
    })

@app.route("/experiment", methods=["POST"])
def experiment():
    """
    Quick experiment endpoint - run a prompt and get stats
    POST /experiment
    {
      "model": "mlx-community/qwen3.5-35b-a3b",  // optional, uses current if loaded
      "prompt": "Your question here",
      "max_tokens": 150  // optional, default 100
    }
    Returns: response + timing stats
    """
    import time
    data = request.get_json()
    
    model = data.get("model", manager.current_model or "default")
    prompt = data.get("prompt", "Hello!")
    max_tokens = data.get("max_tokens", 100)
    
    # Load model if needed
    if model != manager.current_model:
        try:
            start_load = time.time()
            manager.load_model(model)
            load_time = time.time() - start_load
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {e}"}), 500
    else:
        load_time = 0
    
    # Run inference
    start_infer = time.time()
    try:
        response = manager.proxy_request("/v1/chat/completions", "POST", {
            "model": manager.current_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        })
        infer_time = time.time() - start_infer
        
        result = response.get_json()
        output_tokens = result.get("usage", {}).get("completion_tokens", 0)
        
        return jsonify({
            "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "stats": {
                "model": manager.current_model,
                "load_time_sec": round(load_time, 2),
                "inference_time_sec": round(infer_time, 2),
                "output_tokens": output_tokens,
                "tokens_per_sec": round(output_tokens / infer_time, 2) if infer_time > 0 else 0
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def signal_handler(sig, frame):
    """Clean shutdown"""
    print("\nShutting down...")
    manager.stop_current_model()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("vLLM-MLX Model Manager starting on port 1234")
    print("Backend vLLM-MLX on port 8001")
    print("Send requests to http://localhost:1234")
    
    app.run(host="0.0.0.0", port=1234, threaded=True)
