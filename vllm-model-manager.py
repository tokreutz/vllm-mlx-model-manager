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
    def __init__(self, port=8001, cache_dir="~/.cache/lm-studio/models"):
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
            
            # Set HF_HOME to LM Studio cache directory
            env = os.environ.copy()
            env['HF_HOME'] = self.cache_dir
            
            cmd = [
                "vllm-mlx", "serve", model_name,
                "--port", str(self.port)
            ]
            if continuous_batching:
                cmd.append("--continuous-batching")
            
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
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
