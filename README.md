# vLLM-MLX Model Manager

Wrapper service that provides LM Studio-like dynamic model loading on top of vLLM-MLX.

## Problem Solved

vLLM-MLX loads one model at startup. This manager:
- Loads models on-demand when requested via API
- Unloads old models when switching
- Provides same API interface as LM Studio (OpenAI-compatible + native)
- Supports benchmarking multiple models without manual restarts

## Installation

```bash
# Install dependencies
pip install flask requests

# Ensure vLLM-MLX is installed
uv tool install git+https://github.com/waybarrios/vllm-mlx.git
```

## Configuration

By default, the manager uses **LM Studio's model cache** (`~/.lmstudio/models/`) to avoid re-downloading models.

To use a different cache directory:

```python
# Edit vllm-model-manager.py
manager = VLLMModelManager(port=8001, cache_dir="~/.cache/huggingface/hub")
```

## Usage

```bash
# Start the manager (listens on port 1234, proxies to vLLM-MLX on 8001)
python experiments/vllm-model-manager.py

# Request will auto-load the model
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen3.5-35b-a3b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Switch to different model (auto-unloads previous, loads new)
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen3.5-122b-a10b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Benchmarking

Your existing benchmark scripts will work with this manager:

```bash
# Point benchmark to the manager instead of LM Studio
LM_STUDIO_URL="http://localhost:1234" ./experiments/benchmark-context-sizes.sh
```

The manager will automatically:
1. Load each model when benchmark requests it
2. Unload previous model to free memory
3. Wait for new model to be ready
4. Proxy requests to vLLM-MLX backend

## API Endpoints

- `GET /v1/models` - List currently loaded model
- `POST /v1/chat/completions` - OpenAI-compatible chat (auto-loads model from request)
- `POST /api/v1/chat` - Native LM Studio format (for compatibility)
- `GET /health` - Health check + current model status

## Architecture

```
Benchmark/Client (port 1234)
    ↓
Model Manager (Flask)
    ↓ (spawns/kills vllm-mlx processes)
vLLM-MLX Server (port 8001)
    ↓
MLX Framework (GPU)
```

## Limitations

- ~10-30s model loading time on first request (depends on model size)
- Only one model loaded at a time (like LM Studio)
- No persistent KV cache across model switches
- Streaming responses supported but not fully tested

## Future Enhancements

- Pre-load models in background
- Keep multiple models loaded (memory permitting)
- Model metadata cache
- Better error handling for model download/loading failures
