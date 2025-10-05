# RKLLM Go API Wrapper

Golang REST API wrapper for Rockchip's RKLLM library. This allows you to run LLMs on Rockchip NPUs (RK3588/RK3588S) and query them from any application via HTTP.

## Prerequisites

- Rockchip RK3588/RK3588S SoC
- NPU driver (Tested with 0.9.8)

## Installation

### Docker

```bash
# Quick start with Docker
./docker-build.sh
docker-compose up -d

# Or with specific RKLLM version
./docker-build.sh --version v1.2.1
docker-compose up -d
```

### From Source

Ensure you setup the RKNN and RKLLM libraries as per the [ezrknpu](https://github.com/rockchip-linux/ezrknpu) instructions.

```bash
./build.sh
```

## Quick Start

```bash
# Start the server
./rkllm-api

# In another terminal, initialize a model
curl -X POST http://localhost:8080/init \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/model.rkllm",
    "max_context_len": 1024,
    "max_new_tokens": 512
  }'

# Send a chat request
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

## API Endpoints

### `GET /health`
Check server and model status.

### `POST /init`
Initialize an RKLLM model.

**Request Body:**
```json
{
  "model_path": "/path/to/model.rkllm",
  "max_context_len": 1024,
  "max_new_tokens": 512,
  "temperature": 0.8,
  "top_k": 1,
  "top_p": 0.95,
  "embed_flash": 1
}
```

### `POST /chat`
Send a prompt and get a completion.

**Request Body:**
```json
{
  "prompt": "What is AI?",
  "role": "user",
  "keep_history": 0
}
```

### `POST /destroy`
Unload the model and free resources.

### `POST /abort`
Stop a currently running inference.

## License

Same license as the parent ezrknpu repository.
