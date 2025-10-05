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

### Option 1: Auto-Initialize with Environment Variables
```bash
# Set environment variables for auto-initialization
export AUTO_INIT_MODEL="/models/your-model.rkllm"
export SYSTEM_PROMPT_PATH="./system_prompt.txt"
export MCP_SERVERS_PATH="./mcp_servers.json"

# Start the server (model loads automatically)
./rkllm-api
```

### Option 2: Manual Initialization
```bash
# Start the server
./rkllm-api

# In another terminal, initialize a model
curl -X POST http://localhost:8080/init \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/your-model.rkllm",
    "max_context_len": 1024,
    "max_new_tokens": 512
  }'
```

### Chat Examples
```bash
# Basic chat request (system prompt and MCP tools always enabled)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "stream": false}'

# Streaming chat
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Search for recent AI developments",
    "stream": true
  }' --no-buffer
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
Send a prompt and get a completion. Supports streaming, system prompts, and MCP tool integration.

**Request Body:**
```json
{
  "prompt": "What is AI?",
  "role": "user",
  "keep_history": 0,
  "stream": false
}
```

**Parameters:**
- `prompt`: The user's input text (required)
- `role`: Role identifier for multi-turn conversations (optional)
- `keep_history`: Number of previous messages to keep in context (optional)
- `stream`: Enable real-time token streaming (optional, default: false)

**Note:** System prompt and MCP tools are always enabled for all requests. Configure them via `system_prompt.txt` and `mcp_servers.json` files.

**Streaming Mode:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "stream": true}' \
  --no-buffer
```

### MCP Integration Endpoints

#### `GET /mcp/servers`
List configured MCP servers.

#### `GET /mcp/tools`
List all available tools from MCP servers.

#### `POST /mcp/call`
Call a specific MCP tool directly.
```json
{
  "server": "filesystem",
  "tool": "read_file",
  "params": {"path": "/tmp/example.txt"}
}
```

### System Prompt Endpoints

#### `GET /system-prompt`
Get the current system prompt.

#### `POST /system-prompt`
Update the system prompt.
```json
{
  "system_prompt": "You are a helpful coding assistant..."
}
```

### `POST /destroy`
Unload the model and free resources.

### `POST /abort`
Stop a currently running inference.

## Configuration

### Environment Variables

- `AUTO_INIT_MODEL`: Path to model file for auto-initialization on startup
- `MAX_CONTEXT_LEN`: Maximum context length (default from model)
- `MAX_NEW_TOKENS`: Maximum new tokens to generate (default from model)
- `TEMPERATURE`: Sampling temperature (default from model)
- `SYSTEM_PROMPT_PATH`: Path to system prompt file (default: `/app/system_prompt.txt`)
- `MCP_SERVERS_PATH`: Path to MCP servers config (default: `/app/mcp_servers.json`)

### Configuration Files

#### System Prompt (`system_prompt.txt`)
Defines the AI's role and capabilities. Can be mounted as Docker volume for easy customization.

#### MCP Servers (`mcp_servers.json`)
Lists MCP servers for tool integration:
```json
[
  {
    "name": "filesystem",
    "url": "http://mcp-server:8000/mcp",
    "headers": {"Authorization": "Bearer token"}
  }
]
```

## Examples

### Comprehensive Test Client
The enhanced bash client tests all features:
```bash
cd examples && ./client.sh
```

This script demonstrates:
- Auto-initialization detection
- System prompt management
- MCP server and tool discovery
- Basic, enhanced, and streaming chat modes
- Direct MCP tool calls
- Dynamic configuration updates

### Manual API Usage
```bash
# List available MCP tools
curl http://localhost:8080/mcp/tools

# Call a MCP tool directly
curl -X POST http://localhost:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"server": "filesystem", "tool": "read_file", "params": {"path": "/tmp/test.txt"}}'

# Chat with streaming (system prompt and tools automatically enabled)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "List files in /tmp directory", "stream": true}'

# Update system prompt
curl -X POST http://localhost:8080/system-prompt \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a specialized assistant..."}'
```

## License

Same license as the parent ezrknpu repository.
