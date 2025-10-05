#!/bin/bash
# Enhanced RKLLM Go API Bash Client - Tests all features including MCP and system prompts

API_BASE="http://localhost:8080"
MODEL_PATH="/models/Qwen3-4B-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm"

echo "=== Enhanced RKLLM Go API Bash Client ==="
echo ""

# Health check
echo "1. Checking API health..."
curl -s "$API_BASE/health" | jq . || curl -s "$API_BASE/health"
echo ""

# Check if model is auto-initialized (might be set via AUTO_INIT_MODEL env var)
echo "2. Checking initialization status..."
health=$(curl -s "$API_BASE/health")
if echo "$health" | grep -q '"initialized":true'; then
    echo "   ✓ Model is already initialized!"
else
    echo "   → Initializing model manually..."
    curl -s -X POST "$API_BASE/init" \
      -H "Content-Type: application/json" \
      -d "{
        \"model_path\": \"$MODEL_PATH\",
        \"max_context_len\": 1024,
        \"max_new_tokens\": 512,
        \"top_k\": 1,
        \"top_p\": 0.95,
        \"temperature\": 0.8,
        \"repeat_penalty\": 1.1,
        \"skip_special_token\": true,
        \"embed_flash\": 1
      }" | jq . || curl -s -X POST "$API_BASE/init" -H "Content-Type: application/json" -d "{\"model_path\": \"$MODEL_PATH\"}"
    echo ""
    sleep 1
fi
echo ""

# Check system prompt
echo "3. Checking system prompt..."
curl -s "$API_BASE/system-prompt" | jq . || curl -s "$API_BASE/system-prompt"
echo ""

# List MCP servers
echo "4. Checking MCP servers..."
curl -s "$API_BASE/mcp/servers" | jq . || curl -s "$API_BASE/mcp/servers"
echo ""

# List available MCP tools
echo "5. Listing available MCP tools..."
curl -s "$API_BASE/mcp/tools" | jq . || curl -s "$API_BASE/mcp/tools"
echo ""

# Basic chat request (system prompt and MCP tools always enabled)
echo "6. Basic chat request..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "stream": false}' | jq . || curl -s -X POST "$API_BASE/chat" -H "Content-Type: application/json" -d '{"prompt": "What is 2+2?"}'
echo ""

# Chat to test system prompt integration
echo "7. Chat to verify system prompt integration..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe your capabilities and role", "stream": false}' | jq . || curl -s -X POST "$API_BASE/chat" -H "Content-Type: application/json" -d '{"prompt": "Describe your capabilities"}'
echo ""

# Chat to test MCP tools awareness
echo "8. Chat to verify MCP tools awareness..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What tools do you have available to help me?", "stream": false}' | jq . || curl -s -X POST "$API_BASE/chat" -H "Content-Type: application/json" -d '{"prompt": "What tools are available?"}'
echo ""

# Streaming chat (all features always enabled)
echo "9. Streaming chat with integrated features..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me about yourself and show me how you can use tools to help users", "stream": true}' \
  --no-buffer
echo ""

# Test direct MCP tool call (if available)
echo "10. Testing direct MCP tool call..."
echo "    → Attempting to list /tmp directory via filesystem MCP server..."
curl -s -X POST "$API_BASE/mcp/call" \
  -H "Content-Type: application/json" \
  -d '{"server": "filesystem", "tool": "list_directory", "params": {"path": "/tmp"}}' | jq . || echo "    ⚠ MCP filesystem server not available"
echo ""

# Update system prompt dynamically
echo "11. Updating system prompt..."
curl -s -X POST "$API_BASE/system-prompt" \
  -H "Content-Type: application/json" \
  -d '{"system_prompt": "You are a helpful AI assistant specialized in system administration and file operations. You have access to filesystem tools and should use them when users ask about files or directories."}' | jq . || curl -s -X POST "$API_BASE/system-prompt" -H "Content-Type: application/json" -d '{"system_prompt": "Updated specialized prompt"}'
echo ""

# Test with updated system prompt
echo "12. Chat with updated system prompt..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are you specialized in?", "stream": false}' | jq . || curl -s -X POST "$API_BASE/chat" -H "Content-Type: application/json" -d '{"prompt": "What are you specialized in?"}'
echo ""

# Comprehensive streaming test
echo "13. Final comprehensive streaming test..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How many r'\''s are in the word strawberry? Also, if you have file system tools available, create a test file with this information.", "stream": true}' \
  --no-buffer
echo ""

# Cleanup - destroy model
echo "14. Destroying model..."
curl -s -X POST "$API_BASE/destroy" | jq . || curl -s -X POST "$API_BASE/destroy"
echo ""

echo "=== Example completed ==="
echo ""
echo "Features tested:"
echo "  ✓ Health check and auto-initialization detection"
echo "  ✓ Manual model initialization"
echo "  ✓ System prompt retrieval and updates"
echo "  ✓ MCP server and tool discovery"
echo "  ✓ Basic chat (system prompt & MCP always enabled)"
echo "  ✓ System prompt integration verification"
echo "  ✓ MCP tools awareness verification"
echo "  ✓ Streaming with integrated features"
echo "  ✓ Direct MCP tool calls"
echo "  ✓ Dynamic system prompt updates"
echo "  ✓ Comprehensive integration test"
