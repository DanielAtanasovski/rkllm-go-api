#!/bin/bash
# Example bash client for RKLLM Go API

API_BASE="http://localhost:8080"
MODEL_PATH="/models/Qwen3-4B-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm"

echo "=== RKLLM Go API Bash Client Example ==="
echo ""

# Health check
echo "1. Checking API health..."
curl -s "$API_BASE/health" | jq . || curl -s "$API_BASE/health"
echo ""

# Initialize model
echo "2. Initializing model..."
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
  }" | jq . || curl -s -X POST "$API_BASE/init" -H "Content-Type: application/json" -d "{\"model_path\": \"$MODEL_PATH\", \"max_context_len\": 1024, \"max_new_tokens\": 512}"
echo ""

# Give it a moment to fully initialize
sleep 1

# Send regular chat request (non-streaming)
echo "3. Sending regular chat request..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How many r'\''s in the word strawberry?", "role": "user", "stream": false}' | jq . || curl -s -X POST "$API_BASE/chat" -H "Content-Type: application/json" -d '{"prompt": "What is the capital of France?"}'
echo ""

# Send streaming chat request
echo "4. Sending streaming chat request..."
curl -s -X POST "$API_BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a short story about a robot.", "role": "user", "stream": true}' \
  --no-buffer
echo ""

# Destroy model
echo "5. Destroying model..."
curl -s -X POST "$API_BASE/destroy" | jq . || curl -s -X POST "$API_BASE/destroy"
echo ""

echo "=== Example completed ==="
