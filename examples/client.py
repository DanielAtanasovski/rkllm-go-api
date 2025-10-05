#!/usr/bin/env python3
"""Example Python client for RKLLM Go API"""

import requests
import json
import sys

API_BASE = "http://localhost:8080"
MODEL_PATH = "/home/armbian/deepseek-R1-distill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"

class RKLLMClient:
    def __init__(self, base_url=API_BASE):
        self.base_url = base_url
        self.session = requests.Session()

    def health(self):
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def init(self, model_path, **kwargs):
        """Initialize the LLM model"""
        data = {
            "model_path": model_path,
            "max_context_len": kwargs.get("max_context_len", 1024),
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "top_k": kwargs.get("top_k", 1),
            "top_p": kwargs.get("top_p", 0.95),
            "temperature": kwargs.get("temperature", 0.8),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            "skip_special_token": kwargs.get("skip_special_token", True),
            "embed_flash": kwargs.get("embed_flash", 1),
        }
        response = self.session.post(f"{self.base_url}/init", json=data)
        response.raise_for_status()
        return response.json()

    def chat(self, prompt, role="user", keep_history=0):
        """Send a chat request"""
        data = {"prompt": prompt, "role": role, "keep_history": keep_history}
        response = self.session.post(f"{self.base_url}/chat", json=data)
        response.raise_for_status()
        return response.json()

    def destroy(self):
        """Destroy the model"""
        response = self.session.post(f"{self.base_url}/destroy")
        response.raise_for_status()
        return response.json()

def main():
    print("=== RKLLM Go API Python Client Example ===\n")
    client = RKLLMClient()

    try:
        print("1. Checking health...")
        health = client.health()
        print(f"   Status: {health['status']}\n")

        print("2. Initializing model...")
        client.init(MODEL_PATH, max_context_len=1024, max_new_tokens=512)
        print("   Model initialized\n")

        print("3. Sending chat request...")
        response = client.chat("What is the capital of France?")
        print(f"   Response: {response['text']}\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            print("4. Destroying model...")
            client.destroy()
            print("   Done\n")
        except:
            pass

    print("=== Example completed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
