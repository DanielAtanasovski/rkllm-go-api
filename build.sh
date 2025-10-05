#!/bin/bash
# Build script for RKLLM Go API

set -e

echo "=== Building RKLLM Go API ==="
echo ""

# Check if we're in the right directory
if [ ! -f "main.go" ]; then
    echo "Error: main.go not found. Please run this script from the rkllm-go-api directory."
    exit 1
fi

# Check for Go installation
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed. Please install Go 1.21 or higher."
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
echo "Found Go version: $GO_VERSION"
echo ""

# Download dependencies
echo "Downloading dependencies..."
go get github.com/gin-gonic/gin@v1.9.1
go mod tidy
echo ""

# Build the binary
echo "Building rkllm-api..."
go build -o rkllm-api main.go

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Binary created: ./rkllm-api"
    echo ""
    echo "To run the server:"
    echo "  export LD_LIBRARY_PATH=/home/armbian/ezrknpu/ezrknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:\$LD_LIBRARY_PATH"
    echo "  ./rkllm-api"
else
    echo ""
    echo "✗ Build failed"
    exit 1
fi
