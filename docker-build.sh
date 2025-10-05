#!/bin/bash
set -e

echo "=== Building RKLLM Go API Docker Image ==="
echo ""

# Check if we're on the right architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "Warning: Building on $ARCH, but target is arm64/aarch64"
    echo "Consider building directly on the Orange Pi for best results"
fi

# Parse command line arguments
RKLLM_VERSION="main"
CACHE_OPTION="--no-cache"
BUILD_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            RKLLM_VERSION="$2"
            shift 2
            ;;
        --cache)
            CACHE_OPTION=""
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version VERSION    Specify RKLLM version/branch (default: main)"
            echo "  --cache             Use Docker build cache (default: no cache)"
            echo "  --build-arg ARG     Pass additional build arguments to Docker"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build with defaults"
            echo "  $0 --version v1.2.1                  # Build with specific version"
            echo "  $0 --cache                           # Build with cache enabled"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the image
echo "Building Docker image with RKLLM version: $RKLLM_VERSION"
echo "This will download RKLLM runtime from GitHub during build..."
echo ""

docker build \
    $CACHE_OPTION \
    --build-arg RKLLM_VERSION=$RKLLM_VERSION \
    $BUILD_ARGS \
    -t rkllm-go-api:latest \
    -t rkllm-go-api:$RKLLM_VERSION \
    .

echo ""
echo "✓ Build successful!"
echo ""
echo "Image tags created:"
echo "  - rkllm-go-api:latest"
echo "  - rkllm-go-api:$RKLLM_VERSION"
echo ""
echo "To run the container:"
echo "  docker-compose up -d"
echo ""
echo "Or manually:"
echo "  docker run -d \\"
echo "    --privileged \\"
echo "    -p 8080:8080 \\"
echo "    -v /home/armbian/llm/models:/models:ro \\"
echo "    --name rkllm-api \\"
echo "    rkllm-go-api:latest"
echo ""
