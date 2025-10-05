# Multi-stage Dockerfile for RKLLM Go API
# Stage 1: Builder - downloads RKLLM runtime and builds the Go application

FROM --platform=linux/arm64 golang:1.22-bookworm AS builder

# Install required build tools
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Download RKLLM runtime from GitHub
ARG RKLLM_VERSION=main
RUN echo "Downloading RKLLM runtime (${RKLLM_VERSION})..." && \
    wget -q https://github.com/Pelochus/ezrknn-llm/archive/refs/heads/${RKLLM_VERSION}.zip -O ezrknn-llm.zip && \
    unzip -q ezrknn-llm.zip && \
    mv ezrknn-llm-${RKLLM_VERSION}/rkllm-runtime ./rkllm-runtime && \
    rm -rf ezrknn-llm.zip ezrknn-llm-${RKLLM_VERSION}

# Copy Go module files and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY rkllm/ ./rkllm/
COPY main.go ./

# Set CGO flags to find RKLLM headers and libraries
ENV CGO_ENABLED=1
ENV GOOS=linux
ENV GOARCH=arm64
ENV CGO_CFLAGS="-I/build/rkllm-runtime/Linux/librkllm_api/include"
ENV CGO_LDFLAGS="-L/build/rkllm-runtime/Linux/librkllm_api/aarch64 -lrkllmrt"

# Build the application
RUN go build -o rkllm-api .

# Stage 2: Runtime
# Use minimal Debian base for smaller image size
FROM --platform=linux/arm64 debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built binary from builder stage
COPY --from=builder /build/rkllm-api /app/

# Copy RKLLM runtime library
COPY --from=builder /build/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /usr/local/lib/

# Update library cache
RUN ldconfig

# Create directory for models (will be mounted from host)
RUN mkdir -p /models

# Expose API port
EXPOSE 8080

# Set environment variables
ENV GIN_MODE=release
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the API server
CMD ["/app/rkllm-api"]
