package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/Pelochus/rkllm-go-api/rkllm"
	"github.com/gin-gonic/gin"
)

// Global LLM handle and state
var (
	llmHandle     *rkllm.Handle
	llmMutex      sync.Mutex
	isInitialized bool
	currentResult string
	resultMutex   sync.Mutex
	// Streaming state
	streamChannel chan *ChatResponse
	streamMutex   sync.Mutex
	isStreaming   bool
)

// InitRequest represents the request body for initializing the LLM
type InitRequest struct {
	ModelPath         string  `json:"model_path" binding:"required"`
	MaxContextLen     int32   `json:"max_context_len"`
	MaxNewTokens      int32   `json:"max_new_tokens"`
	TopK              int32   `json:"top_k"`
	TopP              float32 `json:"top_p"`
	Temperature       float32 `json:"temperature"`
	RepeatPenalty     float32 `json:"repeat_penalty"`
	FrequencyPenalty  float32 `json:"frequency_penalty"`
	PresencePenalty   float32 `json:"presence_penalty"`
	SkipSpecialToken  bool    `json:"skip_special_token"`
	BaseDomainID      int32   `json:"base_domain_id"`
	EmbedFlash        int8    `json:"embed_flash"`
}

// ChatRequest represents a chat/completion request
type ChatRequest struct {
	Prompt         string `json:"prompt" binding:"required"`
	Role           string `json:"role"`
	EnableThinking bool   `json:"enable_thinking"`
	KeepHistory    int    `json:"keep_history"`
	Stream         bool   `json:"stream"`
}

// ChatResponse represents the response from a chat request
type ChatResponse struct {
	Text           string             `json:"text"`
	TokenID        int32              `json:"token_id,omitempty"`
	PerfStats      *rkllm.PerfStat    `json:"perf_stats,omitempty"`
	Error          string             `json:"error,omitempty"`
	Delta          string             `json:"delta,omitempty"`       // For streaming: incremental text
	Finished       bool               `json:"finished,omitempty"`    // For streaming: indicates completion
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status      string `json:"status"`
	Initialized bool   `json:"initialized"`
	Running     bool   `json:"running"`
}

// Result callback that accumulates text and handles streaming
func resultCallback(result *rkllm.Result, state rkllm.CallState) int {
	resultMutex.Lock()
	defer resultMutex.Unlock()

	switch state {
	case rkllm.CallStateNormal:
		currentResult += result.Text
		fmt.Print(result.Text) // Also print to console

		// Send to stream if streaming is enabled
		streamMutex.Lock()
		if isStreaming && streamChannel != nil {
			select {
			case streamChannel <- &ChatResponse{
				Text:     currentResult, // Full accumulated text so far
				Delta:    result.Text,   // Incremental token
				TokenID:  result.TokenID,
				Finished: false,
			}:
			default:
				// Channel is full or closed, continue
			}
		}
		streamMutex.Unlock()

	case rkllm.CallStateFinish:
		fmt.Printf("\n[Perf] Prefill: %.2fms (%d tokens), Generate: %.2fms (%d tokens, %.2f tokens/s), Memory: %.2fMB\n",
			result.Perf.PrefillTimeMs,
			result.Perf.PrefillTokens,
			result.Perf.GenerateTimeMs,
			result.Perf.GenerateTokens,
			float32(result.Perf.GenerateTokens)*1000.0/result.Perf.GenerateTimeMs,
			result.Perf.MemoryUsageMB,
		)

		// Send final response to stream if streaming is enabled
		streamMutex.Lock()
		if isStreaming && streamChannel != nil {
			select {
			case streamChannel <- &ChatResponse{
				Text:      currentResult, // Complete final text
				Finished:  true,
				PerfStats: &result.Perf,
			}:
			default:
				// Channel is full or closed, continue
			}
			close(streamChannel)
			streamChannel = nil
			isStreaming = false
		}
		streamMutex.Unlock()

	case rkllm.CallStateError:
		log.Println("Error in LLM callback")

		// Send error to stream if streaming is enabled
		streamMutex.Lock()
		if isStreaming && streamChannel != nil {
			select {
			case streamChannel <- &ChatResponse{
				Error:    "LLM callback error",
				Finished: true,
			}:
			default:
				// Channel is full or closed, continue
			}
			close(streamChannel)
			streamChannel = nil
			isStreaming = false
		}
		streamMutex.Unlock()
	}

	return 0
}

// Initialize the LLM
func initHandler(c *gin.Context) {
	llmMutex.Lock()
	defer llmMutex.Unlock()

	if isInitialized {
		c.JSON(http.StatusBadRequest, gin.H{"error": "LLM already initialized. Destroy first."})
		return
	}

	var req InitRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Create parameters
	param := rkllm.CreateDefaultParam()
	param.ModelPath = req.ModelPath

	// Override with request values if provided
	if req.MaxContextLen > 0 {
		param.MaxContextLen = req.MaxContextLen
	}
	if req.MaxNewTokens > 0 {
		param.MaxNewTokens = req.MaxNewTokens
	}
	if req.TopK > 0 {
		param.TopK = req.TopK
	}
	if req.TopP > 0 {
		param.TopP = req.TopP
	}
	if req.Temperature > 0 {
		param.Temperature = req.Temperature
	}
	if req.RepeatPenalty > 0 {
		param.RepeatPenalty = req.RepeatPenalty
	}

	param.FrequencyPenalty = req.FrequencyPenalty
	param.PresencePenalty = req.PresencePenalty
	param.SkipSpecialToken = req.SkipSpecialToken
	param.ExtendParam.BaseDomainID = req.BaseDomainID
	param.ExtendParam.EmbedFlash = req.EmbedFlash

	// Initialize the LLM
	log.Println("Initializing RKLLM...")
	handle, err := rkllm.Init(param, resultCallback)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to initialize LLM: %v", err)})
		return
	}

	llmHandle = handle
	isInitialized = true
	log.Println("RKLLM initialized successfully")

	c.JSON(http.StatusOK, gin.H{
		"status": "initialized",
		"model_path": req.ModelPath,
		"max_context_len": param.MaxContextLen,
		"max_new_tokens": param.MaxNewTokens,
	})
}

// Chat/completion endpoint with streaming support
func chatHandler(c *gin.Context) {
	llmMutex.Lock()
	if !isInitialized {
		llmMutex.Unlock()
		c.JSON(http.StatusBadRequest, gin.H{"error": "LLM not initialized. Call /init first."})
		return
	}

	if llmHandle.IsRunning() {
		llmMutex.Unlock()
		c.JSON(http.StatusTooManyRequests, gin.H{"error": "LLM is currently processing another request"})
		return
	}
	llmMutex.Unlock()

	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Reset current result
	resultMutex.Lock()
	currentResult = ""
	resultMutex.Unlock()

	// Prepare input
	input := rkllm.Input{
		Role:           req.Role,
		EnableThinking: req.EnableThinking,
		InputType:      rkllm.InputTypePrompt,
		PromptInput:    req.Prompt,
	}

	inferParam := rkllm.InferParam{
		Mode:        rkllm.InferModeGenerate,
		KeepHistory: req.KeepHistory,
	}

	log.Printf("Running inference for prompt: %s (stream=%v)\n", req.Prompt, req.Stream)
	startTime := time.Now()

	if req.Stream {
		// Handle streaming response
		handleStreamingChat(c, input, inferParam)
	} else {
		// Handle non-streaming response (existing behavior)
		err := llmHandle.Run(input, inferParam)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Inference failed: %v", err)})
			return
		}

		elapsed := time.Since(startTime)
		log.Printf("Inference completed in %v\n", elapsed)

		// Get the accumulated result
		resultMutex.Lock()
		response := ChatResponse{
			Text: currentResult,
		}
		resultMutex.Unlock()

		c.JSON(http.StatusOK, response)
	}
}

// handleStreamingChat handles Server-Sent Events streaming
func handleStreamingChat(c *gin.Context, input rkllm.Input, inferParam rkllm.InferParam) {
	// Set up streaming
	streamMutex.Lock()
	streamChannel = make(chan *ChatResponse, 10) // Buffered channel
	isStreaming = true
	streamMutex.Unlock()

	// Set headers for Server-Sent Events
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	// Start inference in a goroutine
	go func() {
		err := llmHandle.Run(input, inferParam)
		if err != nil {
			log.Printf("Streaming inference failed: %v\n", err)
			// Send error through stream
			streamMutex.Lock()
			if streamChannel != nil {
				select {
				case streamChannel <- &ChatResponse{
					Error:    fmt.Sprintf("Inference failed: %v", err),
					Finished: true,
				}:
				default:
				}
				close(streamChannel)
				streamChannel = nil
				isStreaming = false
			}
			streamMutex.Unlock()
		}
	}()

	// Stream responses
	for response := range streamChannel {
		if response.Error != "" {
			// Send error event
			c.SSEvent("error", response)
			break
		} else if response.Finished {
			// Send completion event with performance stats
			c.SSEvent("done", response)
			break
		} else {
			// Send delta event
			c.SSEvent("delta", response)
		}
		c.Writer.Flush()
	}
}

// Destroy the LLM
func destroyHandler(c *gin.Context) {
	llmMutex.Lock()
	defer llmMutex.Unlock()

	if !isInitialized {
		c.JSON(http.StatusBadRequest, gin.H{"error": "LLM not initialized"})
		return
	}

	log.Println("Destroying RKLLM...")
	err := llmHandle.Destroy()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to destroy LLM: %v", err)})
		return
	}

	llmHandle = nil
	isInitialized = false
	log.Println("RKLLM destroyed successfully")

	c.JSON(http.StatusOK, gin.H{"status": "destroyed"})
}

// Health check endpoint
func healthHandler(c *gin.Context) {
	llmMutex.Lock()
	running := false
	if isInitialized && llmHandle != nil {
		running = llmHandle.IsRunning()
	}
	llmMutex.Unlock()

	c.JSON(http.StatusOK, HealthResponse{
		Status:      "ok",
		Initialized: isInitialized,
		Running:     running,
	})
}

// Abort current inference
func abortHandler(c *gin.Context) {
	llmMutex.Lock()
	defer llmMutex.Unlock()

	if !isInitialized {
		c.JSON(http.StatusBadRequest, gin.H{"error": "LLM not initialized"})
		return
	}

	err := llmHandle.Abort()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to abort: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "aborted"})
}

func main() {
	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("\nReceived interrupt signal, shutting down...")

		llmMutex.Lock()
		if isInitialized && llmHandle != nil {
			log.Println("Destroying LLM handle...")
			llmHandle.Destroy()
		}
		llmMutex.Unlock()

		os.Exit(0)
	}()

	// Setup Gin router
	router := gin.Default()

	// API endpoints
	router.POST("/init", initHandler)
	router.POST("/chat", chatHandler)
	router.POST("/destroy", destroyHandler)
	router.POST("/abort", abortHandler)
	router.GET("/health", healthHandler)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting RKLLM API server on port %s...\n", port)
	log.Printf("API endpoints:\n")
	log.Printf("  POST /init     - Initialize the LLM model\n")
	log.Printf("  POST /chat     - Send a chat/completion request\n")
	log.Printf("  POST /destroy  - Destroy the LLM model\n")
	log.Printf("  POST /abort    - Abort current inference\n")
	log.Printf("  GET  /health   - Health check\n")

	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
