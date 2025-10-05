package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Pelochus/rkllm-go-api/rkllm"
	"github.com/gin-gonic/gin"
)

// MCP Server configuration
type MCPServer struct {
	URL     string            `json:"url"`
	Name    string            `json:"name"`
	Tools   []MCPTool         `json:"tools"`
	Headers map[string]string `json:"headers,omitempty"`
}

type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type MCPRequest struct {
	Jsonrpc string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params"`
}

type MCPResponse struct {
	Jsonrpc string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   interface{} `json:"error,omitempty"`
}

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
	// MCP and system prompt
	mcpServers    []MCPServer
	mcpMutex      sync.RWMutex
	systemPrompt  string
	promptMutex   sync.RWMutex
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

// Load system prompt from file
func loadSystemPrompt() {
	promptPath := os.Getenv("SYSTEM_PROMPT_PATH")
	if promptPath == "" {
		promptPath = "/app/system_prompt.txt"
	}

	if content, err := os.ReadFile(promptPath); err == nil {
		promptMutex.Lock()
		systemPrompt = strings.TrimSpace(string(content))
		promptMutex.Unlock()
		log.Printf("Loaded system prompt from %s (%d characters)", promptPath, len(systemPrompt))
	} else {
		log.Printf("No system prompt file found at %s, using default", promptPath)
		promptMutex.Lock()
		systemPrompt = "You are a helpful AI assistant."
		promptMutex.Unlock()
	}
}

// Load MCP servers configuration
func loadMCPServers() {
	mcpPath := os.Getenv("MCP_SERVERS_PATH")
	if mcpPath == "" {
		mcpPath = "/app/mcp_servers.json"
	}

	if content, err := os.ReadFile(mcpPath); err == nil {
		var servers []MCPServer
		if err := json.Unmarshal(content, &servers); err == nil {
			mcpMutex.Lock()
			mcpServers = servers
			mcpMutex.Unlock()
			log.Printf("Loaded %d MCP servers from %s", len(servers), mcpPath)

			// Discover tools from each server
			for i := range mcpServers {
				discoverMCPTools(&mcpServers[i])
			}
		} else {
			log.Printf("Failed to parse MCP servers config: %v", err)
		}
	} else {
		log.Printf("No MCP servers config found at %s", mcpPath)
	}
}

// Discover tools from an MCP server
func discoverMCPTools(server *MCPServer) {
	req := MCPRequest{
		Jsonrpc: "2.0",
		ID:      1,
		Method:  "tools/list",
		Params:  map[string]interface{}{},
	}

	reqBody, _ := json.Marshal(req)
	resp, err := http.Post(server.URL, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("Failed to discover tools from %s: %v", server.Name, err)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var mcpResp MCPResponse
	if err := json.Unmarshal(body, &mcpResp); err == nil {
		if toolsData, ok := mcpResp.Result.(map[string]interface{}); ok {
			if tools, ok := toolsData["tools"].([]interface{}); ok {
				server.Tools = []MCPTool{}
				for _, tool := range tools {
					if toolMap, ok := tool.(map[string]interface{}); ok {
						mcpTool := MCPTool{
							Name:        getString(toolMap, "name"),
							Description: getString(toolMap, "description"),
							InputSchema: getMap(toolMap, "inputSchema"),
						}
						server.Tools = append(server.Tools, mcpTool)
					}
				}
				log.Printf("Discovered %d tools from %s", len(server.Tools), server.Name)
			}
		}
	}
}

// Helper functions for type assertions
func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getMap(m map[string]interface{}, key string) map[string]interface{} {
	if val, ok := m[key].(map[string]interface{}); ok {
		return val
	}
	return nil
}

// Call an MCP tool
func callMCPTool(serverName, toolName string, params map[string]interface{}) (interface{}, error) {
	mcpMutex.RLock()
	var server *MCPServer
	for i := range mcpServers {
		if mcpServers[i].Name == serverName {
			server = &mcpServers[i]
			break
		}
	}
	mcpMutex.RUnlock()

	if server == nil {
		return nil, fmt.Errorf("MCP server %s not found", serverName)
	}

	req := MCPRequest{
		Jsonrpc: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name":      toolName,
			"arguments": params,
		},
	}

	reqBody, _ := json.Marshal(req)
	httpReq, _ := http.NewRequest("POST", server.URL, bytes.NewBuffer(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")

	// Add custom headers
	for k, v := range server.Headers {
		httpReq.Header.Set(k, v)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var mcpResp MCPResponse
	if err := json.Unmarshal(body, &mcpResp); err != nil {
		return nil, err
	}

	if mcpResp.Error != nil {
		return nil, fmt.Errorf("MCP error: %v", mcpResp.Error)
	}

	return mcpResp.Result, nil
}

// Build enhanced prompt with system prompt and available tools (always enabled)
func buildEnhancedPrompt(originalPrompt string) string {
	var parts []string

	// Always include system prompt
	promptMutex.RLock()
	if systemPrompt != "" {
		parts = append(parts, "System: "+systemPrompt)
	}
	promptMutex.RUnlock()

	// Always include MCP tools if available
	mcpMutex.RLock()
	if len(mcpServers) > 0 {
		parts = append(parts, "\nAvailable Tools:")
		for _, server := range mcpServers {
			for _, tool := range server.Tools {
				toolDesc := fmt.Sprintf("- %s (%s): %s", tool.Name, server.Name, tool.Description)
				parts = append(parts, toolDesc)
			}
		}
		parts = append(parts, "\nTo use a tool, respond with JSON: {\"tool\": \"toolName\", \"server\": \"serverName\", \"params\": {...}}")
	}
	mcpMutex.RUnlock()

	parts = append(parts, "\nUser: "+originalPrompt)

	return strings.Join(parts, "\n")
}

// Auto-initialize model on startup if environment variable is set
func autoInitializeModel() {
	modelPath := os.Getenv("AUTO_INIT_MODEL")
	if modelPath == "" {
		return
	}

	log.Printf("Auto-initializing model: %s", modelPath)

	// Create default parameters
	param := rkllm.CreateDefaultParam()
	param.ModelPath = modelPath

	// Override with environment variables if provided
	if maxCtx := os.Getenv("MAX_CONTEXT_LEN"); maxCtx != "" {
		if val := parseInt32(maxCtx); val > 0 {
			param.MaxContextLen = val
		}
	}
	if maxTokens := os.Getenv("MAX_NEW_TOKENS"); maxTokens != "" {
		if val := parseInt32(maxTokens); val > 0 {
			param.MaxNewTokens = val
		}
	}
	if temp := os.Getenv("TEMPERATURE"); temp != "" {
		if val := parseFloat32(temp); val > 0 {
			param.Temperature = val
		}
	}

	// Initialize the LLM
	handle, err := rkllm.Init(param, resultCallback)
	if err != nil {
		log.Printf("Failed to auto-initialize model: %v", err)
		return
	}

	llmMutex.Lock()
	llmHandle = handle
	isInitialized = true
	llmMutex.Unlock()

	log.Println("Model auto-initialized successfully")
}

// Helper parsing functions
func parseInt32(s string) int32 {
	var val int32
	fmt.Sscanf(s, "%d", &val)
	return val
}

func parseFloat32(s string) float32 {
	var val float32
	fmt.Sscanf(s, "%f", &val)
	return val
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

	// Build enhanced prompt with system prompt and MCP tools (always enabled)
	enhancedPrompt := buildEnhancedPrompt(req.Prompt)

	// Prepare input
	input := rkllm.Input{
		Role:           req.Role,
		EnableThinking: req.EnableThinking,
		InputType:      rkllm.InputTypePrompt,
		PromptInput:    enhancedPrompt,
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

// MCP Servers endpoint
func mcpServersHandler(c *gin.Context) {
	mcpMutex.RLock()
	servers := make([]map[string]interface{}, len(mcpServers))
	for i, server := range mcpServers {
		servers[i] = map[string]interface{}{
			"name": server.Name,
			"url":  server.URL,
			"tool_count": len(server.Tools),
		}
	}
	mcpMutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{"servers": servers})
}

// MCP Tools endpoint
func mcpToolsHandler(c *gin.Context) {
	mcpMutex.RLock()
	allTools := []map[string]interface{}{}
	for _, server := range mcpServers {
		for _, tool := range server.Tools {
			allTools = append(allTools, map[string]interface{}{
				"name":        tool.Name,
				"description": tool.Description,
				"server":      server.Name,
				"inputSchema": tool.InputSchema,
			})
		}
	}
	mcpMutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{"tools": allTools})
}

// MCP Call endpoint
func mcpCallHandler(c *gin.Context) {
	var req struct {
		Server string                 `json:"server" binding:"required"`
		Tool   string                 `json:"tool" binding:"required"`
		Params map[string]interface{} `json:"params"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := callMCPTool(req.Server, req.Tool, req.Params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"result": result})
}

// Get system prompt endpoint
func getSystemPromptHandler(c *gin.Context) {
	promptMutex.RLock()
	prompt := systemPrompt
	promptMutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{"system_prompt": prompt})
}

// Set system prompt endpoint
func setSystemPromptHandler(c *gin.Context) {
	var req struct {
		SystemPrompt string `json:"system_prompt" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	promptMutex.Lock()
	systemPrompt = req.SystemPrompt
	promptMutex.Unlock()

	c.JSON(http.StatusOK, gin.H{"status": "system prompt updated"})
}

func main() {
	// Load system prompt and MCP servers on startup
	loadSystemPrompt()
	loadMCPServers()

	// Auto-initialize model if environment variable is set
	autoInitializeModel()

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

	// MCP and system prompt endpoints
	router.GET("/mcp/servers", mcpServersHandler)
	router.GET("/mcp/tools", mcpToolsHandler)
	router.POST("/mcp/call", mcpCallHandler)
	router.GET("/system-prompt", getSystemPromptHandler)
	router.POST("/system-prompt", setSystemPromptHandler)

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
