package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/cookiejar"
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
	ID      interface{} `json:"id"` // Can be int or string
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
	mcpServers      []MCPServer
	mcpMutex        sync.RWMutex
	systemPrompt    string
	plannerPrompt   string
	finalizerPrompt string
	promptMutex     sync.RWMutex
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
	// Load system prompt
	systemPromptPath := os.Getenv("SYSTEM_PROMPT_PATH")
	if systemPromptPath == "" {
		systemPromptPath = "/app/system_prompt.txt"
	}

	if content, err := os.ReadFile(systemPromptPath); err == nil {
		promptMutex.Lock()
		systemPrompt = strings.TrimSpace(string(content))
		promptMutex.Unlock()
		log.Printf("Loaded system prompt from %s (%d characters)", systemPromptPath, len(systemPrompt))
	} else {
		log.Printf("No system prompt file found at %s, using default", systemPromptPath)
		promptMutex.Lock()
		systemPrompt = "You are a helpful AI assistant."
		promptMutex.Unlock()
	}

	// Load planner prompt
	plannerPromptPath := os.Getenv("PLANNER_PROMPT_PATH")
	if plannerPromptPath == "" {
		plannerPromptPath = "/app/planner_prompt.txt"
	}

	if content, err := os.ReadFile(plannerPromptPath); err == nil {
		promptMutex.Lock()
		plannerPrompt = strings.TrimSpace(string(content))
		promptMutex.Unlock()
		log.Printf("Loaded planner prompt from %s (%d characters)", plannerPromptPath, len(plannerPrompt))
	} else {
		log.Printf("No planner prompt file found at %s, using default", plannerPromptPath)
		promptMutex.Lock()
		plannerPrompt = "You are a planner."
		promptMutex.Unlock()
	}

	// Load finalizer prompt
	finalizerPromptPath := os.Getenv("FINALIZER_PROMPT_PATH")
	if finalizerPromptPath == "" {
		finalizerPromptPath = "/app/finalizer_prompt.txt"
	}

	if content, err := os.ReadFile(finalizerPromptPath); err == nil {
		promptMutex.Lock()
		finalizerPrompt = strings.TrimSpace(string(content))
		promptMutex.Unlock()
		log.Printf("Loaded finalizer prompt from %s (%d characters)", finalizerPromptPath, len(finalizerPrompt))
	} else {
		log.Printf("No finalizer prompt file found at %s, using default", finalizerPromptPath)
		promptMutex.Lock()
		finalizerPrompt = "Provide a helpful response."
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
			for i := range servers {
				discoverMCPTools(&servers[i])
			}
		} else {
			log.Printf("Failed to parse MCP servers config: %v", err)
		}
	} else {
		log.Printf("No MCP servers config found at %s", mcpPath)
	}
}

// Discover tools from an MCP server with proper lifecycle
func discoverMCPTools(server *MCPServer) {
	log.Printf("Starting MCP tool discovery for server: %s at %s", server.Name, server.URL)

	// Create HTTP client with cookie jar to maintain session
	jar, _ := cookiejar.New(nil)
	client := &http.Client{
		Timeout: 30 * time.Second,
		Jar:     jar,
	}
	baseURL := server.URL + "/mcp"

	// Step 1: Initialize the MCP session
	initReq := MCPRequest{
		Jsonrpc: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]interface{}{
				"roots": map[string]interface{}{
					"listChanged": true,
				},
			},
			"clientInfo": map[string]interface{}{
				"name":    "RKLLM-Go-API",
				"version": "1.0.0",
			},
		},
	}

	sessionID, success := sendMCPRequestWithSession(client, baseURL, server, initReq, "initialize", "")
	if !success {
		return
	}

	// Step 2: Send initialized notification
	initNotification := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
	}

	if !sendMCPNotificationWithSession(client, baseURL, server, initNotification, sessionID) {
		return
	}

	// Step 3: Now we can list tools
	toolsReq := MCPRequest{
		Jsonrpc: "2.0",
		ID:      2,
		Method:  "tools/list",
		Params:  map[string]interface{}{},
	}

	if _, success := sendMCPRequestWithSession(client, baseURL, server, toolsReq, "tools/list", sessionID); !success {
		return
	}
}

// Helper function to send MCP requests and handle responses with session management
func sendMCPRequestWithSession(client *http.Client, baseURL string, server *MCPServer, req MCPRequest, requestType string, sessionID string) (string, bool) {
	return sendMCPRequestWithSessionAndResult(client, baseURL, server, req, requestType, sessionID, nil)
}

// Extended version that can capture tool call results
func sendMCPRequestWithSessionAndResult(client *http.Client, baseURL string, server *MCPServer, req MCPRequest, requestType string, sessionID string, resultPtr *interface{}) (string, bool) {
	reqBody, _ := json.Marshal(req)

	httpReq, err := http.NewRequest("POST", baseURL, bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("Failed to create %s request for %s: %v", requestType, server.Name, err)
		return "", false
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("MCP-Protocol-Version", "2024-11-05")

	// Add session ID if provided
	if sessionID != "" {
		httpReq.Header.Set("mcp-session-id", sessionID)
	}

	// Add custom headers if provided
	for k, v := range server.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Failed to send %s request to %s: %v", requestType, server.Name, err)
		return "", false
	}
	defer resp.Body.Close()

	log.Printf("MCP %s response from %s: status=%d", requestType, server.Name, resp.StatusCode)

	// Extract session ID from response headers (for initialize requests)
	newSessionID := resp.Header.Get("mcp-session-id")
	if requestType == "initialize" && newSessionID != "" {
		log.Printf("Received session ID from %s: %s", server.Name, newSessionID)
	}

	body, _ := io.ReadAll(resp.Body)

	// Handle SSE response format from FastMCP
	var mcpResp MCPResponse
	bodyStr := string(body)

	// Check if it's SSE format
	if strings.HasPrefix(bodyStr, "event:") {
		// Parse SSE format: extract JSON from "data: {...}" line
		lines := strings.Split(bodyStr, "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "data: ") {
				jsonData := strings.TrimPrefix(line, "data: ")
				if err := json.Unmarshal([]byte(jsonData), &mcpResp); err != nil {
					log.Printf("Failed to parse SSE JSON from %s: %v", server.Name, err)
					return "", false
				}
				break
			}
		}
	} else {
		// Standard JSON response
		if err := json.Unmarshal(body, &mcpResp); err != nil {
			log.Printf("Failed to parse JSON from %s: %v", server.Name, err)
			return "", false
		}
	}

	// Check for errors
	if mcpResp.Error != nil {
		log.Printf("MCP %s error from %s: %v", requestType, server.Name, mcpResp.Error)
		return newSessionID, false
	}

	// Also check HTTP status for 400+ errors
	if resp.StatusCode >= 400 {
		log.Printf("MCP %s HTTP error from %s: status=%d, body=%s", requestType, server.Name, resp.StatusCode, bodyStr)
		return newSessionID, false
	}

	log.Printf("MCP %s response from %s parsed successfully", requestType, server.Name)

	// Special handling for tools/list response
	if requestType == "tools/list" {
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

	// Capture result if pointer provided (for tool calls)
	if resultPtr != nil && mcpResp.Result != nil {
		*resultPtr = mcpResp.Result
	}

	return newSessionID, true
}

// Helper function to send MCP notifications (no response expected)
func sendMCPNotificationWithSession(client *http.Client, baseURL string, server *MCPServer, notification map[string]interface{}, sessionID string) bool {
	reqBody, _ := json.Marshal(notification)

	httpReq, err := http.NewRequest("POST", baseURL, bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("Failed to create initialized notification for %s: %v", server.Name, err)
		return false
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("MCP-Protocol-Version", "2024-11-05")

	// Add session ID
	if sessionID != "" {
		httpReq.Header.Set("mcp-session-id", sessionID)
	}

	// Add custom headers if provided
	for k, v := range server.Headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Failed to send initialized notification to %s: %v", server.Name, err)
		return false
	}
	defer resp.Body.Close()

	log.Printf("MCP initialized notification sent to %s: status=%d", server.Name, resp.StatusCode)
	return true
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

// Call an MCP tool with proper session management for FastMCP
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

	// Create a new session for this tool call (FastMCP requirement)
	jar, _ := cookiejar.New(nil)
	client := &http.Client{
		Timeout: 30 * time.Second,
		Jar:     jar,
	}
	baseURL := server.URL + "/mcp"

	// Step 1: Initialize session
	initReq := MCPRequest{
		Jsonrpc: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]interface{}{
				"roots": map[string]interface{}{
					"listChanged": true,
				},
			},
			"clientInfo": map[string]interface{}{
				"name":    "RKLLM-Go-API-ToolCall",
				"version": "1.0.0",
			},
		},
	}

	sessionID, success := sendMCPRequestWithSession(client, baseURL, server, initReq, "initialize", "")
	if !success {
		return nil, fmt.Errorf("failed to initialize MCP session for tool call")
	}

	// Step 2: Send initialized notification
	initNotification := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
	}

	if !sendMCPNotificationWithSession(client, baseURL, server, initNotification, sessionID) {
		return nil, fmt.Errorf("failed to send initialized notification for tool call")
	}

	// Step 3: Call the tool
	toolReq := MCPRequest{
		Jsonrpc: "2.0",
		ID:      3,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name":      toolName,
			"arguments": params,
		},
	}

	var toolResult interface{}
	_, success = sendMCPRequestWithSessionAndResult(client, baseURL, server, toolReq, "tools/call", sessionID, &toolResult)
	if !success {
		return nil, fmt.Errorf("tool call failed")
	}

	return toolResult, nil
}

// Handle tool result interpretation by having the AI respond conversationally
func handleToolResultInterpretation(toolCall struct {
	Tool   string                 `json:"tool"`
	Server string                 `json:"server"`
	Params map[string]interface{} `json:"params"`
}, result interface{}) (string, bool) {
	// Create a follow-up prompt for the AI to interpret the tool results
	resultJSON, _ := json.MarshalIndent(result, "", "  ")

	interpretationPrompt := fmt.Sprintf(`You just called the tool "%s" with parameters %v and received these results:

%s

Now respond to the user in a helpful, conversational way. Summarize what you found and offer next steps if appropriate. Do not show raw JSON - interpret and explain the results naturally.`,
		toolCall.Tool,
		toolCall.Params,
		string(resultJSON))

	// Build the interpretation prompt with system context
	fullPrompt := buildEnhancedPrompt(interpretationPrompt)

	// Run inference to get conversational response
	input := rkllm.Input{
		Role:        "",
		InputType:   rkllm.InputTypePrompt,
		PromptInput: fullPrompt,
	}

	inferParam := rkllm.InferParam{
		Mode:        rkllm.InferModeGenerate,
		KeepHistory: 0, // Don't keep history for tool interpretation
	}

	// Temporarily store current result and run interpretation
	resultMutex.Lock()
	originalResult := currentResult
	currentResult = ""
	resultMutex.Unlock()

	err := llmHandle.Run(input, inferParam)
	if err != nil {
		log.Printf("Tool result interpretation failed: %v", err)
		// Restore original result and return raw data as fallback
		resultMutex.Lock()
		currentResult = originalResult
		resultMutex.Unlock()
		return fmt.Sprintf("Tool call completed but interpretation failed. Raw result:\n```json\n%s\n```", string(resultJSON)), true
	}

	// Get the interpretation result
	resultMutex.Lock()
	interpretation := currentResult
	currentResult = originalResult // Restore original for any further processing
	resultMutex.Unlock()

	return interpretation, true
}

// Run represents a multi-step agent execution
type Run struct {
	ID          string       `json:"id"`
	UserQuery   string       `json:"user_query"`
	Scratchpad  []string     `json:"scratchpad"`
	ToolResults []ToolResult `json:"tool_results"`
	Status      string       `json:"status"`
	Facts       []string     `json:"facts"`
}

// ToolCall represents a planned tool execution
type ToolCall struct {
	Name   string                 `json:"name"`
	Args   map[string]interface{} `json:"args"`
	CallID string                 `json:"call_id"`
}

// ToolResult represents the result of a tool execution
type ToolResult struct {
	CallID string      `json:"call_id"`
	Name   string      `json:"name"`
	OK     bool        `json:"ok"`
	Data   interface{} `json:"data,omitempty"`
	Error  string      `json:"error,omitempty"`
}

// PlannerResponse represents the planner's JSON output
type PlannerResponse struct {
	ToolCalls  []ToolCall `json:"tool_calls"`
	StopReason string     `json:"stop_reason"`
	Facts      []string   `json:"facts"`
	Reasoning  string     `json:"reasoning,omitempty"`
}

// Execute a multi-phase agent run
func executeAgentRun(userQuery string) (string, error) {
	run := &Run{
		ID:          fmt.Sprintf("run_%d", time.Now().Unix()),
		UserQuery:   userQuery,
		Scratchpad:  []string{},
		ToolResults: []ToolResult{},
		Status:      "planning",
		Facts:       []string{},
	}

	maxSteps := 6
	for step := 1; step <= maxSteps; step++ {
		log.Printf("Agent Run %s - Step %d: %s", run.ID, step, run.Status)

		// Phase 1: PLAN
		plan, err := runPlanner(run)
		if err != nil {
			return "", fmt.Errorf("planner failed at step %d: %v", step, err)
		}

		// Update facts from planner
		run.Facts = append(run.Facts, plan.Facts...)

		// Phase 2: EXECUTE tools if any were planned
		if len(plan.ToolCalls) > 0 {
			log.Printf("Agent Run %s - Step %d: executing %d tool calls", run.ID, step, len(plan.ToolCalls))
			run.Status = "executing"
			results := executeToolCalls(plan.ToolCalls)
			run.ToolResults = append(run.ToolResults, results...)

			// Update scratchpad with results
			for _, result := range results {
				if result.OK {
					run.Scratchpad = append(run.Scratchpad,
						fmt.Sprintf("Tool %s (call_id:%s) succeeded: %v",
							result.Name, result.CallID, result.Data))
				} else {
					run.Scratchpad = append(run.Scratchpad,
						fmt.Sprintf("Tool %s (call_id:%s) failed: %s",
							result.Name, result.CallID, result.Error))
				}
			}
			run.Status = "planning" // Continue planning after tool execution
			// Check if ready to answer after executing tools
			if plan.StopReason == "ready_to_answer" && len(run.Facts) > 0 {
				run.Status = "finalizing"
				break
			}
		} else if plan.StopReason == "ready_to_answer" {
			// No tool calls planned - either trivial query or enough info already
			log.Printf("Agent Run %s - Step %d: ready to answer without tool calls", run.ID, step)
			run.Status = "finalizing"
			break
		}
	}

	// Phase 3: FINALIZE
	finalAnswer, err := runFinalizer(run)
	if err != nil {
		return "", fmt.Errorf("finalizer failed: %v", err)
	}

	return finalAnswer, nil
}

// Run the planner phase
func runPlanner(run *Run) (*PlannerResponse, error) {
	plannerPrompt := buildPlannerPrompt(run)

	// Use low temperature for structured planning
	input := rkllm.Input{
		Role:        "",
		InputType:   rkllm.InputTypePrompt,
		PromptInput: plannerPrompt,
	}

	inferParam := rkllm.InferParam{
		Mode:        rkllm.InferModeGenerate,
		KeepHistory: 0,
	}

	// Run planner inference
	resultMutex.Lock()
	originalResult := currentResult
	currentResult = ""
	resultMutex.Unlock()

	err := llmHandle.Run(input, inferParam)
	if err != nil {
		resultMutex.Lock()
		currentResult = originalResult
		resultMutex.Unlock()
		return nil, err
	}

	// Get planner result
	resultMutex.Lock()
	plannerOutput := currentResult
	currentResult = originalResult
	resultMutex.Unlock()

	// Parse JSON response
	var plan PlannerResponse
	// Handle thinking tags if present
	jsonContent := plannerOutput
	if strings.Contains(plannerOutput, "</think>") {
		parts := strings.Split(plannerOutput, "</think>")
		if len(parts) > 1 {
			jsonContent = strings.TrimSpace(parts[1])
		}
	}

	log.Printf("Planner JSON response: %s", jsonContent)

	if err := json.Unmarshal([]byte(jsonContent), &plan); err != nil {
		return nil, fmt.Errorf("failed to parse planner JSON: %v\nOutput: %s", err, plannerOutput)
	}

	log.Printf("Planner result: %d tool_calls, stop_reason=%s, %d facts",
		len(plan.ToolCalls), plan.StopReason, len(plan.Facts))

	return &plan, nil
}

// Execute multiple tool calls
func executeToolCalls(toolCalls []ToolCall) []ToolResult {
	var results []ToolResult

	for _, toolCall := range toolCalls {
		// Map tool names to server names (simplified mapping)
		serverName := "media" // Default to media server

		log.Printf("Executing tool call: %s.%s with args: %v (call_id: %s)",
			serverName, toolCall.Name, toolCall.Args, toolCall.CallID)

		result, err := callMCPTool(serverName, toolCall.Name, toolCall.Args)
		if err != nil {
			log.Printf("Tool call %s failed: %v", toolCall.CallID, err)
			results = append(results, ToolResult{
				CallID: toolCall.CallID,
				Name:   toolCall.Name,
				OK:     false,
				Error:  err.Error(),
			})
		} else {
			resultJSON, _ := json.MarshalIndent(result, "", "  ")
			log.Printf("Tool call %s succeeded. Result: %s", toolCall.CallID, string(resultJSON))
			results = append(results, ToolResult{
				CallID: toolCall.CallID,
				Name:   toolCall.Name,
				OK:     true,
				Data:   result,
			})
		}
	}

	return results
}

// Run the finalizer phase
func runFinalizer(run *Run) (string, error) {
	finalizerPrompt := buildFinalizerPrompt(run)

	// Use higher temperature for creative final response
	input := rkllm.Input{
		Role:        "",
		InputType:   rkllm.InputTypePrompt,
		PromptInput: finalizerPrompt,
	}

	inferParam := rkllm.InferParam{
		Mode:        rkllm.InferModeGenerate,
		KeepHistory: 0,
	}

	// Run finalizer inference
	resultMutex.Lock()
	originalResult := currentResult
	currentResult = ""
	resultMutex.Unlock()

	err := llmHandle.Run(input, inferParam)
	if err != nil {
		resultMutex.Lock()
		currentResult = originalResult
		resultMutex.Unlock()
		return "", err
	}

	// Get finalizer result
	resultMutex.Lock()
	finalAnswer := currentResult
	currentResult = originalResult
	resultMutex.Unlock()

	return finalAnswer, nil
}

// Build planner prompt
func buildPlannerPrompt(run *Run) string {
	// Get tools list
	var toolsList []string
	mcpMutex.RLock()
	for _, server := range mcpServers {
		for _, tool := range server.Tools {
			toolsList = append(toolsList, fmt.Sprintf("- %s: %s", tool.Name, tool.Description))
		}
	}
	mcpMutex.RUnlock()
	tools := strings.Join(toolsList, "\n")

	// Build scratchpad section
	var scratchpad string
	if len(run.Scratchpad) > 0 {
		scratchpad = "\nPrevious results:\n"
		for _, entry := range run.Scratchpad {
			scratchpad += "- " + entry + "\n"
		}
	}

	// Replace placeholders in template
	promptMutex.RLock()
	prompt := plannerPrompt
	promptMutex.RUnlock()

	prompt = strings.ReplaceAll(prompt, "{TOOLS}", tools)
	prompt = strings.ReplaceAll(prompt, "{USER_QUERY}", run.UserQuery)
	prompt = strings.ReplaceAll(prompt, "{SCRATCHPAD}", scratchpad)

	return prompt
}

// Extract user query from enhanced prompt
func extractUserQuery(enhancedPrompt string) string {
	lines := strings.Split(enhancedPrompt, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "User: ") {
			return strings.TrimPrefix(line, "User: ")
		}
	}
	// Fallback - return the last line or the whole prompt
	if len(lines) > 0 {
		return lines[len(lines)-1]
	}
	return enhancedPrompt
}

// Build finalizer prompt
func buildFinalizerPrompt(run *Run) string {
	// Build facts section
	var factsSection string
	if len(run.Facts) > 0 {
		for i, fact := range run.Facts {
			factsSection += fmt.Sprintf("%d. %s\n", i+1, fact)
		}
	} else {
		factsSection = "(No specific facts gathered from tools)\n"
	}

	// Build tool results section
	var toolResultsSection string
	if len(run.ToolResults) > 0 {
		toolResultsSection = "\nTool results:\n"
		for _, result := range run.ToolResults {
			if result.OK {
				resultJSON, _ := json.MarshalIndent(result.Data, "", "  ")
				toolResultsSection += fmt.Sprintf("- %s: %s\n", result.Name, string(resultJSON))
			}
		}
	}

	// Replace placeholders in template
	promptMutex.RLock()
	prompt := finalizerPrompt
	sysPrompt := systemPrompt
	promptMutex.RUnlock()

	prompt = strings.ReplaceAll(prompt, "{SYSTEM_PROMPT}", "System: "+sysPrompt)
	prompt = strings.ReplaceAll(prompt, "{USER_QUERY}", run.UserQuery)
	prompt = strings.ReplaceAll(prompt, "{FACTS}", factsSection)
	prompt = strings.ReplaceAll(prompt, "{TOOL_RESULTS}", toolResultsSection)

	return prompt
}

// Handle tool calls in LLM responses (legacy - now using executeAgentRun)
func handleToolCall(response string) (string, bool) {
	// Look for JSON tool calls in the response
	// Handle both thinking format and direct JSON
	var jsonContent string

	// Check if response contains thinking tags
	if strings.Contains(response, "</think>") {
		// Extract content after thinking
		parts := strings.Split(response, "</think>")
		if len(parts) > 1 {
			jsonContent = strings.TrimSpace(parts[1])
		}
	} else {
		jsonContent = strings.TrimSpace(response)
	}

	// Try to parse as JSON tool call
	var toolCall struct {
		Tool   string                 `json:"tool"`
		Server string                 `json:"server"`
		Params map[string]interface{} `json:"params"`
	}

	if err := json.Unmarshal([]byte(jsonContent), &toolCall); err != nil {
		// Not a tool call, return original response
		return response, false
	}

	// Validate required fields
	if toolCall.Tool == "" || toolCall.Server == "" {
		return response, false
	}

	log.Printf("Executing tool call: %s.%s with params: %v", toolCall.Server, toolCall.Tool, toolCall.Params)

	// Execute the tool call
	result, err := callMCPTool(toolCall.Server, toolCall.Tool, toolCall.Params)
	if err != nil {
		log.Printf("Tool call failed: %v", err)
		return fmt.Sprintf("Tool call failed: %v", err), true
	}

	// Instead of returning raw JSON, let the AI interpret the results
	// We'll re-run the LLM with the tool results to get a conversational response
	return handleToolResultInterpretation(toolCall, result)
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
				// Extract required parameters from schema
				var paramInfo string
				if schema, ok := tool.InputSchema["properties"].(map[string]interface{}); ok {
					var paramNames []string
					for paramName := range schema {
						paramNames = append(paramNames, fmt.Sprintf(`"%s"`, paramName))
					}
					if len(paramNames) > 0 {
						paramInfo = fmt.Sprintf(" [params: %s]", strings.Join(paramNames, ", "))
					}
				}
				toolDesc := fmt.Sprintf("- %s (server: %s)%s: %s", tool.Name, server.Name, paramInfo, tool.Description)
				parts = append(parts, toolDesc)
			}
		}
		parts = append(parts, "\nTo use a tool, respond with JSON: {\"tool\": \"toolName\", \"server\": \"serverName\", \"params\": {...}}")
		parts = append(parts, fmt.Sprintf("CRITICAL: The server name is '%s' - do NOT use 'Radarr', 'Sonarr', or any other name!", mcpServers[0].Name))
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

	log.Printf("Running inference for prompt: %s (stream=%v)\n", req.Prompt, req.Stream)
	startTime := time.Now()

	if req.Stream {
		// Handle streaming with agent architecture
		handleStreamingChatWithAgent(c, req.Prompt)
	} else {
		// Use multi-phase agent architecture directly
		agentResponse, err := executeAgentRun(req.Prompt)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Agent run failed: %v", err)})
			return
		}

		elapsed := time.Since(startTime)
		log.Printf("Agent run completed in %v\n", elapsed)

		response := ChatResponse{
			Text: agentResponse,
		}
		c.JSON(http.StatusOK, response)
	}
}

// handleStreamingChat handles Server-Sent Events streaming with tool call support
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

	// Buffer to detect tool calls before streaming
	var responseBuffer strings.Builder
	var isBuffering = false // Track if we're in buffering mode

	// Stream responses with tool call detection
	for response := range streamChannel {
		if response.Error != "" {
			// Send error event
			c.SSEvent("error", response)
			break
		} else if response.Finished {
			// Check if we need to use agent architecture or legacy tool handling
			finalText := responseBuffer.String()
			if looksLikeToolCall(finalText) {
				// Extract the original user query from the enhanced prompt
				userQuery := extractUserQuery(input.PromptInput)
				if agentResponse, err := executeAgentRun(userQuery); err != nil {
					log.Printf("Streaming agent run failed: %v", err)
					// Fallback to legacy tool handling
					if toolCallResponse, executed := handleToolCall(finalText); executed {
						streamToolInterpretation(c, toolCallResponse)
					} else {
						streamToolInterpretation(c, finalText)
					}
				} else {
					// Stream the agent's final response
					streamToolInterpretation(c, agentResponse)
				}
			} else {
				// Send the final buffered response (if we were buffering) or final event
				if isBuffering {
					c.SSEvent("done", &ChatResponse{
						Text:      finalText,
						Finished:  true,
						PerfStats: response.PerfStats,
					})
				} else {
					c.SSEvent("done", response)
				}
			}
			break
		} else {
			// Always buffer the delta
			responseBuffer.WriteString(response.Delta)
			bufferedText := responseBuffer.String()

			// Check if we should start/continue buffering
			if looksLikeToolCall(bufferedText) {
				isBuffering = true
				// Don't stream anything - keep buffering
			} else if !isBuffering {
				// Stream normally if we haven't started buffering yet
				c.SSEvent("delta", response)
				c.Writer.Flush()
			}
			// If isBuffering is true but doesn't look like tool call anymore,
			// we still keep buffering until completion to be safe
		}
	}
}

// Check if the text looks like it might be a tool call (to buffer vs stream)
func looksLikeToolCall(text string) bool {
	trimmed := strings.TrimSpace(text)
	// Be very aggressive - any JSON-like start should be buffered
	// This prevents "{" and "{\"tool" from leaking through
	return strings.HasPrefix(trimmed, "{") ||
		   strings.HasPrefix(trimmed, `{"`) ||
		   strings.Contains(trimmed, `"tool"`) ||
		   strings.Contains(trimmed, `"server"`)
}

// Stream tool interpretation results
func streamToolInterpretation(c *gin.Context, interpretation string) {
	// Split interpretation into chunks for streaming effect
	words := strings.Fields(interpretation)

	c.SSEvent("delta", &ChatResponse{
		Text:     "",
		Delta:    "🔧 ", // Tool indicator
		Finished: false,
	})
	c.Writer.Flush()

	// Stream words with small delays to simulate natural typing
	var accumulated strings.Builder
	for i, word := range words {
		accumulated.WriteString(word)
		if i < len(words)-1 {
			accumulated.WriteString(" ")
		}

		c.SSEvent("delta", &ChatResponse{
			Text:     accumulated.String(),
			Delta:    word + " ",
			Finished: false,
		})
		c.Writer.Flush()

		// Small delay for natural feel (optional)
		time.Sleep(50 * time.Millisecond)
	}

	// Send completion
	c.SSEvent("done", &ChatResponse{
		Text:     interpretation,
		Finished: true,
	})
}

// handleStreamingChatWithAgent handles streaming with the agent architecture
func handleStreamingChatWithAgent(c *gin.Context, userQuery string) {
	// Set headers for Server-Sent Events
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	// Run agent in background
	go func() {
		agentResponse, err := executeAgentRun(userQuery)
		if err != nil {
			log.Printf("Streaming agent run failed: %v", err)
			c.SSEvent("error", &ChatResponse{
				Error:    fmt.Sprintf("Agent run failed: %v", err),
				Finished: true,
			})
			return
		}

		// Stream the response
		streamToolInterpretation(c, agentResponse)
	}()
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
