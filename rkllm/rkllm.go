package rkllm

/*
#cgo CFLAGS: -I${SRCDIR}/../../ezrknn-llm/rkllm-runtime/Linux/librkllm_api/include
#cgo LDFLAGS: -L${SRCDIR}/../../ezrknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64 -lrkllmrt -Wl,-rpath,${SRCDIR}/../../ezrknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "rkllm.h"

// Callback wrapper for Go
extern int goCallbackWrapper(RKLLMResult *result, void *userdata, LLMCallState state);

// Helper function to create a callback pointer
static LLMResultCallback get_callback_ptr() {
    return (LLMResultCallback)goCallbackWrapper;
}

// Helper to set prompt_input in the union
static void set_prompt_input(RKLLMInput* input, const char* prompt) {
    input->prompt_input = prompt;
}
*/
import "C"
import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// CallState represents the state of an LLM call
type CallState int

const (
	CallStateNormal  CallState = 0
	CallStateWaiting CallState = 1
	CallStateFinish  CallState = 2
	CallStateError   CallState = 3
)

// InputType defines the types of inputs that can be fed into the LLM
type InputType int

const (
	InputTypePrompt      InputType = 0
	InputTypeToken       InputType = 1
	InputTypeEmbed       InputType = 2
	InputTypeMultimodal  InputType = 3
)

// InferMode specifies the inference modes of the LLM
type InferMode int

const (
	InferModeGenerate            InferMode = 0
	InferModeGetLastHiddenLayer  InferMode = 1
	InferModeGetLogits           InferMode = 2
)

// ExtendParam contains extended parameters for configuring an LLM instance
type ExtendParam struct {
	BaseDomainID     int32
	EmbedFlash       int8
	EnabledCPUsNum   int8
	EnabledCPUsMask  uint32
	NBatch           uint8
	UseCrossAttn     int8
}

// Param defines the parameters for configuring an LLM instance
type Param struct {
	ModelPath          string
	MaxContextLen      int32
	MaxNewTokens       int32
	TopK               int32
	NKeep              int32
	TopP               float32
	Temperature        float32
	RepeatPenalty      float32
	FrequencyPenalty   float32
	PresencePenalty    float32
	Mirostat           int32
	MirostatTau        float32
	MirostatEta        float32
	SkipSpecialToken   bool
	IsAsync            bool
	ImgStart           string
	ImgEnd             string
	ImgContent         string
	ExtendParam        ExtendParam
}

// Input represents different types of input to the LLM
type Input struct {
	Role           string
	EnableThinking bool
	InputType      InputType
	PromptInput    string
}

// InferParam structure for defining parameters during inference
type InferParam struct {
	Mode        InferMode
	KeepHistory int
}

// PerfStat holds performance statistics
type PerfStat struct {
	PrefillTimeMs   float32
	PrefillTokens   int
	GenerateTimeMs  float32
	GenerateTokens  int
	MemoryUsageMB   float32
}

// Result represents the result of LLM inference
type Result struct {
	Text    string
	TokenID int32
	Perf    PerfStat
}

// Handle represents an LLM instance
type Handle struct {
	cHandle  C.LLMHandle
	callback ResultCallback
	mu       sync.Mutex
}

// ResultCallback is a Go callback function type
type ResultCallback func(result *Result, state CallState) int

var (
	callbackRegistry = make(map[unsafe.Pointer]ResultCallback)
	callbackMutex    sync.RWMutex
)

// CreateDefaultParam creates a default Param structure with preset values
func CreateDefaultParam() Param {
	cParam := C.rkllm_createDefaultParam()

	return Param{
		MaxContextLen:      int32(cParam.max_context_len),
		MaxNewTokens:       int32(cParam.max_new_tokens),
		TopK:               int32(cParam.top_k),
		NKeep:              int32(cParam.n_keep),
		TopP:               float32(cParam.top_p),
		Temperature:        float32(cParam.temperature),
		RepeatPenalty:      float32(cParam.repeat_penalty),
		FrequencyPenalty:   float32(cParam.frequency_penalty),
		PresencePenalty:    float32(cParam.presence_penalty),
		Mirostat:           int32(cParam.mirostat),
		MirostatTau:        float32(cParam.mirostat_tau),
		MirostatEta:        float32(cParam.mirostat_eta),
		SkipSpecialToken:   bool(cParam.skip_special_token),
		IsAsync:            bool(cParam.is_async),
		ExtendParam: ExtendParam{
			BaseDomainID:    int32(cParam.extend_param.base_domain_id),
			EmbedFlash:      int8(cParam.extend_param.embed_flash),
			EnabledCPUsNum:  int8(cParam.extend_param.enabled_cpus_num),
			EnabledCPUsMask: uint32(cParam.extend_param.enabled_cpus_mask),
			NBatch:          uint8(cParam.extend_param.n_batch),
			UseCrossAttn:    int8(cParam.extend_param.use_cross_attn),
		},
	}
}

// Init initializes the LLM with the given parameters
func Init(param Param, callback ResultCallback) (*Handle, error) {
	handle := &Handle{
		callback: callback,
	}

	// Convert Go Param to C RKLLMParam
	// Start with default params to ensure all fields are properly initialized
	cParam := C.rkllm_createDefaultParam()

	cModelPath := C.CString(param.ModelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	cParam.model_path = cModelPath
	cParam.max_context_len = C.int(param.MaxContextLen)
	cParam.max_new_tokens = C.int(param.MaxNewTokens)
	cParam.top_k = C.int(param.TopK)
	cParam.n_keep = C.int(param.NKeep)
	cParam.top_p = C.float(param.TopP)
	cParam.temperature = C.float(param.Temperature)
	cParam.repeat_penalty = C.float(param.RepeatPenalty)
	cParam.frequency_penalty = C.float(param.FrequencyPenalty)
	cParam.presence_penalty = C.float(param.PresencePenalty)
	cParam.mirostat = C.int(param.Mirostat)
	cParam.mirostat_tau = C.float(param.MirostatTau)
	cParam.mirostat_eta = C.float(param.MirostatEta)
	cParam.skip_special_token = C.bool(param.SkipSpecialToken)
	cParam.is_async = C.bool(param.IsAsync)

	// Set extend params - only override defaults if explicitly provided (non-zero)
	// The C library sets good defaults (enabled_cpus_num=4, enabled_cpus_mask=0xF0, embed_flash=1)
	// We should preserve those unless user explicitly overrides them
	if param.ExtendParam.BaseDomainID != 0 {
		cParam.extend_param.base_domain_id = C.int(param.ExtendParam.BaseDomainID)
	}
	if param.ExtendParam.EmbedFlash != 0 {
		cParam.extend_param.embed_flash = C.int8_t(param.ExtendParam.EmbedFlash)
	}
	if param.ExtendParam.EnabledCPUsNum != 0 {
		cParam.extend_param.enabled_cpus_num = C.int8_t(param.ExtendParam.EnabledCPUsNum)
	}
	if param.ExtendParam.EnabledCPUsMask != 0 {
		cParam.extend_param.enabled_cpus_mask = C.uint32_t(param.ExtendParam.EnabledCPUsMask)
	}
	if param.ExtendParam.NBatch != 0 {
		cParam.extend_param.n_batch = C.uint8_t(param.ExtendParam.NBatch)
	}
	// use_cross_attn can be legitimately 0, so always set it
	cParam.extend_param.use_cross_attn = C.int8_t(param.ExtendParam.UseCrossAttn)

	// Pre-register callback with a temporary nil key (will update after init)	// Pre-register callback with a temporary nil key (will update after init)
	// This ensures the callback is available if rkllm_init calls it
	callbackMutex.Lock()
	callbackRegistry[unsafe.Pointer(nil)] = callback
	callbackMutex.Unlock()

	// Initialize the LLM
	ret := C.rkllm_init(&handle.cHandle, &cParam, C.get_callback_ptr())

	// Remove temp registration and add proper one
	callbackMutex.Lock()
	delete(callbackRegistry, unsafe.Pointer(nil))
	if ret == 0 {
		callbackRegistry[unsafe.Pointer(handle.cHandle)] = callback
	}
	callbackMutex.Unlock()

	if ret != 0 {
		return nil, fmt.Errorf("rkllm_init failed with code: %d", ret)
	}

	return handle, nil
}

// Run runs an LLM inference task synchronously
func (h *Handle) Run(input Input, inferParam InferParam) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cHandle == nil {
		return errors.New("handle is not initialized")
	}

	// Convert Go Input to C RKLLMInput
	cInput := C.RKLLMInput{}

	if input.Role != "" {
		cRole := C.CString(input.Role)
		defer C.free(unsafe.Pointer(cRole))
		cInput.role = cRole
	}

	cInput.enable_thinking = C.bool(input.EnableThinking)
	cInput.input_type = C.RKLLMInputType(input.InputType)

	if input.InputType == InputTypePrompt {
		cPrompt := C.CString(input.PromptInput)
		defer C.free(unsafe.Pointer(cPrompt))
		C.set_prompt_input(&cInput, cPrompt)
	}

	// Convert Go InferParam to C RKLLMInferParam
	cInferParam := C.RKLLMInferParam{}
	cInferParam.mode = C.RKLLMInferMode(inferParam.Mode)
	cInferParam.keep_history = C.int(inferParam.KeepHistory)

	// Run inference - pass handle pointer as userdata to identify which callback to use
	ret := C.rkllm_run(h.cHandle, &cInput, &cInferParam, unsafe.Pointer(h.cHandle))

	if ret != 0 {
		return fmt.Errorf("rkllm_run failed with code: %d", ret)
	}

	return nil
}

// Destroy destroys the LLM instance and releases resources
func (h *Handle) Destroy() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cHandle == nil {
		return nil
	}

	// Unregister callback
	callbackMutex.Lock()
	delete(callbackRegistry, unsafe.Pointer(h.cHandle))
	callbackMutex.Unlock()

	ret := C.rkllm_destroy(h.cHandle)
	h.cHandle = nil

	if ret != 0 {
		return fmt.Errorf("rkllm_destroy failed with code: %d", ret)
	}

	return nil
}

// IsRunning checks if an LLM task is currently running
func (h *Handle) IsRunning() bool {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cHandle == nil {
		return false
	}

	ret := C.rkllm_is_running(h.cHandle)
	return ret != 0  // Returns non-zero when running
}

// Abort aborts an ongoing LLM task
func (h *Handle) Abort() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.cHandle == nil {
		return errors.New("handle is not initialized")
	}

	ret := C.rkllm_abort(h.cHandle)

	if ret != 0 {
		return fmt.Errorf("rkllm_abort failed with code: %d", ret)
	}

	return nil
}
