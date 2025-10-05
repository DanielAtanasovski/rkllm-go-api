package rkllm

/*
#include <stdbool.h>
#include <stdlib.h>
#include "rkllm.h"
*/
import "C"
import (
	"unsafe"
)

//export goCallbackWrapper
func goCallbackWrapper(cResult *C.RKLLMResult, userdata unsafe.Pointer, state C.LLMCallState) C.int {
	// Handle NULL result pointer
	if cResult == nil {
		return 0
	}

	// Convert C result to Go result - safely handle potential NULL text pointer
	var text string
	if cResult.text != nil {
		text = C.GoString(cResult.text)
	}

	result := &Result{
		Text:    text,
		TokenID: int32(cResult.token_id),
		Perf: PerfStat{
			PrefillTimeMs:  float32(cResult.perf.prefill_time_ms),
			PrefillTokens:  int(cResult.perf.prefill_tokens),
			GenerateTimeMs: float32(cResult.perf.generate_time_ms),
			GenerateTokens: int(cResult.perf.generate_tokens),
			MemoryUsageMB:  float32(cResult.perf.memory_usage_mb),
		},
	}

	// Find and call the appropriate Go callback
	// First try to use userdata as handle pointer if it's not NULL
	var cb ResultCallback
	var found bool

	callbackMutex.RLock()
	if userdata != nil {
		handlePtr := unsafe.Pointer(userdata)
		cb, found = callbackRegistry[handlePtr]
	}
	// If not found or userdata was NULL, check for nil key (during initialization)
	if !found {
		cb, found = callbackRegistry[unsafe.Pointer(nil)]
	}
	callbackMutex.RUnlock()

	if found && cb != nil {
		ret := cb(result, CallState(state))
		return C.int(ret)
	}

	return 0
}
