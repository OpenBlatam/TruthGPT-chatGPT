// Package inference provides high-performance batch inference processing.
//
// Features:
// - Parallel batch processing with goroutine pools
// - Dynamic batching with configurable timeouts
// - Priority queuing for latency-sensitive requests
// - Memory-efficient buffer pooling
// - Comprehensive metrics and tracing
package inference

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/panjf2000/ants/v2"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// ════════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// BatchConfig holds the configuration for batch processing.
type BatchConfig struct {
	// Batch size limits
	MinBatchSize int `yaml:"min_batch_size"`
	MaxBatchSize int `yaml:"max_batch_size"`
	
	// Timing
	BatchTimeout    time.Duration `yaml:"batch_timeout"`     // Max wait for batch fill
	RequestTimeout  time.Duration `yaml:"request_timeout"`   // Per-request timeout
	ProcessTimeout  time.Duration `yaml:"process_timeout"`   // Batch processing timeout
	
	// Concurrency
	NumWorkers      int `yaml:"num_workers"`
	QueueSize       int `yaml:"queue_size"`
	
	// Memory
	BufferPoolSize  int `yaml:"buffer_pool_size"`
	MaxTokensPerReq int `yaml:"max_tokens_per_req"`
	
	// Features
	EnablePriority    bool `yaml:"enable_priority"`
	EnablePreemption  bool `yaml:"enable_preemption"`
	EnableCompression bool `yaml:"enable_compression"`
}

// DefaultBatchConfig returns the default batch configuration.
func DefaultBatchConfig() BatchConfig {
	return BatchConfig{
		MinBatchSize:      1,
		MaxBatchSize:      32,
		BatchTimeout:      50 * time.Millisecond,
		RequestTimeout:    30 * time.Second,
		ProcessTimeout:    60 * time.Second,
		NumWorkers:        runtime.NumCPU(),
		QueueSize:         1024,
		BufferPoolSize:    256,
		MaxTokensPerReq:   4096,
		EnablePriority:    true,
		EnablePreemption:  false,
		EnableCompression: true,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// REQUEST & RESPONSE TYPES
// ════════════════════════════════════════════════════════════════════════════════

// Priority levels for request scheduling.
type Priority int

const (
	PriorityLow Priority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

func (p Priority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// InferenceRequest represents a single inference request.
type InferenceRequest struct {
	ID          string                 `json:"id"`
	Input       string                 `json:"input"`
	InputTokens []int32                `json:"input_tokens,omitempty"`
	MaxTokens   int                    `json:"max_tokens"`
	Temperature float32                `json:"temperature"`
	TopP        float32                `json:"top_p"`
	TopK        int                    `json:"top_k"`
	Priority    Priority               `json:"priority"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	
	// Timestamps
	CreatedAt   time.Time `json:"created_at"`
	ProcessedAt time.Time `json:"processed_at,omitempty"`
	
	// Internal
	responseCh chan *InferenceResponse
	ctx        context.Context
}

// NewInferenceRequest creates a new inference request with defaults.
func NewInferenceRequest(input string) *InferenceRequest {
	return &InferenceRequest{
		ID:          uuid.NewString(),
		Input:       input,
		MaxTokens:   256,
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        50,
		Priority:    PriorityNormal,
		CreatedAt:   time.Now(),
		responseCh:  make(chan *InferenceResponse, 1),
	}
}

// InferenceResponse represents the result of an inference request.
type InferenceResponse struct {
	ID           string    `json:"id"`
	RequestID    string    `json:"request_id"`
	Output       string    `json:"output"`
	OutputTokens []int32   `json:"output_tokens,omitempty"`
	TokensUsed   int       `json:"tokens_used"`
	FinishReason string    `json:"finish_reason"`
	LatencyMs    float64   `json:"latency_ms"`
	Error        error     `json:"error,omitempty"`
	CompletedAt  time.Time `json:"completed_at"`
}

// Batch represents a collection of requests to process together.
type Batch struct {
	ID        string              `json:"id"`
	Requests  []*InferenceRequest `json:"requests"`
	CreatedAt time.Time           `json:"created_at"`
	Priority  Priority            `json:"priority"`
}

// NewBatch creates a new batch with the given requests.
func NewBatch(requests []*InferenceRequest) *Batch {
	// Determine batch priority from highest request priority
	maxPriority := PriorityLow
	for _, req := range requests {
		if req.Priority > maxPriority {
			maxPriority = req.Priority
		}
	}
	
	return &Batch{
		ID:        uuid.NewString(),
		Requests:  requests,
		CreatedAt: time.Now(),
		Priority:  maxPriority,
	}
}

// Size returns the number of requests in the batch.
func (b *Batch) Size() int {
	return len(b.Requests)
}

// TotalTokens returns the total input tokens across all requests.
func (b *Batch) TotalTokens() int {
	total := 0
	for _, req := range b.Requests {
		if len(req.InputTokens) > 0 {
			total += len(req.InputTokens)
		} else {
			total += len(req.Input) / 4 // Rough estimate
		}
	}
	return total
}

// ════════════════════════════════════════════════════════════════════════════════
// BATCH PROCESSOR STATS
// ════════════════════════════════════════════════════════════════════════════════

// ProcessorStats holds statistics for the batch processor.
type ProcessorStats struct {
	RequestsReceived   uint64
	RequestsProcessed  uint64
	RequestsFailed     uint64
	BatchesProcessed   uint64
	TotalTokens        uint64
	TotalLatencyNs     uint64
	QueueDepth         int64
	ActiveWorkers      int64
	
	// Batch size histogram (buckets: 1, 2-4, 5-8, 9-16, 17-32, 33+)
	BatchSizeHistogram [6]uint64
}

// AvgLatencyMs returns the average request latency in milliseconds.
func (s *ProcessorStats) AvgLatencyMs() float64 {
	processed := atomic.LoadUint64(&s.RequestsProcessed)
	if processed == 0 {
		return 0
	}
	totalNs := atomic.LoadUint64(&s.TotalLatencyNs)
	return float64(totalNs) / float64(processed) / 1e6
}

// Throughput returns requests per second.
func (s *ProcessorStats) Throughput(duration time.Duration) float64 {
	processed := atomic.LoadUint64(&s.RequestsProcessed)
	if duration == 0 {
		return 0
	}
	return float64(processed) / duration.Seconds()
}

// ════════════════════════════════════════════════════════════════════════════════
// BATCH PROCESSOR
// ════════════════════════════════════════════════════════════════════════════════

// ProcessorFunc is the function type for processing a batch.
type ProcessorFunc func(ctx context.Context, batch *Batch) ([]*InferenceResponse, error)

// BatchProcessor handles dynamic batching and parallel processing.
type BatchProcessor struct {
	config    BatchConfig
	processor ProcessorFunc
	logger    *zap.Logger
	tracer    trace.Tracer
	
	// Request queue (priority-aware)
	queues     [4]chan *InferenceRequest // One per priority level
	
	// Worker pool
	workerPool *ants.Pool
	
	// Pending requests for batch formation
	pending     []*InferenceRequest
	pendingLock sync.Mutex
	
	// Batch timer
	batchTimer *time.Timer
	
	// Stats
	stats   ProcessorStats
	startAt time.Time
	
	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewBatchProcessor creates a new batch processor.
func NewBatchProcessor(config BatchConfig, processor ProcessorFunc, logger *zap.Logger) (*BatchProcessor, error) {
	if processor == nil {
		return nil, fmt.Errorf("processor function is required")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	// Create worker pool
	pool, err := ants.NewPool(config.NumWorkers,
		ants.WithPreAlloc(true),
		ants.WithNonblocking(false),
		ants.WithPanicHandler(func(i interface{}) {
			logger.Error("Worker panic", zap.Any("panic", i))
		}),
	)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create worker pool: %w", err)
	}
	
	bp := &BatchProcessor{
		config:     config,
		processor:  processor,
		logger:     logger,
		tracer:     otel.Tracer("inference.BatchProcessor"),
		workerPool: pool,
		pending:    make([]*InferenceRequest, 0, config.MaxBatchSize),
		startAt:    time.Now(),
		ctx:        ctx,
		cancel:     cancel,
	}
	
	// Initialize priority queues
	for i := range bp.queues {
		bp.queues[i] = make(chan *InferenceRequest, config.QueueSize)
	}
	
	// Start background workers
	bp.wg.Add(1)
	go bp.runBatcher()
	
	return bp, nil
}

// ════════════════════════════════════════════════════════════════════════════════
// PUBLIC METHODS
// ════════════════════════════════════════════════════════════════════════════════

// Submit submits a request for processing and returns a channel for the response.
func (bp *BatchProcessor) Submit(ctx context.Context, req *InferenceRequest) <-chan *InferenceResponse {
	req.ctx = ctx
	if req.responseCh == nil {
		req.responseCh = make(chan *InferenceResponse, 1)
	}
	
	atomic.AddUint64(&bp.stats.RequestsReceived, 1)
	atomic.AddInt64(&bp.stats.QueueDepth, 1)
	
	// Submit to appropriate priority queue
	select {
	case bp.queues[req.Priority] <- req:
		// Successfully queued
	default:
		// Queue full - return error immediately
		req.responseCh <- &InferenceResponse{
			RequestID:    req.ID,
			Error:        fmt.Errorf("queue full"),
			FinishReason: "error",
			CompletedAt:  time.Now(),
		}
	}
	
	return req.responseCh
}

// SubmitAndWait submits a request and waits for the response.
func (bp *BatchProcessor) SubmitAndWait(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	respCh := bp.Submit(ctx, req)
	
	select {
	case resp := <-respCh:
		return resp, resp.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// SubmitBatch submits multiple requests at once.
func (bp *BatchProcessor) SubmitBatch(ctx context.Context, requests []*InferenceRequest) []<-chan *InferenceResponse {
	channels := make([]<-chan *InferenceResponse, len(requests))
	for i, req := range requests {
		channels[i] = bp.Submit(ctx, req)
	}
	return channels
}

// Stats returns the current processor statistics.
func (bp *BatchProcessor) Stats() ProcessorStats {
	return ProcessorStats{
		RequestsReceived:   atomic.LoadUint64(&bp.stats.RequestsReceived),
		RequestsProcessed:  atomic.LoadUint64(&bp.stats.RequestsProcessed),
		RequestsFailed:     atomic.LoadUint64(&bp.stats.RequestsFailed),
		BatchesProcessed:   atomic.LoadUint64(&bp.stats.BatchesProcessed),
		TotalTokens:        atomic.LoadUint64(&bp.stats.TotalTokens),
		TotalLatencyNs:     atomic.LoadUint64(&bp.stats.TotalLatencyNs),
		QueueDepth:         atomic.LoadInt64(&bp.stats.QueueDepth),
		ActiveWorkers:      int64(bp.workerPool.Running()),
		BatchSizeHistogram: bp.stats.BatchSizeHistogram,
	}
}

// Shutdown gracefully shuts down the processor.
func (bp *BatchProcessor) Shutdown(ctx context.Context) error {
	bp.logger.Info("Shutting down batch processor")
	
	bp.cancel()
	
	// Close queues
	for i := range bp.queues {
		close(bp.queues[i])
	}
	
	// Wait for background workers
	done := make(chan struct{})
	go func() {
		bp.wg.Wait()
		close(done)
	}()
	
	select {
	case <-done:
		bp.logger.Info("Batch processor shutdown complete")
	case <-ctx.Done():
		bp.logger.Warn("Shutdown timeout, some requests may be lost")
	}
	
	bp.workerPool.Release()
	return nil
}

// ════════════════════════════════════════════════════════════════════════════════
// INTERNAL METHODS
// ════════════════════════════════════════════════════════════════════════════════

// runBatcher runs the main batching loop.
func (bp *BatchProcessor) runBatcher() {
	defer bp.wg.Done()
	
	bp.batchTimer = time.NewTimer(bp.config.BatchTimeout)
	defer bp.batchTimer.Stop()
	
	for {
		select {
		case <-bp.ctx.Done():
			// Process any remaining requests
			bp.flushPending()
			return
			
		case <-bp.batchTimer.C:
			// Timeout - process current batch even if not full
			bp.flushPending()
			bp.batchTimer.Reset(bp.config.BatchTimeout)
			
		default:
			// Try to collect requests from queues (priority order)
			req := bp.dequeueRequest()
			if req != nil {
				bp.addToPending(req)
			} else {
				// No requests, small sleep to avoid busy loop
				time.Sleep(1 * time.Millisecond)
			}
		}
	}
}

// dequeueRequest gets the next request from queues (priority order).
func (bp *BatchProcessor) dequeueRequest() *InferenceRequest {
	// Check queues in priority order (highest first)
	for i := len(bp.queues) - 1; i >= 0; i-- {
		select {
		case req := <-bp.queues[i]:
			return req
		default:
			continue
		}
	}
	return nil
}

// addToPending adds a request to the pending batch.
func (bp *BatchProcessor) addToPending(req *InferenceRequest) {
	bp.pendingLock.Lock()
	defer bp.pendingLock.Unlock()
	
	bp.pending = append(bp.pending, req)
	
	// Check if batch is full
	if len(bp.pending) >= bp.config.MaxBatchSize {
		bp.processPendingLocked()
	}
}

// flushPending processes any pending requests.
func (bp *BatchProcessor) flushPending() {
	bp.pendingLock.Lock()
	defer bp.pendingLock.Unlock()
	
	if len(bp.pending) > 0 {
		bp.processPendingLocked()
	}
}

// processPendingLocked processes the pending batch (must hold lock).
func (bp *BatchProcessor) processPendingLocked() {
	if len(bp.pending) == 0 {
		return
	}
	
	// Take ownership of pending requests
	requests := bp.pending
	bp.pending = make([]*InferenceRequest, 0, bp.config.MaxBatchSize)
	
	// Update stats
	atomic.AddInt64(&bp.stats.QueueDepth, -int64(len(requests)))
	bp.updateBatchSizeHistogram(len(requests))
	
	// Submit batch processing to worker pool
	err := bp.workerPool.Submit(func() {
		bp.processBatch(requests)
	})
	
	if err != nil {
		// Worker pool full - process in current goroutine
		bp.logger.Warn("Worker pool full, processing in batcher goroutine")
		go bp.processBatch(requests)
	}
}

// processBatch processes a batch of requests.
func (bp *BatchProcessor) processBatch(requests []*InferenceRequest) {
	ctx, span := bp.tracer.Start(bp.ctx, "ProcessBatch",
		trace.WithAttributes(
			attribute.Int("batch_size", len(requests)),
		),
	)
	defer span.End()
	
	atomic.AddInt64(&bp.stats.ActiveWorkers, 1)
	defer atomic.AddInt64(&bp.stats.ActiveWorkers, -1)
	
	batch := NewBatch(requests)
	startTime := time.Now()
	
	// Create timeout context
	processCtx, cancel := context.WithTimeout(ctx, bp.config.ProcessTimeout)
	defer cancel()
	
	// Call the processor function
	responses, err := bp.processor(processCtx, batch)
	
	elapsed := time.Since(startTime)
	
	// Handle responses
	if err != nil {
		bp.logger.Error("Batch processing failed",
			zap.String("batch_id", batch.ID),
			zap.Int("size", batch.Size()),
			zap.Error(err),
		)
		
		// Send error responses to all requests
		for _, req := range requests {
			atomic.AddUint64(&bp.stats.RequestsFailed, 1)
			req.responseCh <- &InferenceResponse{
				RequestID:    req.ID,
				Error:        err,
				FinishReason: "error",
				LatencyMs:    float64(elapsed.Milliseconds()),
				CompletedAt:  time.Now(),
			}
		}
		return
	}
	
	// Match responses to requests
	responseMap := make(map[string]*InferenceResponse)
	for _, resp := range responses {
		responseMap[resp.RequestID] = resp
	}
	
	for _, req := range requests {
		resp, ok := responseMap[req.ID]
		if !ok {
			// Missing response - create error response
			resp = &InferenceResponse{
				RequestID:    req.ID,
				Error:        fmt.Errorf("no response from processor"),
				FinishReason: "error",
				CompletedAt:  time.Now(),
			}
			atomic.AddUint64(&bp.stats.RequestsFailed, 1)
		} else {
			atomic.AddUint64(&bp.stats.RequestsProcessed, 1)
			atomic.AddUint64(&bp.stats.TotalTokens, uint64(resp.TokensUsed))
		}
		
		resp.LatencyMs = float64(time.Since(req.CreatedAt).Milliseconds())
		atomic.AddUint64(&bp.stats.TotalLatencyNs, uint64(time.Since(req.CreatedAt).Nanoseconds()))
		
		req.responseCh <- resp
	}
	
	atomic.AddUint64(&bp.stats.BatchesProcessed, 1)
	
	bp.logger.Debug("Batch processed",
		zap.String("batch_id", batch.ID),
		zap.Int("size", batch.Size()),
		zap.Duration("elapsed", elapsed),
	)
}

// updateBatchSizeHistogram updates the batch size histogram.
func (bp *BatchProcessor) updateBatchSizeHistogram(size int) {
	var bucket int
	switch {
	case size == 1:
		bucket = 0
	case size <= 4:
		bucket = 1
	case size <= 8:
		bucket = 2
	case size <= 16:
		bucket = 3
	case size <= 32:
		bucket = 4
	default:
		bucket = 5
	}
	atomic.AddUint64(&bp.stats.BatchSizeHistogram[bucket], 1)
}

// ════════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════════

// ChunkRequests splits requests into chunks of the given size.
func ChunkRequests(requests []*InferenceRequest, chunkSize int) [][]*InferenceRequest {
	return lo.Chunk(requests, chunkSize)
}

// FilterByPriority filters requests by minimum priority.
func FilterByPriority(requests []*InferenceRequest, minPriority Priority) []*InferenceRequest {
	return lo.Filter(requests, func(req *InferenceRequest, _ int) bool {
		return req.Priority >= minPriority
	})
}

// SortByPriority sorts requests by priority (highest first).
func SortByPriority(requests []*InferenceRequest) []*InferenceRequest {
	result := make([]*InferenceRequest, len(requests))
	copy(result, requests)
	
	// Sort descending by priority
	for i := 0; i < len(result)-1; i++ {
		for j := i + 1; j < len(result); j++ {
			if result[j].Priority > result[i].Priority {
				result[i], result[j] = result[j], result[i]
			}
		}
	}
	
	return result
}
