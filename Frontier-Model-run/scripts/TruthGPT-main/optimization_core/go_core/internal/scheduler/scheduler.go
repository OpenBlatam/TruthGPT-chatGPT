// Package scheduler provides request scheduling for TruthGPT inference.
//
// Features:
// - Multi-level priority queues
// - Fair scheduling with aging
// - Batch formation with dynamic sizing
// - Preemption for high-priority requests
// - Deadline-aware scheduling
package scheduler

import (
	"container/heap"
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ════════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// Config holds scheduler configuration.
type Config struct {
	// Queue settings
	MaxQueueSize      int           `yaml:"max_queue_size"`
	MaxWaitTime       time.Duration `yaml:"max_wait_time"`
	PriorityLevels    int           `yaml:"priority_levels"`
	EnableAging       bool          `yaml:"enable_aging"`
	AgingInterval     time.Duration `yaml:"aging_interval"`
	AgingBoost        int           `yaml:"aging_boost"`

	// Batch settings
	MaxBatchSize      int           `yaml:"max_batch_size"`
	MinBatchSize      int           `yaml:"min_batch_size"`
	BatchWaitTime     time.Duration `yaml:"batch_wait_time"`
	DynamicBatching   bool          `yaml:"dynamic_batching"`

	// Preemption
	EnablePreemption  bool          `yaml:"enable_preemption"`
	PreemptThreshold  int           `yaml:"preempt_threshold"`

	// Workers
	NumWorkers        int           `yaml:"num_workers"`
}

// DefaultConfig returns default scheduler configuration.
func DefaultConfig() Config {
	return Config{
		MaxQueueSize:      10000,
		MaxWaitTime:       30 * time.Second,
		PriorityLevels:    4,
		EnableAging:       true,
		AgingInterval:     time.Second,
		AgingBoost:        1,
		MaxBatchSize:      32,
		MinBatchSize:      1,
		BatchWaitTime:     10 * time.Millisecond,
		DynamicBatching:   true,
		EnablePreemption:  true,
		PreemptThreshold:  3, // Priority level that triggers preemption
		NumWorkers:        4,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// JOB
// ════════════════════════════════════════════════════════════════════════════════

// JobID is a unique job identifier.
type JobID uint64

// JobState represents the state of a job.
type JobState int

const (
	JobStatePending JobState = iota
	JobStateRunning
	JobStateCompleted
	JobStateFailed
	JobStateCancelled
	JobStatePreempted
)

// Job represents a schedulable unit of work.
type Job struct {
	ID           JobID
	Priority     int
	EffPriority  int // Effective priority (with aging)
	Deadline     time.Time
	SubmittedAt  time.Time
	StartedAt    time.Time
	CompletedAt  time.Time
	State        JobState
	
	// Payload
	InputTokens  int
	MaxNewTokens int
	Data         interface{}
	
	// Result channel
	Result       chan *JobResult
	
	// Internal
	index        int // heap index
}

// JobResult represents the result of a job.
type JobResult struct {
	JobID   JobID
	Success bool
	Output  interface{}
	Error   error
	Latency time.Duration
}

// NewJob creates a new job.
func NewJob(priority int, data interface{}) *Job {
	return &Job{
		ID:          nextJobID(),
		Priority:    priority,
		EffPriority: priority,
		SubmittedAt: time.Now(),
		State:       JobStatePending,
		Data:        data,
		Result:      make(chan *JobResult, 1),
	}
}

// WithDeadline sets a deadline for the job.
func (j *Job) WithDeadline(deadline time.Time) *Job {
	j.Deadline = deadline
	return j
}

// WithTokens sets token counts for the job.
func (j *Job) WithTokens(input, maxNew int) *Job {
	j.InputTokens = input
	j.MaxNewTokens = maxNew
	return j
}

// WaitTime returns how long the job has been waiting.
func (j *Job) WaitTime() time.Duration {
	if j.State == JobStatePending {
		return time.Since(j.SubmittedAt)
	}
	return j.StartedAt.Sub(j.SubmittedAt)
}

// IsExpired checks if the job has expired.
func (j *Job) IsExpired() bool {
	if j.Deadline.IsZero() {
		return false
	}
	return time.Now().After(j.Deadline)
}

var jobIDCounter atomic.Uint64

func nextJobID() JobID {
	return JobID(jobIDCounter.Add(1))
}

// ════════════════════════════════════════════════════════════════════════════════
// PRIORITY QUEUE
// ════════════════════════════════════════════════════════════════════════════════

// PriorityQueue implements a heap-based priority queue.
type PriorityQueue []*Job

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// Higher effective priority first
	if pq[i].EffPriority != pq[j].EffPriority {
		return pq[i].EffPriority > pq[j].EffPriority
	}
	// Then earlier deadline
	if !pq[i].Deadline.IsZero() && !pq[j].Deadline.IsZero() {
		return pq[i].Deadline.Before(pq[j].Deadline)
	}
	// Then earlier submission
	return pq[i].SubmittedAt.Before(pq[j].SubmittedAt)
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	job := x.(*Job)
	job.index = n
	*pq = append(*pq, job)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	job := old[n-1]
	old[n-1] = nil
	job.index = -1
	*pq = old[0 : n-1]
	return job
}

// ════════════════════════════════════════════════════════════════════════════════
// BATCH
// ════════════════════════════════════════════════════════════════════════════════

// Batch represents a batch of jobs to process together.
type Batch struct {
	ID        uint64
	Jobs      []*Job
	CreatedAt time.Time
	Size      int
	TokensSum int
}

// NewBatch creates a new batch.
func NewBatch() *Batch {
	return &Batch{
		ID:        nextBatchID(),
		Jobs:      make([]*Job, 0, 32),
		CreatedAt: time.Now(),
	}
}

// Add adds a job to the batch.
func (b *Batch) Add(job *Job) {
	b.Jobs = append(b.Jobs, job)
	b.Size++
	b.TokensSum += job.InputTokens
}

var batchIDCounter atomic.Uint64

func nextBatchID() uint64 {
	return batchIDCounter.Add(1)
}

// ════════════════════════════════════════════════════════════════════════════════
// SCHEDULER
// ════════════════════════════════════════════════════════════════════════════════

// Scheduler manages job scheduling and batch formation.
type Scheduler struct {
	config Config
	logger *zap.Logger
	
	// Queues
	queues  []*PriorityQueue
	queueMu sync.Mutex
	
	// Running jobs
	running sync.Map // map[JobID]*Job
	
	// Batch channel
	batches chan *Batch
	
	// Stats
	stats   SchedulerStats
	
	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// SchedulerStats holds scheduler statistics.
type SchedulerStats struct {
	JobsSubmitted   atomic.Uint64
	JobsCompleted   atomic.Uint64
	JobsFailed      atomic.Uint64
	JobsCancelled   atomic.Uint64
	JobsPreempted   atomic.Uint64
	BatchesFormed   atomic.Uint64
	TotalWaitTimeUs atomic.Uint64
	TotalRunTimeUs  atomic.Uint64
}

// New creates a new scheduler.
func New(config Config, logger *zap.Logger) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	
	s := &Scheduler{
		config:  config,
		logger:  logger,
		queues:  make([]*PriorityQueue, config.PriorityLevels),
		batches: make(chan *Batch, 100),
		ctx:     ctx,
		cancel:  cancel,
	}
	
	// Initialize priority queues
	for i := range s.queues {
		pq := make(PriorityQueue, 0, config.MaxQueueSize/config.PriorityLevels)
		heap.Init(&pq)
		s.queues[i] = &pq
	}
	
	// Start workers
	for i := 0; i < config.NumWorkers; i++ {
		s.wg.Add(1)
		go s.batchWorker(i)
	}
	
	// Start aging if enabled
	if config.EnableAging {
		s.wg.Add(1)
		go s.agingWorker()
	}
	
	return s
}

// Submit submits a job to the scheduler.
func (s *Scheduler) Submit(job *Job) error {
	if job.Priority < 0 || job.Priority >= s.config.PriorityLevels {
		return fmt.Errorf("invalid priority %d", job.Priority)
	}
	
	s.queueMu.Lock()
	defer s.queueMu.Unlock()
	
	// Check queue size
	totalSize := 0
	for _, q := range s.queues {
		totalSize += q.Len()
	}
	if totalSize >= s.config.MaxQueueSize {
		return fmt.Errorf("queue full")
	}
	
	// Add to appropriate priority queue
	heap.Push(s.queues[job.Priority], job)
	s.stats.JobsSubmitted.Add(1)
	
	// Check for preemption
	if s.config.EnablePreemption && job.Priority >= s.config.PreemptThreshold {
		s.checkPreemption(job.Priority)
	}
	
	return nil
}

// Cancel cancels a pending job.
func (s *Scheduler) Cancel(jobID JobID) bool {
	s.queueMu.Lock()
	defer s.queueMu.Unlock()
	
	for _, q := range s.queues {
		for i, job := range *q {
			if job.ID == jobID {
				job.State = JobStateCancelled
				heap.Remove(q, i)
				s.stats.JobsCancelled.Add(1)
				close(job.Result)
				return true
			}
		}
	}
	
	return false
}

// NextBatch returns the next batch of jobs.
func (s *Scheduler) NextBatch(ctx context.Context) (*Batch, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case batch := <-s.batches:
		return batch, nil
	}
}

// Complete marks a job as completed.
func (s *Scheduler) Complete(jobID JobID, result *JobResult) {
	if v, ok := s.running.LoadAndDelete(jobID); ok {
		job := v.(*Job)
		job.State = JobStateCompleted
		job.CompletedAt = time.Now()
		result.Latency = job.CompletedAt.Sub(job.StartedAt)
		
		s.stats.JobsCompleted.Add(1)
		s.stats.TotalRunTimeUs.Add(uint64(result.Latency.Microseconds()))
		
		select {
		case job.Result <- result:
		default:
		}
	}
}

// Fail marks a job as failed.
func (s *Scheduler) Fail(jobID JobID, err error) {
	if v, ok := s.running.LoadAndDelete(jobID); ok {
		job := v.(*Job)
		job.State = JobStateFailed
		job.CompletedAt = time.Now()
		
		s.stats.JobsFailed.Add(1)
		
		select {
		case job.Result <- &JobResult{JobID: jobID, Error: err}:
		default:
		}
	}
}

func (s *Scheduler) batchWorker(id int) {
	defer s.wg.Done()
	
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
			batch := s.formBatch()
			if batch != nil && batch.Size > 0 {
				s.batches <- batch
			} else {
				time.Sleep(s.config.BatchWaitTime)
			}
		}
	}
}

func (s *Scheduler) formBatch() *Batch {
	s.queueMu.Lock()
	defer s.queueMu.Unlock()
	
	batch := NewBatch()
	now := time.Now()
	
	// Try to fill batch from highest to lowest priority
	for p := s.config.PriorityLevels - 1; p >= 0; p-- {
		q := s.queues[p]
		
		for q.Len() > 0 && batch.Size < s.config.MaxBatchSize {
			job := (*q)[0]
			
			// Skip expired jobs
			if job.IsExpired() {
				heap.Pop(q)
				job.State = JobStateCancelled
				s.stats.JobsCancelled.Add(1)
				continue
			}
			
			// Check if we should wait for more jobs
			if s.config.DynamicBatching && 
			   batch.Size < s.config.MinBatchSize &&
			   job.WaitTime() < s.config.BatchWaitTime {
				continue
			}
			
			// Add to batch
			job = heap.Pop(q).(*Job)
			job.State = JobStateRunning
			job.StartedAt = now
			
			s.stats.TotalWaitTimeUs.Add(uint64(job.WaitTime().Microseconds()))
			s.running.Store(job.ID, job)
			
			batch.Add(job)
		}
	}
	
	if batch.Size > 0 {
		s.stats.BatchesFormed.Add(1)
	}
	
	return batch
}

func (s *Scheduler) checkPreemption(priority int) {
	// Preempt lower priority running jobs if needed
	if !s.config.EnablePreemption {
		return
	}
	
	// Find running jobs with lower priority
	var toPreempt []*Job
	s.running.Range(func(key, value interface{}) bool {
		job := value.(*Job)
		if job.Priority < priority-1 {
			toPreempt = append(toPreempt, job)
		}
		return true
	})
	
	// Preempt and requeue
	for _, job := range toPreempt {
		if s.running.CompareAndDelete(job.ID, job) {
			job.State = JobStatePreempted
			s.stats.JobsPreempted.Add(1)
			
			// Requeue with boosted priority
			job.EffPriority = min(job.EffPriority+1, s.config.PriorityLevels-1)
			heap.Push(s.queues[job.Priority], job)
		}
	}
}

func (s *Scheduler) agingWorker() {
	defer s.wg.Done()
	
	ticker := time.NewTicker(s.config.AgingInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.applyAging()
		}
	}
}

func (s *Scheduler) applyAging() {
	s.queueMu.Lock()
	defer s.queueMu.Unlock()
	
	for _, q := range s.queues {
		for _, job := range *q {
			// Boost effective priority for old jobs
			if job.WaitTime() > s.config.MaxWaitTime/2 {
				oldPriority := job.EffPriority
				job.EffPriority = min(job.EffPriority+s.config.AgingBoost, s.config.PriorityLevels-1)
				if job.EffPriority != oldPriority {
					heap.Fix(q, job.index)
				}
			}
		}
	}
}

// Stats returns scheduler statistics.
func (s *Scheduler) Stats() map[string]interface{} {
	s.queueMu.Lock()
	queueSizes := make([]int, len(s.queues))
	for i, q := range s.queues {
		queueSizes[i] = q.Len()
	}
	s.queueMu.Unlock()
	
	runningCount := 0
	s.running.Range(func(_, _ interface{}) bool {
		runningCount++
		return true
	})
	
	avgWait := float64(0)
	if s.stats.JobsCompleted.Load() > 0 {
		avgWait = float64(s.stats.TotalWaitTimeUs.Load()) / float64(s.stats.JobsCompleted.Load()) / 1000
	}
	
	avgRun := float64(0)
	if s.stats.JobsCompleted.Load() > 0 {
		avgRun = float64(s.stats.TotalRunTimeUs.Load()) / float64(s.stats.JobsCompleted.Load()) / 1000
	}
	
	return map[string]interface{}{
		"jobs_submitted": s.stats.JobsSubmitted.Load(),
		"jobs_completed": s.stats.JobsCompleted.Load(),
		"jobs_failed":    s.stats.JobsFailed.Load(),
		"jobs_cancelled": s.stats.JobsCancelled.Load(),
		"jobs_preempted": s.stats.JobsPreempted.Load(),
		"batches_formed": s.stats.BatchesFormed.Load(),
		"queue_sizes":    queueSizes,
		"running_count":  runningCount,
		"avg_wait_ms":    avgWait,
		"avg_run_ms":     avgRun,
	}
}

// Close shuts down the scheduler.
func (s *Scheduler) Close() {
	s.cancel()
	s.wg.Wait()
	close(s.batches)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}




