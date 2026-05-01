
//! Batch Inference Module for TruthGPT
//!
//! High-throughput batch inference with dynamic batching and continuous batching.
//!
//! ## Features
//!
//! - Dynamic batching: auto-batch requests for efficiency
//! - Continuous batching: process new requests without waiting
//! - Memory-efficient request queuing
//! - Priority scheduling

use parking_lot::{Mutex, RwLock};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};

// ═══════════════════════════════════════════════════════════════════════════════
// REQUEST TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique request identifier
pub type RequestId = u64;

/// Inference request priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: RequestId,
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Maximum tokens to generate
    pub max_new_tokens: usize,
    /// Request priority
    pub priority: Priority,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Stop sequences
    pub stop_sequences: Vec<Vec<u32>>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Optional deadline
    pub deadline: Option<Instant>,
}

impl InferenceRequest {
    /// Create new request
    pub fn new(input_ids: Vec<u32>) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            input_ids,
            max_new_tokens: 256,
            priority: Priority::Normal,
            temperature: 1.0,
            top_p: 0.9,
            stop_sequences: Vec::new(),
            created_at: Instant::now(),
            deadline: None,
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_new_tokens = max;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Get input length
    pub fn input_len(&self) -> usize {
        self.input_ids.len()
    }

    /// Get wait time
    pub fn wait_time(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if deadline passed
    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| Instant::now() > d).unwrap_or(false)
    }
}

/// Inference response
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Request ID
    pub id: RequestId,
    /// Generated token IDs
    pub output_ids: Vec<u32>,
    /// Generation finished
    pub is_finished: bool,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Time to first token
    pub time_to_first_token_ms: f64,
    /// Total generation time
    pub total_time_ms: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
}

/// Finish reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Max tokens reached
    MaxTokens,
    /// Stop sequence found
    StopSequence,
    /// EOS token generated
    EndOfSequence,
    /// Request cancelled
    Cancelled,
    /// Request expired
    Expired,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Batch inference configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length in batch
    pub max_sequence_length: usize,
    /// Maximum waiting time before forming batch
    pub max_wait_time_ms: u64,
    /// Minimum batch utilization (0.0 - 1.0)
    pub min_batch_utilization: f32,
    /// Enable continuous batching
    pub continuous_batching: bool,
    /// Enable priority scheduling
    pub priority_scheduling: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 2048,
            max_wait_time_ms: 10,
            min_batch_utilization: 0.5,
            continuous_batching: true,
            priority_scheduling: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH SCHEDULER
// ═══════════════════════════════════════════════════════════════════════════════

/// Active request in the scheduler
struct ActiveRequest {
    request: InferenceRequest,
    generated_tokens: Vec<u32>,
    kv_cache_slot: Option<usize>,
    start_time: Instant,
    first_token_time: Option<Instant>,
}

/// Batch scheduler for inference requests
pub struct BatchScheduler {
    /// Configuration
    config: BatchConfig,
    /// Pending requests queue (priority queue)
    pending: Mutex<BinaryHeap<PriorityRequest>>,
    /// Active requests being processed
    active: RwLock<HashMap<RequestId, ActiveRequest>>,
    /// Completed responses channel
    response_tx: Sender<InferenceResponse>,
    response_rx: Receiver<InferenceResponse>,
    /// Statistics
    stats: SchedulerStats,
}

/// Wrapper for priority queue ordering
#[derive(Debug)]
struct PriorityRequest {
    priority: Priority,
    created_at: Instant,
    request: InferenceRequest,
}

impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.created_at == other.created_at
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then older requests first
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.created_at.cmp(&self.created_at),
            ord => ord,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub requests_submitted: AtomicU64,
    pub requests_completed: AtomicU64,
    pub requests_cancelled: AtomicU64,
    pub batches_formed: AtomicU64,
    pub total_tokens_generated: AtomicU64,
    pub avg_batch_size: AtomicU64,
    pub avg_wait_time_us: AtomicU64,
}

impl BatchScheduler {
    /// Create new scheduler
    pub fn new(config: BatchConfig) -> Self {
        let (tx, rx) = bounded(1024);
        
        Self {
            config,
            pending: Mutex::new(BinaryHeap::new()),
            active: RwLock::new(HashMap::new()),
            response_tx: tx,
            response_rx: rx,
            stats: SchedulerStats::default(),
        }
    }

    /// Submit a request
    pub fn submit(&self, request: InferenceRequest) -> RequestId {
        let id = request.id;
        
        self.stats.requests_submitted.fetch_add(1, Ordering::Relaxed);
        
        let priority_req = PriorityRequest {
            priority: request.priority,
            created_at: request.created_at,
            request,
        };
        
        self.pending.lock().push(priority_req);
        id
    }

    /// Get next batch for processing
    pub fn get_batch(&self) -> Option<Vec<InferenceRequest>> {
        let mut pending = self.pending.lock();
        
        if pending.is_empty() {
            return None;
        }

        let mut batch = Vec::with_capacity(self.config.max_batch_size);
        let mut total_length = 0;
        
        // Check oldest request wait time
        if let Some(oldest) = pending.peek() {
            let wait_time = oldest.request.wait_time().as_millis() as u64;
            
            // If not enough wait time and not enough requests, wait
            if wait_time < self.config.max_wait_time_ms 
                && pending.len() < (self.config.max_batch_size as f32 * self.config.min_batch_utilization) as usize
            {
                return None;
            }
        }

        // Form batch
        while let Some(pr) = pending.peek() {
            // Check if request fits in batch
            if batch.len() >= self.config.max_batch_size {
                break;
            }
            
            if total_length + pr.request.input_len() > self.config.max_sequence_length * self.config.max_batch_size {
                break;
            }

            // Skip expired requests
            if pr.request.is_expired() {
                pending.pop();
                self.stats.requests_cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            if let Some(pr) = pending.pop() {
                total_length += pr.request.input_len();
                batch.push(pr.request);
            }
        }

        if batch.is_empty() {
            None
        } else {
            self.stats.batches_formed.fetch_add(1, Ordering::Relaxed);
            
            // Update average batch size (simplified)
            let current_avg = self.stats.avg_batch_size.load(Ordering::Relaxed);
            let new_avg = (current_avg + batch.len() as u64) / 2;
            self.stats.avg_batch_size.store(new_avg, Ordering::Relaxed);
            
            Some(batch)
        }
    }

    /// Start processing a batch
    pub fn start_batch(&self, requests: &[InferenceRequest]) {
        let mut active = self.active.write();
        
        for req in requests {
            let active_req = ActiveRequest {
                request: req.clone(),
                generated_tokens: Vec::with_capacity(req.max_new_tokens),
                kv_cache_slot: None,
                start_time: Instant::now(),
                first_token_time: None,
            };
            
            active.insert(req.id, active_req);
        }
    }

    /// Add generated token for a request
    pub fn add_token(&self, id: RequestId, token: u32) -> bool {
        let mut active = self.active.write();
        
        if let Some(req) = active.get_mut(&id) {
            if req.first_token_time.is_none() {
                req.first_token_time = Some(Instant::now());
            }
            
            req.generated_tokens.push(token);
            self.stats.total_tokens_generated.fetch_add(1, Ordering::Relaxed);
            
            // Check completion conditions
            let is_done = req.generated_tokens.len() >= req.request.max_new_tokens;
            
            if is_done {
                self.complete_request(id, FinishReason::MaxTokens);
            }
            
            is_done
        } else {
            true // Request not found, treat as done
        }
    }

    /// Complete a request
    pub fn complete_request(&self, id: RequestId, reason: FinishReason) {
        let mut active = self.active.write();
        
        if let Some(req) = active.remove(&id) {
            let total_time = req.start_time.elapsed();
            let time_to_first = req.first_token_time
                .map(|t| t.duration_since(req.start_time))
                .unwrap_or(total_time);
            
            let tokens_generated = req.generated_tokens.len();
            let tokens_per_second = if total_time.as_secs_f64() > 0.0 {
                tokens_generated as f64 / total_time.as_secs_f64()
            } else {
                0.0
            };
            
            let response = InferenceResponse {
                id,
                output_ids: req.generated_tokens,
                is_finished: true,
                finish_reason: reason,
                time_to_first_token_ms: time_to_first.as_secs_f64() * 1000.0,
                total_time_ms: total_time.as_secs_f64() * 1000.0,
                tokens_per_second,
            };
            
            let _ = self.response_tx.send(response);
            self.stats.requests_completed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Cancel a request
    pub fn cancel(&self, id: RequestId) -> bool {
        // Check pending
        {
            let mut pending = self.pending.lock();
            let len_before = pending.len();
            
            // Rebuild heap without the cancelled request
            let remaining: Vec<_> = std::mem::take(&mut *pending)
                .into_iter()
                .filter(|pr| pr.request.id != id)
                .collect();
            
            *pending = remaining.into_iter().collect();
            
            if pending.len() < len_before {
                self.stats.requests_cancelled.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        
        // Check active
        {
            let active = self.active.read();
            if active.contains_key(&id) {
                drop(active);
                self.complete_request(id, FinishReason::Cancelled);
                self.stats.requests_cancelled.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        
        false
    }

    /// Get response receiver
    pub fn responses(&self) -> &Receiver<InferenceResponse> {
        &self.response_rx
    }

    /// Get statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Get pending count
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Get active count
    pub fn active_count(&self) -> usize {
        self.active.read().len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTINUOUS BATCHING
// ═══════════════════════════════════════════════════════════════════════════════

/// Continuous batch manager
pub struct ContinuousBatcher {
    /// Batch slots
    slots: Vec<Option<ActiveSlot>>,
    /// Free slot indices
    free_slots: VecDeque<usize>,
    /// Maximum slots
    max_slots: usize,
}

#[derive(Clone)]
struct ActiveSlot {
    request_id: RequestId,
    position: usize,
    tokens: Vec<u32>,
}

impl ContinuousBatcher {
    /// Create new continuous batcher
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: vec![None; max_slots],
            free_slots: (0..max_slots).collect(),
            max_slots,
        }
    }

    /// Try to add a request
    pub fn add(&mut self, request_id: RequestId, input_ids: &[u32]) -> Option<usize> {
        if let Some(slot_idx) = self.free_slots.pop_front() {
            self.slots[slot_idx] = Some(ActiveSlot {
                request_id,
                position: input_ids.len(),
                tokens: input_ids.to_vec(),
            });
            Some(slot_idx)
        } else {
            None
        }
    }

    /// Remove a request
    pub fn remove(&mut self, slot_idx: usize) {
        if slot_idx < self.max_slots {
            self.slots[slot_idx] = None;
            self.free_slots.push_back(slot_idx);
        }
    }

    /// Get active request IDs
    pub fn active_requests(&self) -> Vec<RequestId> {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref().map(|s| s.request_id))
            .collect()
    }

    /// Get number of free slots
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Get number of active slots
    pub fn active_count(&self) -> usize {
        self.max_slots - self.free_slots.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = InferenceRequest::new(vec![1, 2, 3, 4, 5])
            .with_max_tokens(100)
            .with_priority(Priority::High)
            .with_temperature(0.8);
        
        assert_eq!(req.input_len(), 5);
        assert_eq!(req.max_new_tokens, 100);
        assert_eq!(req.priority, Priority::High);
        assert!((req.temperature - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_scheduler_submit() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        
        let id1 = scheduler.submit(InferenceRequest::new(vec![1, 2, 3]));
        let id2 = scheduler.submit(InferenceRequest::new(vec![4, 5, 6]));
        
        assert_ne!(id1, id2);
        assert_eq!(scheduler.pending_count(), 2);
    }

    #[test]
    fn test_priority_ordering() {
        let scheduler = BatchScheduler::new(BatchConfig {
            max_wait_time_ms: 0, // Immediate batch formation
            ..Default::default()
        });
        
        scheduler.submit(InferenceRequest::new(vec![1]).with_priority(Priority::Low));
        scheduler.submit(InferenceRequest::new(vec![2]).with_priority(Priority::High));
        scheduler.submit(InferenceRequest::new(vec![3]).with_priority(Priority::Normal));
        
        let batch = scheduler.get_batch().unwrap();
        
        // High priority should be first
        assert_eq!(batch[0].priority, Priority::High);
    }

    #[test]
    fn test_continuous_batcher() {
        let mut batcher = ContinuousBatcher::new(4);
        
        let slot1 = batcher.add(1, &[1, 2, 3]).unwrap();
        let slot2 = batcher.add(2, &[4, 5]).unwrap();
        
        assert_eq!(batcher.active_count(), 2);
        assert_eq!(batcher.free_count(), 2);
        
        batcher.remove(slot1);
        
        assert_eq!(batcher.active_count(), 1);
        assert_eq!(batcher.free_count(), 3);
    }
}

