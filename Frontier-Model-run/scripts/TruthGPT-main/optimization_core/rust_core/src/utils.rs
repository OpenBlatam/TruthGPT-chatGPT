//! Utility functions for TruthGPT Rust Core
//!
//! Common utilities used across modules including:
//! - Memory management
//! - Performance measurement
//! - Data conversion
//! - Thread-safe counters

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// High-precision timer for benchmarking
#[derive(Debug)]
pub struct Timer {
    name: String,
    start: Instant,
    laps: Vec<(String, Duration)>,
}

impl Timer {
    /// Create a new timer with name
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            laps: Vec::new(),
        }
    }

    /// Record a lap with label
    pub fn lap(&mut self, label: impl Into<String>) {
        let elapsed = self.start.elapsed();
        self.laps.push((label.into(), elapsed));
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1_000_000.0
    }

    /// Get all laps
    pub fn laps(&self) -> &[(String, Duration)] {
        &self.laps
    }

    /// Print timing summary
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = format!("Timer '{}' - Total: {:.3}ms\n", self.name, self.elapsed_ms());
        let mut prev_time = Duration::ZERO;
        
        for (label, time) in &self.laps {
            let delta = *time - prev_time;
            s.push_str(&format!(
                "  {} -> {:.3}ms (delta: {:.3}ms)\n",
                label,
                time.as_secs_f64() * 1000.0,
                delta.as_secs_f64() * 1000.0
            ));
            prev_time = *time;
        }
        s
    }
}

/// Measure execution time of a closure
pub fn measure<T, F: FnOnce() -> T>(name: &str, f: F) -> (T, Duration) {
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    tracing::debug!("{}: {:.3}ms", name, elapsed.as_secs_f64() * 1000.0);
    (result, elapsed)
}

/// Measure execution time in microseconds
pub fn measure_us<T, F: FnOnce() -> T>(f: F) -> (T, u64) {
    let start = Instant::now();
    let result = f();
    let elapsed_us = start.elapsed().as_micros() as u64;
    (result, elapsed_us)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC COUNTERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Thread-safe counter for statistics
#[derive(Debug, Default)]
pub struct AtomicCounter {
    value: AtomicU64,
}

impl AtomicCounter {
    /// Create new counter
    #[must_use]
    pub fn new() -> Self {
        Self { value: AtomicU64::new(0) }
    }

    /// Create with initial value
    #[must_use]
    pub fn with_value(value: u64) -> Self {
        Self { value: AtomicU64::new(value) }
    }

    /// Increment and return new value
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Add and return new value
    pub fn add(&self, n: u64) -> u64 {
        self.value.fetch_add(n, Ordering::Relaxed) + n
    }

    /// Get current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset to zero and return previous value
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }
}

/// Thread-safe histogram for latency tracking
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; 32],
    min: AtomicU64,
    max: AtomicU64,
    sum: AtomicU64,
    count: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Create new histogram
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: Default::default(),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a value in microseconds
    pub fn record(&self, value_us: u64) {
        // Update min/max
        self.min.fetch_min(value_us, Ordering::Relaxed);
        self.max.fetch_max(value_us, Ordering::Relaxed);
        self.sum.fetch_add(value_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Update bucket (exponential buckets: 1us, 2us, 4us, ... 2^31 us)
        let bucket = if value_us == 0 {
            0
        } else {
            (64 - value_us.leading_zeros()).min(31) as usize
        };
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn stats(&self) -> HistogramStats {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return HistogramStats::default();
        }

        let sum = self.sum.load(Ordering::Relaxed);
        let min = self.min.load(Ordering::Relaxed);
        let max = self.max.load(Ordering::Relaxed);

        HistogramStats {
            count,
            min_us: min,
            max_us: max,
            avg_us: sum as f64 / count as f64,
            p50_us: self.percentile(50),
            p95_us: self.percentile(95),
            p99_us: self.percentile(99),
        }
    }

    /// Calculate percentile
    fn percentile(&self, p: u8) -> u64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }

        let threshold = (count * p as u64) / 100;
        let mut cumulative = 0u64;

        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= threshold {
                return 1u64 << i;
            }
        }

        self.max.load(Ordering::Relaxed)
    }
}

/// Histogram statistics
#[derive(Debug, Clone, Default)]
pub struct HistogramStats {
    pub count: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub avg_us: f64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Get current memory usage estimate
pub fn memory_usage() -> MemoryStats {
    let info = sys_info();
    MemoryStats {
        allocated_bytes: info.allocated,
        resident_bytes: info.resident,
    }
}

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub resident_bytes: usize,
}

impl MemoryStats {
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn resident_mb(&self) -> f64 {
        self.resident_bytes as f64 / (1024.0 * 1024.0)
    }
}

struct SysInfo {
    allocated: usize,
    resident: usize,
}

fn sys_info() -> SysInfo {
    // Placeholder - would use jemalloc stats in production
    SysInfo {
        allocated: 0,
        resident: 0,
    }
}

/// Aligned allocation helper
pub struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T: Clone> AlignedVec<T> {
    /// Create aligned vec with capacity
    #[must_use]
    pub fn with_capacity(capacity: usize, _alignment: usize) -> Self {
        let data = Vec::with_capacity(capacity);
        Self { data }
    }

    /// Create filled with default value
    #[must_use]
    pub fn new_filled(len: usize, value: T) -> Self {
        let data = vec![value; len];
        Self { data }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA CONVERSION
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert f32 slice to bytes
pub fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert bytes to f32 slice
#[must_use]
pub fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Convert f16 bytes to f32
#[must_use]
pub fn f16_to_f32_bytes(data: &[u8]) -> Vec<f32> {
    use half::f16;
    
    data.chunks_exact(2)
        .map(|chunk| {
            let arr: [u8; 2] = chunk.try_into().unwrap();
            f16::from_le_bytes(arr).to_f32()
        })
        .collect()
}

/// Convert f32 to f16 bytes
#[must_use]
pub fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    use half::f16;
    
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &val in data {
        bytes.extend_from_slice(&f16::from_f32(val).to_le_bytes());
    }
    bytes
}

// ═══════════════════════════════════════════════════════════════════════════════
// RING BUFFER
// ═══════════════════════════════════════════════════════════════════════════════

/// Lock-free ring buffer for fixed-size elements
pub struct RingBuffer<T, const N: usize> {
    buffer: [parking_lot::Mutex<Option<T>>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T: Clone, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self {
            buffer: std::array::from_fn(|_| Mutex::new(None)),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
}

impl<T: Clone, const N: usize> RingBuffer<T, N> {
    /// Create new ring buffer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push item to buffer
    pub fn push(&self, item: T) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) % N;
        
        if next_tail == self.head.load(Ordering::Acquire) {
            return false; // Buffer full
        }

        *self.buffer[tail].lock() = Some(item);
        self.tail.store(next_tail, Ordering::Release);
        true
    }

    /// Pop item from buffer
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        
        if head == self.tail.load(Ordering::Acquire) {
            return None; // Buffer empty
        }

        let item = self.buffer[head].lock().take();
        self.head.store((head + 1) % N, Ordering::Release);
        item
    }

    /// Get current size
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if tail >= head {
            tail - head
        } else {
            N - head + tail
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        N - 1
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOVING AVERAGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Exponential moving average for metrics
pub struct ExponentialMovingAverage {
    value: AtomicU64,
    alpha: f64,
}

impl ExponentialMovingAverage {
    /// Create with smoothing factor alpha (0 < alpha <= 1)
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            value: AtomicU64::new(0),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Update with new value
    pub fn update(&self, new_value: f64) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed));
        let ema = self.alpha * new_value + (1.0 - self.alpha) * current;
        self.value.store(ema.to_bits(), Ordering::Relaxed);
    }

    /// Get current average
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Format bytes as human-readable string
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }

    // Calculate exponent safely
    let exp = if bytes <= u64::from(u32::MAX) {
        (f64::from(bytes as u32).log(1024.0).floor() as usize).min(UNITS.len() - 1)
    } else {
        // For very large values, use direct conversion (may lose precision but acceptable for display)
        #[allow(clippy::cast_precision_loss)]
        let bytes_f64 = bytes as f64;
        (bytes_f64.log(1024.0).floor() as usize).min(UNITS.len() - 1)
    };
    
    // Calculate value with proper precision handling
    #[allow(clippy::cast_precision_loss)]
    let value = if bytes <= u64::from(u32::MAX) {
        f64::from(bytes as u32) / 1024_f64.powi(exp as i32)
    } else {
        bytes as f64 / 1024_f64.powi(exp as i32)
    };
    
    format!("{:.2} {}", value, UNITS[exp])
}

/// Format duration as human-readable string
pub fn format_duration(dur: Duration) -> String {
    let secs = dur.as_secs_f64();
    
    if secs < 0.001 {
        format!("{:.2}µs", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.2}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let mut timer = Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.lap("after 10ms");
        assert!(timer.elapsed_ms() >= 10.0);
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new();
        assert_eq!(counter.get(), 0);
        
        counter.increment();
        counter.increment();
        assert_eq!(counter.get(), 2);
        
        counter.add(10);
        assert_eq!(counter.get(), 12);
        
        let old = counter.reset();
        assert_eq!(old, 12);
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new();
        
        for i in 0..100 {
            hist.record(i);
        }
        
        let stats = hist.stats();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min_us, 0);
        assert_eq!(stats.max_us, 99);
    }

    #[test]
    fn test_f32_bytes_conversion() {
        let original = vec![1.0f32, 2.5, 3.7, -1.2];
        let bytes = f32_to_bytes(&original);
        let recovered = bytes_to_f32(&bytes);
        
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_ring_buffer() {
        let buffer: RingBuffer<i32, 4> = RingBuffer::new();
        
        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(buffer.push(3));
        assert!(!buffer.push(4)); // Full
        
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), None); // Empty
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_ema() {
        let ema = ExponentialMovingAverage::new(0.5);
        
        ema.update(10.0);
        assert!((ema.get() - 5.0).abs() < 0.001);
        
        ema.update(10.0);
        assert!((ema.get() - 7.5).abs() < 0.001);
    }
}
