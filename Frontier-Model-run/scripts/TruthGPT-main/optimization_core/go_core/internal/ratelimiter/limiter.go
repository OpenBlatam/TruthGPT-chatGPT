// Package ratelimiter provides distributed rate limiting for TruthGPT.
//
// Features:
// - Token bucket algorithm
// - Sliding window rate limiting
// - Per-user and per-API rate limits
// - Redis-backed distributed limiting
// - Local caching for performance
package ratelimiter

import (
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

// Algorithm represents the rate limiting algorithm.
type Algorithm int

const (
	AlgorithmTokenBucket Algorithm = iota
	AlgorithmSlidingWindow
	AlgorithmLeakyBucket
	AlgorithmFixedWindow
)

// Config holds rate limiter configuration.
type Config struct {
	// Default limits
	DefaultRequestsPerSecond int `yaml:"default_requests_per_second"`
	DefaultBurstSize         int `yaml:"default_burst_size"`

	// Token bucket refill
	RefillRate     float64       `yaml:"refill_rate"`
	RefillInterval time.Duration `yaml:"refill_interval"`

	// Sliding window
	WindowSize time.Duration `yaml:"window_size"`

	// Storage
	UseRedis       bool   `yaml:"use_redis"`
	RedisAddr      string `yaml:"redis_addr"`
	RedisKeyPrefix string `yaml:"redis_key_prefix"`

	// Local cache
	LocalCacheSize int           `yaml:"local_cache_size"`
	LocalCacheTTL  time.Duration `yaml:"local_cache_ttl"`
}

// DefaultConfig returns the default rate limiter configuration.
func DefaultConfig() Config {
	return Config{
		DefaultRequestsPerSecond: 100,
		DefaultBurstSize:         200,
		RefillRate:               100.0,
		RefillInterval:           time.Second,
		WindowSize:               time.Minute,
		UseRedis:                 false,
		RedisKeyPrefix:           "ratelimit:",
		LocalCacheSize:           10000,
		LocalCacheTTL:            time.Second,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// LIMIT DEFINITION
// ════════════════════════════════════════════════════════════════════════════════

// Limit defines rate limit parameters.
type Limit struct {
	RequestsPerSecond int           `json:"requests_per_second"`
	BurstSize         int           `json:"burst_size"`
	WindowSize        time.Duration `json:"window_size,omitempty"`
}

// Predefined limits
var (
	LimitFree = Limit{
		RequestsPerSecond: 10,
		BurstSize:         20,
	}
	LimitStandard = Limit{
		RequestsPerSecond: 100,
		BurstSize:         200,
	}
	LimitPremium = Limit{
		RequestsPerSecond: 1000,
		BurstSize:         2000,
	}
	LimitUnlimited = Limit{
		RequestsPerSecond: 0, // 0 = unlimited
		BurstSize:         0,
	}
)

// ════════════════════════════════════════════════════════════════════════════════
// RESULT
// ════════════════════════════════════════════════════════════════════════════════

// Result represents the result of a rate limit check.
type Result struct {
	Allowed    bool
	Remaining  int
	Limit      int
	ResetAt    time.Time
	RetryAfter time.Duration
}

// ════════════════════════════════════════════════════════════════════════════════
// TOKEN BUCKET
// ════════════════════════════════════════════════════════════════════════════════

// TokenBucket implements the token bucket algorithm.
type TokenBucket struct {
	tokens     float64
	capacity   int
	refillRate float64
	lastRefill time.Time
	mu         sync.Mutex
}

// NewTokenBucket creates a new token bucket.
func NewTokenBucket(capacity int, refillRate float64) *TokenBucket {
	return &TokenBucket{
		tokens:     float64(capacity),
		capacity:   capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

// Take attempts to take n tokens from the bucket.
func (tb *TokenBucket) Take(n int) *Result {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	// Refill tokens
	tb.refill()

	// Check if enough tokens
	if tb.tokens >= float64(n) {
		tb.tokens -= float64(n)
		return &Result{
			Allowed:   true,
			Remaining: int(tb.tokens),
			Limit:     tb.capacity,
			ResetAt:   tb.calculateResetTime(),
		}
	}

	// Calculate retry after
	needed := float64(n) - tb.tokens
	retryAfter := time.Duration(needed/tb.refillRate*1000) * time.Millisecond

	return &Result{
		Allowed:    false,
		Remaining:  0,
		Limit:      tb.capacity,
		ResetAt:    tb.calculateResetTime(),
		RetryAfter: retryAfter,
	}
}

func (tb *TokenBucket) refill() {
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens = min(float64(tb.capacity), tb.tokens+elapsed*tb.refillRate)
	tb.lastRefill = now
}

func (tb *TokenBucket) calculateResetTime() time.Time {
	tokensNeeded := float64(tb.capacity) - tb.tokens
	secondsToFull := tokensNeeded / tb.refillRate
	return time.Now().Add(time.Duration(secondsToFull*1000) * time.Millisecond)
}

// ════════════════════════════════════════════════════════════════════════════════
// SLIDING WINDOW
// ════════════════════════════════════════════════════════════════════════════════

// SlidingWindow implements sliding window rate limiting.
type SlidingWindow struct {
	limit      int
	windowSize time.Duration
	requests   []time.Time
	mu         sync.Mutex
}

// NewSlidingWindow creates a new sliding window limiter.
func NewSlidingWindow(limit int, windowSize time.Duration) *SlidingWindow {
	return &SlidingWindow{
		limit:      limit,
		windowSize: windowSize,
		requests:   make([]time.Time, 0, limit),
	}
}

// Allow checks if a request is allowed.
func (sw *SlidingWindow) Allow() *Result {
	sw.mu.Lock()
	defer sw.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-sw.windowSize)

	// Remove expired entries
	validIdx := 0
	for i, t := range sw.requests {
		if t.After(windowStart) {
			validIdx = i
			break
		}
	}
	if validIdx > 0 {
		sw.requests = sw.requests[validIdx:]
	}

	// Check limit
	if len(sw.requests) >= sw.limit {
		retryAfter := sw.requests[0].Add(sw.windowSize).Sub(now)
		return &Result{
			Allowed:    false,
			Remaining:  0,
			Limit:      sw.limit,
			ResetAt:    sw.requests[0].Add(sw.windowSize),
			RetryAfter: retryAfter,
		}
	}

	// Add request
	sw.requests = append(sw.requests, now)

	return &Result{
		Allowed:   true,
		Remaining: sw.limit - len(sw.requests),
		Limit:     sw.limit,
		ResetAt:   now.Add(sw.windowSize),
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// RATE LIMITER
// ════════════════════════════════════════════════════════════════════════════════

// RateLimiter is the main rate limiter.
type RateLimiter struct {
	config  Config
	logger  *zap.Logger
	buckets sync.Map // map[string]*TokenBucket
	windows sync.Map // map[string]*SlidingWindow
	limits  sync.Map // map[string]Limit

	// Stats
	allowed atomic.Uint64
	denied  atomic.Uint64

	// Cleanup
	ctx    context.Context
	cancel context.CancelFunc
}

// New creates a new rate limiter.
func New(config Config, logger *zap.Logger) *RateLimiter {
	ctx, cancel := context.WithCancel(context.Background())

	rl := &RateLimiter{
		config: config,
		logger: logger,
		ctx:    ctx,
		cancel: cancel,
	}

	// Start cleanup goroutine
	go rl.cleanupLoop()

	return rl
}

// Allow checks if a request is allowed for the given key.
func (rl *RateLimiter) Allow(key string) *Result {
	return rl.AllowN(key, 1)
}

// AllowN checks if n requests are allowed for the given key.
func (rl *RateLimiter) AllowN(key string, n int) *Result {
	limit := rl.getLimit(key)

	// Unlimited
	if limit.RequestsPerSecond == 0 {
		return &Result{Allowed: true, Remaining: -1}
	}

	bucket := rl.getBucket(key, limit)
	result := bucket.Take(n)

	if result.Allowed {
		rl.allowed.Add(uint64(n))
	} else {
		rl.denied.Add(uint64(n))
	}

	return result
}

// SetLimit sets a custom limit for a key.
func (rl *RateLimiter) SetLimit(key string, limit Limit) {
	rl.limits.Store(key, limit)
	// Clear existing bucket to apply new limit
	rl.buckets.Delete(key)
}

// GetLimit returns the current limit for a key.
func (rl *RateLimiter) GetLimit(key string) Limit {
	return rl.getLimit(key)
}

func (rl *RateLimiter) getLimit(key string) Limit {
	if v, ok := rl.limits.Load(key); ok {
		return v.(Limit)
	}
	return Limit{
		RequestsPerSecond: rl.config.DefaultRequestsPerSecond,
		BurstSize:         rl.config.DefaultBurstSize,
	}
}

func (rl *RateLimiter) getBucket(key string, limit Limit) *TokenBucket {
	if v, ok := rl.buckets.Load(key); ok {
		return v.(*TokenBucket)
	}

	bucket := NewTokenBucket(limit.BurstSize, float64(limit.RequestsPerSecond))
	actual, _ := rl.buckets.LoadOrStore(key, bucket)
	return actual.(*TokenBucket)
}

func (rl *RateLimiter) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-rl.ctx.Done():
			return
		case <-ticker.C:
			rl.cleanup()
		}
	}
}

func (rl *RateLimiter) cleanup() {
	// Clean up idle buckets
	rl.buckets.Range(func(key, value interface{}) bool {
		bucket := value.(*TokenBucket)
		bucket.mu.Lock()
		idle := time.Since(bucket.lastRefill) > 5*time.Minute
		bucket.mu.Unlock()

		if idle {
			rl.buckets.Delete(key)
		}
		return true
	})
}

// Stats returns rate limiter statistics.
func (rl *RateLimiter) Stats() map[string]interface{} {
	bucketCount := 0
	rl.buckets.Range(func(_, _ interface{}) bool {
		bucketCount++
		return true
	})

	return map[string]interface{}{
		"allowed":      rl.allowed.Load(),
		"denied":       rl.denied.Load(),
		"bucket_count": bucketCount,
	}
}

// Close shuts down the rate limiter.
func (rl *RateLimiter) Close() {
	rl.cancel()
}

// ════════════════════════════════════════════════════════════════════════════════
// MIDDLEWARE
// ════════════════════════════════════════════════════════════════════════════════

// KeyFunc extracts the rate limit key from a request.
type KeyFunc func(interface{}) string

// Middleware wraps a handler with rate limiting.
type Middleware struct {
	limiter *RateLimiter
	keyFunc KeyFunc
	logger  *zap.Logger
}

// NewMiddleware creates a new rate limiting middleware.
func NewMiddleware(limiter *RateLimiter, keyFunc KeyFunc, logger *zap.Logger) *Middleware {
	return &Middleware{
		limiter: limiter,
		keyFunc: keyFunc,
		logger:  logger,
	}
}

// Check performs a rate limit check.
func (m *Middleware) Check(req interface{}) *Result {
	key := m.keyFunc(req)
	result := m.limiter.Allow(key)

	if !result.Allowed {
		m.logger.Warn("Rate limit exceeded",
			zap.String("key", key),
			zap.Duration("retry_after", result.RetryAfter))
	}

	return result
}

// ════════════════════════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════════════════════════

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}




