// Package loadbalancer provides intelligent load balancing for TruthGPT inference servers.
//
// Features:
// - Multiple algorithms (Round Robin, Least Connections, Weighted)
// - Health checking with circuit breaker
// - Request routing based on load
// - Sticky sessions for KV cache locality
package loadbalancer

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ════════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// Algorithm represents load balancing algorithm.
type Algorithm int

const (
	AlgorithmRoundRobin Algorithm = iota
	AlgorithmLeastConnections
	AlgorithmWeighted
	AlgorithmRandom
	AlgorithmConsistentHash
)

// Config holds load balancer configuration.
type Config struct {
	Algorithm           Algorithm     `yaml:"algorithm"`
	HealthCheckInterval time.Duration `yaml:"health_check_interval"`
	HealthCheckTimeout  time.Duration `yaml:"health_check_timeout"`
	MaxRetries          int           `yaml:"max_retries"`
	RetryBackoff        time.Duration `yaml:"retry_backoff"`
	// Circuit breaker
	CircuitBreakerThreshold   int           `yaml:"circuit_breaker_threshold"`
	CircuitBreakerTimeout     time.Duration `yaml:"circuit_breaker_timeout"`
	CircuitBreakerHalfOpenMax int           `yaml:"circuit_breaker_half_open_max"`
}

// DefaultConfig returns default load balancer configuration.
func DefaultConfig() Config {
	return Config{
		Algorithm:                 AlgorithmLeastConnections,
		HealthCheckInterval:       5 * time.Second,
		HealthCheckTimeout:        2 * time.Second,
		MaxRetries:                3,
		RetryBackoff:              100 * time.Millisecond,
		CircuitBreakerThreshold:   5,
		CircuitBreakerTimeout:     30 * time.Second,
		CircuitBreakerHalfOpenMax: 3,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// BACKEND
// ════════════════════════════════════════════════════════════════════════════════

// BackendState represents the health state of a backend.
type BackendState int

const (
	BackendHealthy BackendState = iota
	BackendUnhealthy
	BackendDraining
)

// Backend represents an inference server backend.
type Backend struct {
	ID       string
	Address  string
	Weight   int
	Metadata map[string]string

	// State
	state       atomic.Int32
	connections atomic.Int64
	requests    atomic.Uint64
	errors      atomic.Uint64
	latencySum  atomic.Uint64
	latencyCount atomic.Uint64

	// Circuit breaker
	failures    atomic.Int32
	lastFailure atomic.Int64
	halfOpen    atomic.Int32

	// Health check
	lastHealthCheck atomic.Int64
	healthScore     atomic.Int32 // 0-100
}

// NewBackend creates a new backend.
func NewBackend(id, address string, weight int) *Backend {
	b := &Backend{
		ID:       id,
		Address:  address,
		Weight:   weight,
		Metadata: make(map[string]string),
	}
	b.state.Store(int32(BackendHealthy))
	b.healthScore.Store(100)
	return b
}

// State returns the backend state.
func (b *Backend) State() BackendState {
	return BackendState(b.state.Load())
}

// IsHealthy returns true if backend is healthy.
func (b *Backend) IsHealthy() bool {
	return b.State() == BackendHealthy
}

// Connections returns active connection count.
func (b *Backend) Connections() int64 {
	return b.connections.Load()
}

// AvgLatency returns average latency in milliseconds.
func (b *Backend) AvgLatency() float64 {
	count := b.latencyCount.Load()
	if count == 0 {
		return 0
	}
	return float64(b.latencySum.Load()) / float64(count)
}

// RecordRequest records a completed request.
func (b *Backend) RecordRequest(latencyMs uint64, success bool) {
	b.requests.Add(1)
	b.latencySum.Add(latencyMs)
	b.latencyCount.Add(1)

	if !success {
		b.errors.Add(1)
		b.failures.Add(1)
		b.lastFailure.Store(time.Now().UnixNano())
	} else {
		// Reset failures on success
		b.failures.Store(0)
	}
}

// AcquireConnection increments connection count.
func (b *Backend) AcquireConnection() {
	b.connections.Add(1)
}

// ReleaseConnection decrements connection count.
func (b *Backend) ReleaseConnection() {
	b.connections.Add(-1)
}

// ════════════════════════════════════════════════════════════════════════════════
// LOAD BALANCER
// ════════════════════════════════════════════════════════════════════════════════

// LoadBalancer manages backend selection and health checking.
type LoadBalancer struct {
	config   Config
	backends []*Backend
	mu       sync.RWMutex
	logger   *zap.Logger

	// Round robin state
	rrIndex atomic.Uint64

	// Consistent hash ring
	hashRing *ConsistentHashRing

	// Stats
	totalRequests   atomic.Uint64
	failedRequests  atomic.Uint64
	retriedRequests atomic.Uint64

	// Shutdown
	ctx    context.Context
	cancel context.CancelFunc
}

// New creates a new load balancer.
func New(config Config, logger *zap.Logger) *LoadBalancer {
	ctx, cancel := context.WithCancel(context.Background())

	lb := &LoadBalancer{
		config:   config,
		backends: make([]*Backend, 0),
		logger:   logger,
		hashRing: NewConsistentHashRing(100), // 100 virtual nodes
		ctx:      ctx,
		cancel:   cancel,
	}

	// Start health checker
	go lb.healthCheckLoop()

	return lb
}

// AddBackend adds a backend to the load balancer.
func (lb *LoadBalancer) AddBackend(backend *Backend) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.backends = append(lb.backends, backend)
	lb.hashRing.Add(backend.ID)
	lb.logger.Info("Added backend",
		zap.String("id", backend.ID),
		zap.String("address", backend.Address),
		zap.Int("weight", backend.Weight))
}

// RemoveBackend removes a backend from the load balancer.
func (lb *LoadBalancer) RemoveBackend(id string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	for i, b := range lb.backends {
		if b.ID == id {
			lb.backends = append(lb.backends[:i], lb.backends[i+1:]...)
			lb.hashRing.Remove(id)
			lb.logger.Info("Removed backend", zap.String("id", id))
			return
		}
	}
}

// Select selects a backend based on the configured algorithm.
func (lb *LoadBalancer) Select(key string) (*Backend, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	lb.totalRequests.Add(1)

	// Get healthy backends
	healthy := lb.getHealthyBackends()
	if len(healthy) == 0 {
		lb.failedRequests.Add(1)
		return nil, fmt.Errorf("no healthy backends available")
	}

	var selected *Backend

	switch lb.config.Algorithm {
	case AlgorithmRoundRobin:
		selected = lb.selectRoundRobin(healthy)
	case AlgorithmLeastConnections:
		selected = lb.selectLeastConnections(healthy)
	case AlgorithmWeighted:
		selected = lb.selectWeighted(healthy)
	case AlgorithmRandom:
		selected = lb.selectRandom(healthy)
	case AlgorithmConsistentHash:
		selected = lb.selectConsistentHash(healthy, key)
	default:
		selected = lb.selectRoundRobin(healthy)
	}

	if selected != nil {
		selected.AcquireConnection()
	}

	return selected, nil
}

// SelectWithRetry selects a backend with retries on failure.
func (lb *LoadBalancer) SelectWithRetry(key string, fn func(*Backend) error) error {
	var lastErr error

	for attempt := 0; attempt <= lb.config.MaxRetries; attempt++ {
		backend, err := lb.Select(key)
		if err != nil {
			return err
		}

		start := time.Now()
		err = fn(backend)
		latency := uint64(time.Since(start).Milliseconds())

		backend.RecordRequest(latency, err == nil)
		backend.ReleaseConnection()

		if err == nil {
			return nil
		}

		lastErr = err
		lb.retriedRequests.Add(1)

		// Check circuit breaker
		if lb.shouldOpenCircuit(backend) {
			lb.openCircuit(backend)
		}

		if attempt < lb.config.MaxRetries {
			time.Sleep(lb.config.RetryBackoff * time.Duration(attempt+1))
		}
	}

	lb.failedRequests.Add(1)
	return fmt.Errorf("all retries failed: %w", lastErr)
}

// Release releases a backend connection.
func (lb *LoadBalancer) Release(backend *Backend) {
	if backend != nil {
		backend.ReleaseConnection()
	}
}

func (lb *LoadBalancer) getHealthyBackends() []*Backend {
	healthy := make([]*Backend, 0, len(lb.backends))
	for _, b := range lb.backends {
		if b.IsHealthy() && !lb.isCircuitOpen(b) {
			healthy = append(healthy, b)
		}
	}
	return healthy
}

func (lb *LoadBalancer) selectRoundRobin(backends []*Backend) *Backend {
	idx := lb.rrIndex.Add(1) - 1
	return backends[idx%uint64(len(backends))]
}

func (lb *LoadBalancer) selectLeastConnections(backends []*Backend) *Backend {
	var selected *Backend
	minConns := int64(1<<63 - 1)

	for _, b := range backends {
		conns := b.Connections()
		if conns < minConns {
			minConns = conns
			selected = b
		}
	}

	return selected
}

func (lb *LoadBalancer) selectWeighted(backends []*Backend) *Backend {
	totalWeight := 0
	for _, b := range backends {
		totalWeight += b.Weight
	}

	if totalWeight == 0 {
		return backends[0]
	}

	r := rand.Intn(totalWeight)
	for _, b := range backends {
		r -= b.Weight
		if r < 0 {
			return b
		}
	}

	return backends[0]
}

func (lb *LoadBalancer) selectRandom(backends []*Backend) *Backend {
	return backends[rand.Intn(len(backends))]
}

func (lb *LoadBalancer) selectConsistentHash(backends []*Backend, key string) *Backend {
	id := lb.hashRing.Get(key)
	for _, b := range backends {
		if b.ID == id {
			return b
		}
	}
	// Fallback to round robin
	return lb.selectRoundRobin(backends)
}

// ════════════════════════════════════════════════════════════════════════════════
// CIRCUIT BREAKER
// ════════════════════════════════════════════════════════════════════════════════

func (lb *LoadBalancer) shouldOpenCircuit(b *Backend) bool {
	return int(b.failures.Load()) >= lb.config.CircuitBreakerThreshold
}

func (lb *LoadBalancer) isCircuitOpen(b *Backend) bool {
	failures := b.failures.Load()
	if failures < int32(lb.config.CircuitBreakerThreshold) {
		return false
	}

	// Check if timeout has passed (half-open)
	lastFailure := time.Unix(0, b.lastFailure.Load())
	if time.Since(lastFailure) > lb.config.CircuitBreakerTimeout {
		// Allow limited requests in half-open state
		if b.halfOpen.Load() < int32(lb.config.CircuitBreakerHalfOpenMax) {
			b.halfOpen.Add(1)
			return false
		}
	}

	return true
}

func (lb *LoadBalancer) openCircuit(b *Backend) {
	lb.logger.Warn("Opening circuit breaker",
		zap.String("backend", b.ID),
		zap.Int32("failures", b.failures.Load()))
	b.state.Store(int32(BackendUnhealthy))
}

func (lb *LoadBalancer) closeCircuit(b *Backend) {
	b.failures.Store(0)
	b.halfOpen.Store(0)
	b.state.Store(int32(BackendHealthy))
	lb.logger.Info("Closing circuit breaker", zap.String("backend", b.ID))
}

// ════════════════════════════════════════════════════════════════════════════════
// HEALTH CHECKING
// ════════════════════════════════════════════════════════════════════════════════

func (lb *LoadBalancer) healthCheckLoop() {
	ticker := time.NewTicker(lb.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-lb.ctx.Done():
			return
		case <-ticker.C:
			lb.checkAllBackends()
		}
	}
}

func (lb *LoadBalancer) checkAllBackends() {
	lb.mu.RLock()
	backends := make([]*Backend, len(lb.backends))
	copy(backends, lb.backends)
	lb.mu.RUnlock()

	var wg sync.WaitGroup
	for _, b := range backends {
		wg.Add(1)
		go func(backend *Backend) {
			defer wg.Done()
			lb.checkBackend(backend)
		}(b)
	}
	wg.Wait()
}

func (lb *LoadBalancer) checkBackend(b *Backend) {
	// TODO: Implement actual health check (gRPC health check, HTTP endpoint, etc.)
	// For now, we check based on recent error rate

	errorRate := float64(0)
	if b.requests.Load() > 0 {
		errorRate = float64(b.errors.Load()) / float64(b.requests.Load())
	}

	b.lastHealthCheck.Store(time.Now().UnixNano())

	if errorRate > 0.5 && b.requests.Load() > 10 {
		b.healthScore.Store(int32(100 * (1 - errorRate)))
		if !b.IsHealthy() {
			return // Already unhealthy
		}
		lb.logger.Warn("Backend health degraded",
			zap.String("backend", b.ID),
			zap.Float64("error_rate", errorRate))
	} else if !b.IsHealthy() {
		// Check if circuit can be closed
		lastFailure := time.Unix(0, b.lastFailure.Load())
		if time.Since(lastFailure) > lb.config.CircuitBreakerTimeout*2 {
			lb.closeCircuit(b)
		}
	}
}

// Stats returns load balancer statistics.
func (lb *LoadBalancer) Stats() map[string]interface{} {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	backendStats := make([]map[string]interface{}, len(lb.backends))
	for i, b := range lb.backends {
		backendStats[i] = map[string]interface{}{
			"id":          b.ID,
			"address":     b.Address,
			"state":       b.State().String(),
			"connections": b.Connections(),
			"requests":    b.requests.Load(),
			"errors":      b.errors.Load(),
			"avg_latency": b.AvgLatency(),
			"health":      b.healthScore.Load(),
		}
	}

	return map[string]interface{}{
		"total_requests":   lb.totalRequests.Load(),
		"failed_requests":  lb.failedRequests.Load(),
		"retried_requests": lb.retriedRequests.Load(),
		"backends":         backendStats,
	}
}

// Close shuts down the load balancer.
func (lb *LoadBalancer) Close() {
	lb.cancel()
}

// String returns string representation of backend state.
func (s BackendState) String() string {
	switch s {
	case BackendHealthy:
		return "healthy"
	case BackendUnhealthy:
		return "unhealthy"
	case BackendDraining:
		return "draining"
	default:
		return "unknown"
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// CONSISTENT HASH RING
// ════════════════════════════════════════════════════════════════════════════════

// ConsistentHashRing implements consistent hashing for sticky sessions.
type ConsistentHashRing struct {
	virtualNodes int
	ring         map[uint32]string
	sortedKeys   []uint32
	mu           sync.RWMutex
}

// NewConsistentHashRing creates a new consistent hash ring.
func NewConsistentHashRing(virtualNodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		virtualNodes: virtualNodes,
		ring:         make(map[uint32]string),
		sortedKeys:   make([]uint32, 0),
	}
}

// Add adds a node to the ring.
func (c *ConsistentHashRing) Add(node string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.virtualNodes; i++ {
		key := c.hash(fmt.Sprintf("%s:%d", node, i))
		c.ring[key] = node
		c.sortedKeys = append(c.sortedKeys, key)
	}

	// Sort keys
	c.sortKeys()
}

// Remove removes a node from the ring.
func (c *ConsistentHashRing) Remove(node string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := 0; i < c.virtualNodes; i++ {
		key := c.hash(fmt.Sprintf("%s:%d", node, i))
		delete(c.ring, key)
	}

	// Rebuild sorted keys
	c.sortedKeys = c.sortedKeys[:0]
	for k := range c.ring {
		c.sortedKeys = append(c.sortedKeys, k)
	}
	c.sortKeys()
}

// Get returns the node for a given key.
func (c *ConsistentHashRing) Get(key string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.ring) == 0 {
		return ""
	}

	h := c.hash(key)

	// Binary search for the first key >= h
	idx := c.search(h)
	return c.ring[c.sortedKeys[idx]]
}

func (c *ConsistentHashRing) hash(key string) uint32 {
	// FNV-1a hash
	var h uint32 = 2166136261
	for i := 0; i < len(key); i++ {
		h ^= uint32(key[i])
		h *= 16777619
	}
	return h
}

func (c *ConsistentHashRing) sortKeys() {
	// Simple bubble sort for small arrays
	for i := 0; i < len(c.sortedKeys)-1; i++ {
		for j := 0; j < len(c.sortedKeys)-i-1; j++ {
			if c.sortedKeys[j] > c.sortedKeys[j+1] {
				c.sortedKeys[j], c.sortedKeys[j+1] = c.sortedKeys[j+1], c.sortedKeys[j]
			}
		}
	}
}

func (c *ConsistentHashRing) search(h uint32) int {
	lo, hi := 0, len(c.sortedKeys)
	for lo < hi {
		mid := (lo + hi) / 2
		if c.sortedKeys[mid] < h {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= len(c.sortedKeys) {
		lo = 0
	}
	return lo
}




