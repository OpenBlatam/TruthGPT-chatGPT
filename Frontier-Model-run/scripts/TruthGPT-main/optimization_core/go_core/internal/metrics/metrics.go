// Package metrics provides Prometheus metrics for TruthGPT Go Core.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ════════════════════════════════════════════════════════════════════════════════
// INFERENCE METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	// InferenceRequestsTotal counts total inference requests
	InferenceRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "inference",
			Name:      "requests_total",
			Help:      "Total number of inference requests",
		},
		[]string{"status", "model"},
	)

	// InferenceLatency measures inference latency
	InferenceLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "truthgpt",
			Subsystem: "inference",
			Name:      "latency_seconds",
			Help:      "Inference request latency in seconds",
			Buckets:   []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		},
		[]string{"model"},
	)

	// TokensGenerated counts total tokens generated
	TokensGenerated = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "inference",
			Name:      "tokens_generated_total",
			Help:      "Total number of tokens generated",
		},
		[]string{"model"},
	)

	// ActiveRequests tracks currently active requests
	ActiveRequests = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "truthgpt",
			Subsystem: "inference",
			Name:      "active_requests",
			Help:      "Number of currently active inference requests",
		},
	)
)

// ════════════════════════════════════════════════════════════════════════════════
// CACHE METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	// CacheHits counts cache hits
	CacheHits = promauto.NewCounter(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "cache",
			Name:      "hits_total",
			Help:      "Total number of cache hits",
		},
	)

	// CacheMisses counts cache misses
	CacheMisses = promauto.NewCounter(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "cache",
			Name:      "misses_total",
			Help:      "Total number of cache misses",
		},
	)

	// CacheSize tracks current cache size in bytes
	CacheSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "truthgpt",
			Subsystem: "cache",
			Name:      "size_bytes",
			Help:      "Current cache size in bytes",
		},
	)

	// CacheOperations counts cache operations
	CacheOperations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "cache",
			Name:      "operations_total",
			Help:      "Total number of cache operations",
		},
		[]string{"operation"}, // get, put, delete
	)

	// CacheLatency measures cache operation latency
	CacheLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "truthgpt",
			Subsystem: "cache",
			Name:      "latency_seconds",
			Help:      "Cache operation latency in seconds",
			Buckets:   []float64{.0001, .0005, .001, .005, .01, .025, .05, .1},
		},
		[]string{"operation", "tier"}, // tier: hot, warm, cold
	)
)

// ════════════════════════════════════════════════════════════════════════════════
// MESSAGING METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	// MessagesPublished counts published messages
	MessagesPublished = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "messaging",
			Name:      "messages_published_total",
			Help:      "Total number of messages published",
		},
		[]string{"topic"},
	)

	// MessagesReceived counts received messages
	MessagesReceived = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "truthgpt",
			Subsystem: "messaging",
			Name:      "messages_received_total",
			Help:      "Total number of messages received",
		},
		[]string{"topic"},
	)

	// MessageSize tracks message sizes
	MessageSize = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "truthgpt",
			Subsystem: "messaging",
			Name:      "message_size_bytes",
			Help:      "Size of messages in bytes",
			Buckets:   []float64{100, 1000, 10000, 100000, 1000000, 10000000},
		},
		[]string{"topic"},
	)
)

// ════════════════════════════════════════════════════════════════════════════════
// SYSTEM METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	// GoroutineCount tracks number of goroutines
	GoroutineCount = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "truthgpt",
			Subsystem: "system",
			Name:      "goroutines",
			Help:      "Current number of goroutines",
		},
	)

	// HeapAlloc tracks heap allocation
	HeapAlloc = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "truthgpt",
			Subsystem: "system",
			Name:      "heap_alloc_bytes",
			Help:      "Current heap allocation in bytes",
		},
	)

	// BuildInfo provides build information
	BuildInfo = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "truthgpt",
			Subsystem: "system",
			Name:      "build_info",
			Help:      "Build information",
		},
		[]string{"version", "commit", "go_version"},
	)
)

// ════════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════════

// RecordInferenceRequest records an inference request.
func RecordInferenceRequest(status, model string, latency float64, tokens int) {
	InferenceRequestsTotal.WithLabelValues(status, model).Inc()
	InferenceLatency.WithLabelValues(model).Observe(latency)
	if status == "success" {
		TokensGenerated.WithLabelValues(model).Add(float64(tokens))
	}
}

// RecordCacheOperation records a cache operation.
func RecordCacheOperation(operation string, hit bool, latency float64, tier string) {
	CacheOperations.WithLabelValues(operation).Inc()
	CacheLatency.WithLabelValues(operation, tier).Observe(latency)
	if operation == "get" {
		if hit {
			CacheHits.Inc()
		} else {
			CacheMisses.Inc()
		}
	}
}

// RecordMessage records a messaging operation.
func RecordMessage(topic string, published bool, size int) {
	if published {
		MessagesPublished.WithLabelValues(topic).Inc()
	} else {
		MessagesReceived.WithLabelValues(topic).Inc()
	}
	MessageSize.WithLabelValues(topic).Observe(float64(size))
}
