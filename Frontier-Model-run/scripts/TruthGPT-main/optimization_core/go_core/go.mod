module github.com/truthgpt/optimization_core/go_core

go 1.22

// ════════════════════════════════════════════════════════════════════════════════
// 🚀 HIGH-PERFORMANCE WEB & API
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Web Frameworks (Fiber: fastest Go framework, ~370K req/s)
	github.com/gofiber/fiber/v2 v2.52.0
	github.com/gofiber/contrib/otelfiber/v2 v2.0.0
	github.com/gofiber/contrib/websocket v1.3.0
	github.com/valyala/fasthttp v1.52.0

	// gRPC & Protocol Buffers
	google.golang.org/grpc v1.62.0
	google.golang.org/protobuf v1.33.0
	github.com/grpc-ecosystem/go-grpc-middleware/v2 v2.0.1
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.19.1
	
	// GraphQL (optional)
	github.com/99designs/gqlgen v0.17.44
)

// ════════════════════════════════════════════════════════════════════════════════
// 💾 DATABASES & CACHING (500K+ ops/s)
// ════════════════════════════════════════════════════════════════════════════════

require (
	// BadgerDB - Fast embedded KV store (written in Go, LSM tree)
	github.com/dgraph-io/badger/v4 v4.2.0
	
	// Ristretto - High-performance LRU with TinyLFU admission
	github.com/dgraph-io/ristretto v0.1.1
	
	// fastcache - Ultra-fast in-memory cache (50M+ ops/s)
	github.com/VictoriaMetrics/fastcache v1.12.2
	
	// Redis client (go-redis v9 with generics)
	github.com/redis/go-redis/v9 v9.5.1
	
	// PostgreSQL driver (pgx: fastest Go Postgres driver)
	github.com/jackc/pgx/v5 v5.5.3
	
	// SQLite (for local development/testing)
	modernc.org/sqlite v1.29.2
)

// ════════════════════════════════════════════════════════════════════════════════
// 📡 MESSAGING & EVENTS (18M+ msg/s)
// ════════════════════════════════════════════════════════════════════════════════

require (
	// NATS - Ultra-fast messaging (18M+ msg/s)
	github.com/nats-io/nats.go v1.33.1
	github.com/nats-io/nats-server/v2 v2.10.11
	
	// Watermill - Event-driven architecture framework
	github.com/ThreeDotsLabs/watermill v1.3.5
	github.com/ThreeDotsLabs/watermill-nats/v2 v2.0.2
	
	// Kafka (optional, for high-throughput scenarios)
	github.com/segmentio/kafka-go v0.4.47
)

// ════════════════════════════════════════════════════════════════════════════════
// 📊 DATA PROCESSING & ML
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Gonum - Scientific computing (NumPy for Go)
	gonum.org/v1/gonum v0.14.0
	
	// Arrow - Columnar data processing
	github.com/apache/arrow/go/v15 v15.0.0
	
	// Parquet - Efficient columnar storage
	github.com/xitongsys/parquet-go v1.6.2
)

// ════════════════════════════════════════════════════════════════════════════════
// 🗜️ COMPRESSION (5GB/s+)
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Zstd - Best ratio/speed balance (klauspost's pure Go impl)
	github.com/klauspost/compress v1.17.7
	
	// LZ4 - Ultra-fast compression (~5GB/s)
	github.com/pierrec/lz4/v4 v4.1.21
	
	// S2 - Snappy-compatible, faster
	// (included in klauspost/compress)
)

// ════════════════════════════════════════════════════════════════════════════════
// 📦 SERIALIZATION (High-performance JSON, MessagePack, etc.)
// ════════════════════════════════════════════════════════════════════════════════

require (
	// go-json - Fastest JSON library for Go
	github.com/goccy/go-json v0.10.2
	
	// sonic - ByteDance's ultra-fast JSON (uses SIMD)
	github.com/bytedance/sonic v1.11.2
	
	// MessagePack - Efficient binary serialization
	github.com/vmihailenco/msgpack/v5 v5.4.1
	
	// FlatBuffers - Zero-copy serialization
	github.com/google/flatbuffers v24.3.7+incompatible
	
	// CBOR - Compact Binary Object Representation
	github.com/fxamacker/cbor/v2 v2.6.0
)

// ════════════════════════════════════════════════════════════════════════════════
// 🔐 SECURITY & CRYPTOGRAPHY
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Standard crypto
	golang.org/x/crypto v0.21.0
	
	// Age - Modern file encryption
	filippo.io/age v1.1.1
	
	// JWT - JSON Web Tokens
	github.com/golang-jwt/jwt/v5 v5.2.1
	
	// PASETO - Platform-Agnostic Security Tokens
	github.com/o1egl/paseto v1.0.0
	
	// Argon2 password hashing (in golang.org/x/crypto)
)

// ════════════════════════════════════════════════════════════════════════════════
// ⚙️ CONFIGURATION & CLI
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Viper - Configuration management
	github.com/spf13/viper v1.18.2
	
	// Cobra - CLI framework
	github.com/spf13/cobra v1.8.0
	
	// envconfig - Environment-based configuration
	github.com/kelseyhightower/envconfig v1.4.0
	
	// validator - Struct validation
	github.com/go-playground/validator/v10 v10.19.0
)

// ════════════════════════════════════════════════════════════════════════════════
// 📝 LOGGING & OBSERVABILITY
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Zap - Blazing fast, structured logging
	go.uber.org/zap v1.27.0
	
	// Zerolog - Zero allocation JSON logger
	github.com/rs/zerolog v1.32.0
	
	// Prometheus - Metrics
	github.com/prometheus/client_golang v1.19.0
	
	// OpenTelemetry - Distributed tracing
	go.opentelemetry.io/otel v1.24.0
	go.opentelemetry.io/otel/trace v1.24.0
	go.opentelemetry.io/otel/metric v1.24.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.24.0
	go.opentelemetry.io/otel/exporters/prometheus v0.46.0
	go.opentelemetry.io/otel/sdk v1.24.0
)

// ════════════════════════════════════════════════════════════════════════════════
// 🛠️ UTILITIES & CONCURRENCY
// ════════════════════════════════════════════════════════════════════════════════

require (
	// UUID generation
	github.com/google/uuid v1.6.0
	
	// ULID - Universally Unique Lexicographically Sortable Identifier
	github.com/oklog/ulid/v2 v2.1.0
	
	// Sync utilities
	golang.org/x/sync v0.6.0
	
	// Rate limiting
	golang.org/x/time v0.5.0
	
	// ants - Goroutine pool (10x less memory)
	github.com/panjf2000/ants/v2 v2.9.0
	
	// conc - Structured concurrency
	github.com/sourcegraph/conc v0.3.0
	
	// errgroup with semaphore
	golang.org/x/exp v0.0.0-20240318143956-a85f2c67cd81
	
	// lo - Lodash-style Go utilities (generics)
	github.com/samber/lo v1.39.0
)

// ════════════════════════════════════════════════════════════════════════════════
// 🧪 TESTING & BENCHMARKING
// ════════════════════════════════════════════════════════════════════════════════

require (
	// Testify - Assertions and mocking
	github.com/stretchr/testify v1.9.0
	
	// GoMock - Mocking framework
	go.uber.org/mock v0.4.0
	
	// Rapid - Property-based testing
	pgregory.net/rapid v1.1.0
	
	// Ginko + Gomega - BDD testing
	github.com/onsi/ginkgo/v2 v2.16.0
	github.com/onsi/gomega v1.31.1
)

// ════════════════════════════════════════════════════════════════════════════════
// INDIRECT DEPENDENCIES
// ════════════════════════════════════════════════════════════════════════════════

require (
	github.com/andybalholm/brotli v1.1.0 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/glog v1.2.0 // indirect
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da // indirect
	github.com/golang/protobuf v1.5.4 // indirect
	github.com/golang/snappy v0.0.4 // indirect
	github.com/klauspost/cpuid/v2 v2.2.7 // indirect
	github.com/mattn/go-colorable v0.1.13 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mattn/go-runewidth v0.0.15 // indirect
	github.com/nats-io/nkeys v0.4.7 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/valyala/bytebufferpool v1.0.0 // indirect
	github.com/valyala/tcplisten v1.0.0 // indirect
	go.opencensus.io v0.24.0 // indirect
	golang.org/x/net v0.22.0 // indirect
	golang.org/x/sys v0.18.0 // indirect
	golang.org/x/text v0.14.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240318140521-94a12d6c2237 // indirect
)

// ════════════════════════════════════════════════════════════════════════════════
// BUILD TAGS FOR OPTIONAL FEATURES
// ════════════════════════════════════════════════════════════════════════════════
// Use: go build -tags "sonic" for SIMD JSON
// Use: go build -tags "cuda" for CUDA acceleration (if available)
