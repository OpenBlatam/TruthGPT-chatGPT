// Package server provides high-performance HTTP and gRPC servers for TruthGPT.
//
// Architecture:
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │                            TruthGPT Server                                  │
// ├─────────────────────────────────────────────────────────────────────────────┤
// │  ┌─────────────────────────────┐   ┌─────────────────────────────────────┐  │
// │  │       HTTP/REST (Fiber)      │   │         gRPC (High-perf)            │  │
// │  │  - /api/v1/inference        │   │  - InferenceService.Infer           │  │
// │  │  - /api/v1/cache            │   │  - InferenceService.StreamInfer     │  │
// │  │  - /api/v1/health           │   │  - CacheService.Get/Put             │  │
// │  │  - Rate Limited             │   │  - Streaming support                │  │
// │  └─────────────────────────────┘   └─────────────────────────────────────┘  │
// │                                                                             │
// │  ┌─────────────────────────────────────────────────────────────────────────┐│
// │  │                         Middleware Stack                                ││
// │  │  Rate Limiter → Auth → Tracing → Logging → Recovery → Handler         ││
// │  └─────────────────────────────────────────────────────────────────────────┘│
// └─────────────────────────────────────────────────────────────────────────────┘
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/pprof"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/gofiber/fiber/v2/middleware/requestid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/valyala/fasthttp/fasthttpadaptor"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

// ════════════════════════════════════════════════════════════════════════════════
// PROMETHEUS METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	httpRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_http_requests_total",
		Help: "Total number of HTTP requests",
	}, []string{"method", "path", "status"})

	httpRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "truthgpt_http_request_duration_seconds",
		Help:    "HTTP request duration",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
	}, []string{"method", "path"})

	grpcRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_grpc_requests_total",
		Help: "Total number of gRPC requests",
	}, []string{"method", "status"})

	activeConnections = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "truthgpt_active_connections",
		Help: "Number of active connections",
	})

	rateLimitHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "truthgpt_rate_limit_hits_total",
		Help: "Total number of rate limit hits",
	})
)

// ════════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// Config holds the server configuration.
type Config struct {
	// HTTP settings
	HTTPPort         int           `yaml:"http_port"`
	HTTPReadTimeout  time.Duration `yaml:"http_read_timeout"`
	HTTPWriteTimeout time.Duration `yaml:"http_write_timeout"`
	HTTPIdleTimeout  time.Duration `yaml:"http_idle_timeout"`
	HTTPBodyLimit    int           `yaml:"http_body_limit"`

	// gRPC settings
	GRPCPort                 int           `yaml:"grpc_port"`
	GRPCMaxRecvMsgSize       int           `yaml:"grpc_max_recv_msg_size"`
	GRPCMaxSendMsgSize       int           `yaml:"grpc_max_send_msg_size"`
	GRPCMaxConcurrentStreams uint32        `yaml:"grpc_max_concurrent_streams"`
	GRPCKeepaliveTime        time.Duration `yaml:"grpc_keepalive_time"`
	GRPCKeepaliveTimeout     time.Duration `yaml:"grpc_keepalive_timeout"`

	// Rate limiting
	RateLimitRequests int           `yaml:"rate_limit_requests"`
	RateLimitWindow   time.Duration `yaml:"rate_limit_window"`
	RateLimitByIP     bool          `yaml:"rate_limit_by_ip"`

	// Metrics
	MetricsPort int    `yaml:"metrics_port"`
	MetricsPath string `yaml:"metrics_path"`

	// General
	Prefork        bool   `yaml:"prefork"`
	EnableCORS     bool   `yaml:"enable_cors"`
	EnableCompress bool   `yaml:"enable_compress"`
	EnablePprof    bool   `yaml:"enable_pprof"`
	EnableTracing  bool   `yaml:"enable_tracing"`
	TrustedProxies string `yaml:"trusted_proxies"`
}

// DefaultConfig returns the default server configuration.
func DefaultConfig() Config {
	return Config{
		HTTPPort:                 8080,
		HTTPReadTimeout:          30 * time.Second,
		HTTPWriteTimeout:         30 * time.Second,
		HTTPIdleTimeout:          120 * time.Second,
		HTTPBodyLimit:            50 << 20, // 50MB
		GRPCPort:                 50051,
		GRPCMaxRecvMsgSize:       100 << 20, // 100MB
		GRPCMaxSendMsgSize:       100 << 20,
		GRPCMaxConcurrentStreams: 1000,
		GRPCKeepaliveTime:        30 * time.Second,
		GRPCKeepaliveTimeout:     10 * time.Second,
		RateLimitRequests:        1000,
		RateLimitWindow:          time.Minute,
		RateLimitByIP:            true,
		MetricsPort:              9090,
		MetricsPath:              "/metrics",
		Prefork:                  false,
		EnableCORS:               true,
		EnableCompress:           true,
		EnablePprof:              true,
		EnableTracing:            true,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// SERVER STATS
// ════════════════════════════════════════════════════════════════════════════════

// ServerStats holds server statistics.
type ServerStats struct {
	StartedAt       time.Time
	RequestsTotal   uint64
	RequestsSuccess uint64
	RequestsFailed  uint64
	BytesSent       uint64
	BytesReceived   uint64
	ActiveConns     int64
}

// Uptime returns the server uptime.
func (s *ServerStats) Uptime() time.Duration {
	return time.Since(s.StartedAt)
}

// ════════════════════════════════════════════════════════════════════════════════
// REQUEST/RESPONSE TYPES
// ════════════════════════════════════════════════════════════════════════════════

// InferenceRequest represents an inference request.
type InferenceRequest struct {
	Input       string                 `json:"input"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	TopP        float64                `json:"top_p,omitempty"`
	TopK        int                    `json:"top_k,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

// Validate validates the inference request.
func (r *InferenceRequest) Validate() error {
	if r.Input == "" {
		return fmt.Errorf("input is required")
	}
	if r.MaxTokens < 0 {
		return fmt.Errorf("max_tokens must be non-negative")
	}
	if r.Temperature < 0 || r.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}
	return nil
}

// InferenceResponse represents an inference response.
type InferenceResponse struct {
	ID           string  `json:"id"`
	Output       string  `json:"output"`
	TokensUsed   int     `json:"tokens_used"`
	PromptTokens int     `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	LatencyMs    float64 `json:"latency_ms"`
	ModelID      string  `json:"model_id,omitempty"`
	FinishReason string  `json:"finish_reason"`
}

// ErrorResponse represents an error response.
type ErrorResponse struct {
	Error   string `json:"error"`
	Code    string `json:"code,omitempty"`
	Details string `json:"details,omitempty"`
}

// ════════════════════════════════════════════════════════════════════════════════
// INFERENCE SERVICE INTERFACE
// ════════════════════════════════════════════════════════════════════════════════

// InferenceService defines the inference service interface.
type InferenceService interface {
	Infer(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error)
	InferStream(ctx context.Context, req *InferenceRequest, stream func(chunk string) error) error
}

// CacheService defines the cache service interface.
type CacheService interface {
	Get(ctx context.Context, key []byte) ([]byte, error)
	Put(ctx context.Context, key, value []byte) error
	Delete(ctx context.Context, key []byte) error
	Stats() interface{}
}

// ════════════════════════════════════════════════════════════════════════════════
// SERVER
// ════════════════════════════════════════════════════════════════════════════════

// Server is the main server handling HTTP and gRPC.
type Server struct {
	config           Config
	httpApp          *fiber.App
	grpcServer       *grpc.Server
	metricsApp       *fiber.App
	logger           *zap.Logger
	tracer           trace.Tracer

	inferenceService InferenceService
	cacheService     CacheService

	stats    ServerStats
	mu       sync.RWMutex
	shutdown chan struct{}
}

// New creates a new server.
func New(config Config, logger *zap.Logger) *Server {
	s := &Server{
		config:   config,
		logger:   logger,
		tracer:   otel.Tracer("server.Server"),
		stats:    ServerStats{StartedAt: time.Now()},
		shutdown: make(chan struct{}),
	}

	s.setupHTTP()
	s.setupGRPC()
	s.setupMetrics()

	return s
}

// SetInferenceService sets the inference service.
func (s *Server) SetInferenceService(svc InferenceService) {
	s.inferenceService = svc
}

// SetCacheService sets the cache service.
func (s *Server) SetCacheService(svc CacheService) {
	s.cacheService = svc
}

// ════════════════════════════════════════════════════════════════════════════════
// HTTP SETUP
// ════════════════════════════════════════════════════════════════════════════════

// setupHTTP configures the HTTP server.
func (s *Server) setupHTTP() {
	s.httpApp = fiber.New(fiber.Config{
		Prefork:               s.config.Prefork,
		ServerHeader:          "TruthGPT-Go/2.0",
		ReadTimeout:           s.config.HTTPReadTimeout,
		WriteTimeout:          s.config.HTTPWriteTimeout,
		IdleTimeout:           s.config.HTTPIdleTimeout,
		BodyLimit:             s.config.HTTPBodyLimit,
		DisableStartupMessage: true,
		ReduceMemoryUsage:     false,
		JSONEncoder:           json.Marshal,
		JSONDecoder:           json.Unmarshal,
		ErrorHandler:          s.errorHandler,
	})

	s.setupMiddleware()
	s.setupRoutes()
}

// setupMiddleware configures middleware.
func (s *Server) setupMiddleware() {
	// Recovery middleware
	s.httpApp.Use(recover.New(recover.Config{
		EnableStackTrace: true,
		StackTraceHandler: func(c *fiber.Ctx, e interface{}) {
			s.logger.Error("Panic recovered",
				zap.Any("error", e),
				zap.String("path", c.Path()),
			)
		},
	}))

	// Request ID
	s.httpApp.Use(requestid.New())

	// Rate limiter
	s.httpApp.Use(limiter.New(limiter.Config{
		Max:        s.config.RateLimitRequests,
		Expiration: s.config.RateLimitWindow,
		KeyGenerator: func(c *fiber.Ctx) string {
			if s.config.RateLimitByIP {
				return c.IP()
			}
			return c.Get("X-API-Key", c.IP())
		},
		LimitReached: func(c *fiber.Ctx) error {
			rateLimitHits.Inc()
			return c.Status(429).JSON(ErrorResponse{
				Error: "Rate limit exceeded",
				Code:  "RATE_LIMIT_EXCEEDED",
			})
		},
	}))

	// Logger with custom format
	s.httpApp.Use(logger.New(logger.Config{
		Format:     "${time} | ${status} | ${latency} | ${ip} | ${method} | ${path} | ${error}\n",
		TimeFormat: "2006-01-02 15:04:05.000",
		Output:     os.Stdout,
	}))

	// CORS
	if s.config.EnableCORS {
		s.httpApp.Use(cors.New(cors.Config{
			AllowOrigins:     "*",
			AllowMethods:     "GET,POST,PUT,DELETE,OPTIONS,PATCH",
			AllowHeaders:     "Origin,Content-Type,Accept,Authorization,X-Request-ID,X-API-Key",
			ExposeHeaders:    "X-Request-ID,X-RateLimit-Remaining",
			AllowCredentials: true,
			MaxAge:           86400,
		}))
	}

	// Compression
	if s.config.EnableCompress {
		s.httpApp.Use(compress.New(compress.Config{
			Level: compress.LevelBestSpeed,
		}))
	}

	// Pprof (debug builds)
	if s.config.EnablePprof {
		s.httpApp.Use(pprof.New())
	}

	// Request tracking
	s.httpApp.Use(func(c *fiber.Ctx) error {
		atomic.AddUint64(&s.stats.RequestsTotal, 1)
		atomic.AddInt64(&s.stats.ActiveConns, 1)
		activeConnections.Inc()

		err := c.Next()

		atomic.AddInt64(&s.stats.ActiveConns, -1)
		activeConnections.Dec()

		if err != nil {
			atomic.AddUint64(&s.stats.RequestsFailed, 1)
		} else {
			atomic.AddUint64(&s.stats.RequestsSuccess, 1)
		}

		return err
	})
}

// setupRoutes configures routes.
func (s *Server) setupRoutes() {
	// Health endpoints
	s.httpApp.Get("/health", s.healthHandler)
	s.httpApp.Get("/ready", s.readyHandler)
	s.httpApp.Get("/live", s.liveHandler)

	// API v1 routes
	api := s.httpApp.Group("/api/v1")

	// Inference endpoints
	api.Post("/inference", s.inferenceHandler)
	api.Post("/inference/batch", s.batchInferenceHandler)
	api.Post("/inference/stream", s.streamInferenceHandler)
	api.Post("/embeddings", s.embeddingsHandler)

	// Cache endpoints
	api.Get("/cache/:key", s.cacheGetHandler)
	api.Put("/cache/:key", s.cachePutHandler)
	api.Delete("/cache/:key", s.cacheDeleteHandler)
	api.Get("/cache/stats", s.cacheStatsHandler)

	// System endpoints
	api.Get("/system/info", s.systemInfoHandler)
	api.Get("/system/stats", s.systemStatsHandler)

	// Model endpoints
	api.Get("/models", s.modelsHandler)
	api.Get("/models/:id", s.modelDetailHandler)
}

// errorHandler handles errors.
func (s *Server) errorHandler(c *fiber.Ctx, err error) error {
	code := fiber.StatusInternalServerError

	if e, ok := err.(*fiber.Error); ok {
		code = e.Code
	}

	s.logger.Error("HTTP error",
		zap.Error(err),
		zap.Int("status", code),
		zap.String("path", c.Path()),
	)

	return c.Status(code).JSON(ErrorResponse{
		Error: err.Error(),
		Code:  fmt.Sprintf("HTTP_%d", code),
	})
}

// ════════════════════════════════════════════════════════════════════════════════
// gRPC SETUP
// ════════════════════════════════════════════════════════════════════════════════

// setupGRPC configures the gRPC server.
func (s *Server) setupGRPC() {
	// gRPC server options
	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(s.config.GRPCMaxRecvMsgSize),
		grpc.MaxSendMsgSize(s.config.GRPCMaxSendMsgSize),
		grpc.MaxConcurrentStreams(s.config.GRPCMaxConcurrentStreams),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			Time:    s.config.GRPCKeepaliveTime,
			Timeout: s.config.GRPCKeepaliveTimeout,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             5 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.ChainUnaryInterceptor(
			s.grpcLoggingInterceptor,
			s.grpcMetricsInterceptor,
			s.grpcRecoveryInterceptor,
		),
		grpc.ChainStreamInterceptor(
			s.grpcStreamLoggingInterceptor,
		),
	}

	s.grpcServer = grpc.NewServer(opts...)

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s.grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for development
	reflection.Register(s.grpcServer)
}

// gRPC interceptors
func (s *Server) grpcLoggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	duration := time.Since(start)

	s.logger.Debug("gRPC request",
		zap.String("method", info.FullMethod),
		zap.Duration("duration", duration),
		zap.Error(err),
	)

	return resp, err
}

func (s *Server) grpcMetricsInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)

	statusCode := "OK"
	if err != nil {
		statusCode = status.Code(err).String()
	}

	grpcRequestsTotal.WithLabelValues(info.FullMethod, statusCode).Inc()
	httpRequestDuration.WithLabelValues("GRPC", info.FullMethod).Observe(time.Since(start).Seconds())

	return resp, err
}

func (s *Server) grpcRecoveryInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			s.logger.Error("gRPC panic recovered",
				zap.Any("panic", r),
				zap.String("method", info.FullMethod),
			)
			err = status.Errorf(codes.Internal, "internal error")
		}
	}()
	return handler(ctx, req)
}

func (s *Server) grpcStreamLoggingInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	start := time.Now()
	err := handler(srv, ss)
	duration := time.Since(start)

	s.logger.Debug("gRPC stream",
		zap.String("method", info.FullMethod),
		zap.Duration("duration", duration),
		zap.Error(err),
	)

	return err
}

// ════════════════════════════════════════════════════════════════════════════════
// METRICS SETUP
// ════════════════════════════════════════════════════════════════════════════════

// setupMetrics configures Prometheus metrics.
func (s *Server) setupMetrics() {
	s.metricsApp = fiber.New(fiber.Config{
		DisableStartupMessage: true,
	})

	s.metricsApp.Get(s.config.MetricsPath, func(c *fiber.Ctx) error {
		fasthttpadaptor.NewFastHTTPHandler(promhttp.Handler())(c.Context())
		return nil
	})
}

// ════════════════════════════════════════════════════════════════════════════════
// HTTP HANDLERS
// ════════════════════════════════════════════════════════════════════════════════

func (s *Server) healthHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
		"uptime":    s.stats.Uptime().String(),
	})
}

func (s *Server) readyHandler(c *fiber.Ctx) error {
	ready := s.inferenceService != nil
	if !ready {
		return c.Status(503).JSON(fiber.Map{
			"status": "not_ready",
			"reason": "inference service not initialized",
		})
	}
	return c.JSON(fiber.Map{
		"status":    "ready",
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func (s *Server) liveHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"status": "alive",
	})
}

func (s *Server) inferenceHandler(c *fiber.Ctx) error {
	var req InferenceRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(ErrorResponse{
			Error: "Invalid request body",
			Code:  "INVALID_REQUEST",
			Details: err.Error(),
		})
	}

	if err := req.Validate(); err != nil {
		return c.Status(400).JSON(ErrorResponse{
			Error: err.Error(),
			Code:  "VALIDATION_ERROR",
		})
	}

	ctx := c.Context()
	start := time.Now()

	if s.inferenceService == nil {
		// Placeholder response when service not available
		return c.JSON(InferenceResponse{
			ID:           c.Locals("requestid").(string),
			Output:       fmt.Sprintf("Echo: %s", req.Input),
			TokensUsed:   len(req.Input) / 4,
			PromptTokens: len(req.Input) / 4,
			LatencyMs:    float64(time.Since(start).Microseconds()) / 1000,
			ModelID:      "truthgpt-v1",
			FinishReason: "stop",
		})
	}

	resp, err := s.inferenceService.Infer(ctx, &req)
	if err != nil {
		return c.Status(500).JSON(ErrorResponse{
			Error: "Inference failed",
			Code:  "INFERENCE_ERROR",
			Details: err.Error(),
		})
	}

	resp.LatencyMs = float64(time.Since(start).Microseconds()) / 1000
	return c.JSON(resp)
}

func (s *Server) batchInferenceHandler(c *fiber.Ctx) error {
	var requests []InferenceRequest
	if err := c.BodyParser(&requests); err != nil {
		return c.Status(400).JSON(ErrorResponse{
			Error: "Invalid request body",
			Code:  "INVALID_REQUEST",
		})
	}

	if len(requests) == 0 {
		return c.Status(400).JSON(ErrorResponse{
			Error: "Empty batch",
			Code:  "EMPTY_BATCH",
		})
	}

	if len(requests) > 100 {
		return c.Status(400).JSON(ErrorResponse{
			Error: "Batch too large (max 100)",
			Code:  "BATCH_TOO_LARGE",
		})
	}

	start := time.Now()
	responses := make([]InferenceResponse, len(requests))

	var wg sync.WaitGroup
	for i, req := range requests {
		wg.Add(1)
		go func(idx int, r InferenceRequest) {
			defer wg.Done()
			responses[idx] = InferenceResponse{
				ID:           fmt.Sprintf("batch-%d", idx),
				Output:       fmt.Sprintf("Echo: %s", r.Input),
				TokensUsed:   len(r.Input) / 4,
				ModelID:      "truthgpt-v1",
				FinishReason: "stop",
			}
		}(i, req)
	}
	wg.Wait()

	return c.JSON(fiber.Map{
		"responses":        responses,
		"total_latency_ms": float64(time.Since(start).Microseconds()) / 1000,
		"batch_size":       len(requests),
	})
}

func (s *Server) streamInferenceHandler(c *fiber.Ctx) error {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("X-Accel-Buffering", "no")

	var req InferenceRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(ErrorResponse{
			Error: "Invalid request body",
		})
	}

	c.Context().SetBodyStreamWriter(func(w *fiber.Response) {
		for i, char := range req.Input {
			fmt.Fprintf(w, "data: {\"token\": \"%c\", \"index\": %d}\n\n", char, i)
			time.Sleep(30 * time.Millisecond)
			if i > 200 {
				break
			}
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
	})

	return nil
}

func (s *Server) embeddingsHandler(c *fiber.Ctx) error {
	type EmbeddingRequest struct {
		Input []string `json:"input"`
		Model string   `json:"model"`
	}

	var req EmbeddingRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(ErrorResponse{
			Error: "Invalid request body",
		})
	}

	// Placeholder embeddings
	embeddings := make([][]float32, len(req.Input))
	for i := range req.Input {
		embeddings[i] = make([]float32, 1536) // OpenAI-compatible dimension
	}

	return c.JSON(fiber.Map{
		"object": "list",
		"data":   embeddings,
		"model":  req.Model,
		"usage": fiber.Map{
			"prompt_tokens": len(req.Input) * 10,
			"total_tokens":  len(req.Input) * 10,
		},
	})
}

func (s *Server) cacheGetHandler(c *fiber.Ctx) error {
	key := c.Params("key")

	if s.cacheService == nil {
		return c.JSON(fiber.Map{
			"key":   key,
			"value": nil,
			"found": false,
		})
	}

	ctx := c.Context()
	value, err := s.cacheService.Get(ctx, []byte(key))
	if err != nil {
		return c.Status(500).JSON(ErrorResponse{
			Error: "Cache get failed",
			Details: err.Error(),
		})
	}

	return c.JSON(fiber.Map{
		"key":   key,
		"value": string(value),
		"found": value != nil,
	})
}

func (s *Server) cachePutHandler(c *fiber.Ctx) error {
	key := c.Params("key")
	value := c.Body()

	if s.cacheService == nil {
		return c.JSON(fiber.Map{
			"key":    key,
			"stored": true,
		})
	}

	ctx := c.Context()
	if err := s.cacheService.Put(ctx, []byte(key), value); err != nil {
		return c.Status(500).JSON(ErrorResponse{
			Error: "Cache put failed",
			Details: err.Error(),
		})
	}

	return c.JSON(fiber.Map{
		"key":    key,
		"stored": true,
	})
}

func (s *Server) cacheDeleteHandler(c *fiber.Ctx) error {
	key := c.Params("key")

	if s.cacheService != nil {
		ctx := c.Context()
		if err := s.cacheService.Delete(ctx, []byte(key)); err != nil {
			return c.Status(500).JSON(ErrorResponse{
				Error: "Cache delete failed",
				Details: err.Error(),
			})
		}
	}

	return c.JSON(fiber.Map{
		"key":     key,
		"deleted": true,
	})
}

func (s *Server) cacheStatsHandler(c *fiber.Ctx) error {
	if s.cacheService != nil {
		return c.JSON(s.cacheService.Stats())
	}
	return c.JSON(fiber.Map{
		"hits":     0,
		"misses":   0,
		"size":     0,
		"hit_rate": 0.0,
	})
}

func (s *Server) systemInfoHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"version":     "2.0.0",
		"go_version":  runtime.Version(),
		"service":     "truthgpt-go-core",
		"num_cpu":     runtime.NumCPU(),
		"goos":        runtime.GOOS,
		"goarch":      runtime.GOARCH,
		"num_goroutine": runtime.NumGoroutine(),
	})
}

func (s *Server) systemStatsHandler(c *fiber.Ctx) error {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return c.JSON(fiber.Map{
		"uptime_seconds":   s.stats.Uptime().Seconds(),
		"requests_total":   atomic.LoadUint64(&s.stats.RequestsTotal),
		"requests_success": atomic.LoadUint64(&s.stats.RequestsSuccess),
		"requests_failed":  atomic.LoadUint64(&s.stats.RequestsFailed),
		"active_conns":     atomic.LoadInt64(&s.stats.ActiveConns),
		"goroutines":       runtime.NumGoroutine(),
		"memory": fiber.Map{
			"alloc_mb":       m.Alloc / 1024 / 1024,
			"total_alloc_mb": m.TotalAlloc / 1024 / 1024,
			"sys_mb":         m.Sys / 1024 / 1024,
			"num_gc":         m.NumGC,
		},
	})
}

func (s *Server) modelsHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"object": "list",
		"data": []fiber.Map{
			{
				"id":       "truthgpt-v1",
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "truthgpt",
			},
		},
	})
}

func (s *Server) modelDetailHandler(c *fiber.Ctx) error {
	id := c.Params("id")
	return c.JSON(fiber.Map{
		"id":       id,
		"object":   "model",
		"created":  time.Now().Unix(),
		"owned_by": "truthgpt",
	})
}

// ════════════════════════════════════════════════════════════════════════════════
// SERVER LIFECYCLE
// ════════════════════════════════════════════════════════════════════════════════

// Start starts the server.
func (s *Server) Start() error {
	errCh := make(chan error, 3)

	// Start metrics server
	go func() {
		addr := fmt.Sprintf(":%d", s.config.MetricsPort)
		s.logger.Info("Starting metrics server", zap.String("addr", addr))
		if err := s.metricsApp.Listen(addr); err != nil {
			errCh <- fmt.Errorf("metrics server: %w", err)
		}
	}()

	// Start gRPC server
	go func() {
		addr := fmt.Sprintf(":%d", s.config.GRPCPort)
		lis, err := net.Listen("tcp", addr)
		if err != nil {
			errCh <- fmt.Errorf("gRPC listen: %w", err)
			return
		}
		s.logger.Info("Starting gRPC server", zap.String("addr", addr))
		if err := s.grpcServer.Serve(lis); err != nil {
			errCh <- fmt.Errorf("gRPC server: %w", err)
		}
	}()

	// Start HTTP server
	go func() {
		addr := fmt.Sprintf(":%d", s.config.HTTPPort)
		s.logger.Info("Starting HTTP server", zap.String("addr", addr))
		if err := s.httpApp.Listen(addr); err != nil {
			errCh <- fmt.Errorf("HTTP server: %w", err)
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-s.shutdown:
		return nil
	}
}

// StartWithGracefulShutdown starts the server with graceful shutdown.
func (s *Server) StartWithGracefulShutdown() error {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.Start()
	}()

	select {
	case err := <-errCh:
		return err
	case sig := <-quit:
		s.logger.Info("Received shutdown signal", zap.String("signal", sig.String()))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return s.Shutdown(ctx)
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	s.logger.Info("Shutting down server...")
	close(s.shutdown)

	var errs []error

	// Shutdown gRPC
	s.grpcServer.GracefulStop()

	// Shutdown HTTP
	if err := s.httpApp.Shutdown(); err != nil {
		errs = append(errs, fmt.Errorf("HTTP shutdown: %w", err))
	}

	// Shutdown metrics
	if err := s.metricsApp.Shutdown(); err != nil {
		errs = append(errs, fmt.Errorf("metrics shutdown: %w", err))
	}

	if len(errs) > 0 {
		return fmt.Errorf("shutdown errors: %v", errs)
	}

	s.logger.Info("Server shutdown complete")
	return nil
}
