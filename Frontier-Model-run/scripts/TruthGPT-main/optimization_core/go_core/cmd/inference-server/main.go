// Package main implements the TruthGPT inference server.
//
// This is the main entry point for the high-performance inference service,
// providing both HTTP REST and gRPC interfaces for model inference.
//
// Usage:
//
//	inference-server [flags]
//
// Flags:
//
//	-config string     Path to configuration file (default "config/default.yaml")
//	-http-port int     HTTP port (default 8080)
//	-grpc-port int     gRPC port (default 50051)
//	-metrics-port int  Metrics port (default 9090)
//	-cache-path string Path for cache storage (default "/tmp/truthgpt_cache")
//	-nats-url string   NATS server URL (default "nats://localhost:4222")
//	-workers int       Number of inference workers (default: NumCPU)
//	-batch-size int    Maximum batch size (default 32)
//	-prefork bool      Enable prefork mode (default false)
//	-debug bool        Enable debug logging (default false)
//	-version           Print version and exit
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/spf13/viper"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/truthgpt/optimization_core/go_core/internal/cache"
	"github.com/truthgpt/optimization_core/go_core/internal/inference"
	"github.com/truthgpt/optimization_core/go_core/internal/messaging"
	"github.com/truthgpt/optimization_core/go_core/internal/server"
)

// Build information (set via ldflags)
var (
	Version   = "2.0.0"
	GitCommit = "unknown"
	BuildTime = "unknown"
	GoVersion = runtime.Version()
)

// Configuration structure
type Config struct {
	Server    server.Config    `yaml:"server"`
	Cache     cache.Config     `yaml:"cache"`
	Messaging messaging.Config `yaml:"messaging"`
	Inference InferenceConfig  `yaml:"inference"`
}

type InferenceConfig struct {
	Workers      int           `yaml:"workers"`
	BatchSize    int           `yaml:"batch_size"`
	BatchTimeout time.Duration `yaml:"batch_timeout"`
	MaxTokens    int           `yaml:"max_tokens"`
	ModelPath    string        `yaml:"model_path"`
}

func main() {
	// Parse command-line flags
	var (
		configPath  = flag.String("config", "", "Path to configuration file")
		httpPort    = flag.Int("http-port", 8080, "HTTP port")
		grpcPort    = flag.Int("grpc-port", 50051, "gRPC port")
		metricsPort = flag.Int("metrics-port", 9090, "Metrics port")
		cachePath   = flag.String("cache-path", "/tmp/truthgpt_cache", "Cache storage path")
		natsURL     = flag.String("nats-url", "nats://localhost:4222", "NATS server URL")
		workers     = flag.Int("workers", runtime.NumCPU(), "Number of inference workers")
		batchSize   = flag.Int("batch-size", 32, "Maximum batch size")
		prefork     = flag.Bool("prefork", false, "Enable prefork mode")
		debug       = flag.Bool("debug", false, "Enable debug logging")
		version     = flag.Bool("version", false, "Print version and exit")
	)
	flag.Parse()

	// Print version and exit if requested
	if *version {
		printVersion()
		os.Exit(0)
	}

	// Setup logger
	logger := setupLogger(*debug)
	defer logger.Sync()

	// Print startup banner
	printBanner(logger)

	// Load configuration
	config := loadConfig(*configPath, logger)

	// Override config with command-line flags
	if *httpPort != 8080 {
		config.Server.HTTPPort = *httpPort
	}
	if *grpcPort != 50051 {
		config.Server.GRPCPort = *grpcPort
	}
	if *metricsPort != 9090 {
		config.Server.MetricsPort = *metricsPort
	}
	if *cachePath != "/tmp/truthgpt_cache" {
		config.Cache.BadgerPath = *cachePath
	}
	if *natsURL != "nats://localhost:4222" {
		config.Messaging.URLs = []string{*natsURL}
	}
	if *workers != runtime.NumCPU() {
		config.Inference.Workers = *workers
	}
	if *batchSize != 32 {
		config.Inference.BatchSize = *batchSize
	}
	config.Server.Prefork = *prefork

	// Setup OpenTelemetry metrics
	setupMetrics(logger)

	// Log configuration
	logger.Info("Configuration loaded",
		zap.Int("http_port", config.Server.HTTPPort),
		zap.Int("grpc_port", config.Server.GRPCPort),
		zap.Int("metrics_port", config.Server.MetricsPort),
		zap.Int("workers", config.Inference.Workers),
		zap.Int("batch_size", config.Inference.BatchSize),
		zap.String("cache_path", config.Cache.BadgerPath),
	)

	// Initialize cache
	kvCache, err := initCache(config.Cache, logger)
	if err != nil {
		logger.Fatal("Failed to initialize cache", zap.Error(err))
	}
	defer kvCache.Close()
	logger.Info("Cache initialized",
		zap.Int("shards", config.Cache.NumShards),
		zap.String("path", config.Cache.BadgerPath),
	)

	// Initialize messaging (optional)
	var natsClient *messaging.Client
	if len(config.Messaging.URLs) > 0 {
		natsClient, err = initMessaging(config.Messaging, logger)
		if err != nil {
			logger.Warn("Failed to initialize messaging, continuing without it", zap.Error(err))
		} else {
			defer natsClient.Close()
			logger.Info("Messaging initialized",
				zap.Strings("urls", config.Messaging.URLs),
				zap.Bool("jetstream", config.Messaging.EnableJetStream),
			)
		}
	}

	// Initialize batch processor
	batchProcessor, err := initBatchProcessor(config.Inference, logger)
	if err != nil {
		logger.Fatal("Failed to initialize batch processor", zap.Error(err))
	}
	defer batchProcessor.Shutdown(context.Background())
	logger.Info("Batch processor initialized",
		zap.Int("workers", config.Inference.Workers),
		zap.Int("batch_size", config.Inference.BatchSize),
	)

	// Create server
	srv := server.New(config.Server, logger)

	// Wire up services
	srv.SetCacheService(NewCacheServiceAdapter(kvCache))
	srv.SetInferenceService(NewInferenceServiceAdapter(batchProcessor, logger))

	logger.Info("Starting TruthGPT Inference Server",
		zap.String("version", Version),
		zap.String("go_version", GoVersion),
		zap.Int("num_cpu", runtime.NumCPU()),
		zap.Int("gomaxprocs", runtime.GOMAXPROCS(0)),
	)

	// Start server with graceful shutdown
	if err := srv.StartWithGracefulShutdown(); err != nil {
		logger.Fatal("Server error", zap.Error(err))
	}

	logger.Info("Server stopped gracefully")
}

// ════════════════════════════════════════════════════════════════════════════════
// INITIALIZATION FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════════

func setupLogger(debug bool) *zap.Logger {
	var config zap.Config

	if debug {
		config = zap.NewDevelopmentConfig()
		config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
		config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	} else {
		config = zap.NewProductionConfig()
		config.EncoderConfig.TimeKey = "timestamp"
		config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	}

	logger, err := config.Build()
	if err != nil {
		panic(fmt.Sprintf("Failed to create logger: %v", err))
	}

	return logger
}

func loadConfig(path string, logger *zap.Logger) Config {
	config := Config{
		Server:    server.DefaultConfig(),
		Cache:     cache.DefaultConfig(),
		Messaging: messaging.DefaultConfig(),
		Inference: InferenceConfig{
			Workers:      runtime.NumCPU(),
			BatchSize:    32,
			BatchTimeout: 50 * time.Millisecond,
			MaxTokens:    4096,
		},
	}

	if path != "" {
		viper.SetConfigFile(path)
		if err := viper.ReadInConfig(); err != nil {
			logger.Warn("Failed to read config file, using defaults", zap.Error(err))
		} else {
			if err := viper.Unmarshal(&config); err != nil {
				logger.Warn("Failed to unmarshal config, using defaults", zap.Error(err))
			}
		}
	}

	// Environment variable overrides
	viper.AutomaticEnv()
	viper.SetEnvPrefix("TRUTHGPT")

	return config
}

func setupMetrics(logger *zap.Logger) {
	exporter, err := prometheus.New()
	if err != nil {
		logger.Warn("Failed to create Prometheus exporter", zap.Error(err))
		return
	}

	provider := metric.NewMeterProvider(metric.WithReader(exporter))
	otel.SetMeterProvider(provider)

	logger.Info("OpenTelemetry metrics initialized")
}

func initCache(config cache.Config, logger *zap.Logger) (*cache.ShardedKVCache, error) {
	return cache.New(config, logger)
}

func initMessaging(config messaging.Config, logger *zap.Logger) (*messaging.Client, error) {
	return messaging.NewClient(config, logger)
}

func initBatchProcessor(config InferenceConfig, logger *zap.Logger) (*inference.BatchProcessor, error) {
	batchConfig := inference.DefaultBatchConfig()
	batchConfig.NumWorkers = config.Workers
	batchConfig.MaxBatchSize = config.BatchSize
	batchConfig.BatchTimeout = config.BatchTimeout

	// Processor function - this is where actual inference would happen
	processor := func(ctx context.Context, batch *inference.Batch) ([]*inference.InferenceResponse, error) {
		responses := make([]*inference.InferenceResponse, len(batch.Requests))

		for i, req := range batch.Requests {
			// Placeholder inference logic
			// In production, this would call the actual model
			responses[i] = &inference.InferenceResponse{
				RequestID:    req.ID,
				Output:       fmt.Sprintf("Response to: %s", req.Input),
				TokensUsed:   len(req.Input) / 4,
				FinishReason: "stop",
				CompletedAt:  time.Now(),
			}
		}

		return responses, nil
	}

	return inference.NewBatchProcessor(batchConfig, processor, logger)
}

// ════════════════════════════════════════════════════════════════════════════════
// SERVICE ADAPTERS
// ════════════════════════════════════════════════════════════════════════════════

// CacheServiceAdapter adapts ShardedKVCache to server.CacheService
type CacheServiceAdapter struct {
	cache *cache.ShardedKVCache
}

func NewCacheServiceAdapter(c *cache.ShardedKVCache) *CacheServiceAdapter {
	return &CacheServiceAdapter{cache: c}
}

func (a *CacheServiceAdapter) Get(ctx context.Context, key []byte) ([]byte, error) {
	return a.cache.Get(ctx, key)
}

func (a *CacheServiceAdapter) Put(ctx context.Context, key, value []byte) error {
	return a.cache.Put(ctx, key, value)
}

func (a *CacheServiceAdapter) Delete(ctx context.Context, key []byte) error {
	return a.cache.Delete(ctx, key)
}

func (a *CacheServiceAdapter) Stats() interface{} {
	return a.cache.Stats()
}

// InferenceServiceAdapter adapts BatchProcessor to server.InferenceService
type InferenceServiceAdapter struct {
	processor *inference.BatchProcessor
	logger    *zap.Logger
}

func NewInferenceServiceAdapter(p *inference.BatchProcessor, logger *zap.Logger) *InferenceServiceAdapter {
	return &InferenceServiceAdapter{processor: p, logger: logger}
}

func (a *InferenceServiceAdapter) Infer(ctx context.Context, req *server.InferenceRequest) (*server.InferenceResponse, error) {
	inferReq := inference.NewInferenceRequest(req.Input)
	inferReq.MaxTokens = req.MaxTokens
	inferReq.Temperature = float32(req.Temperature)
	inferReq.TopP = float32(req.TopP)
	inferReq.TopK = req.TopK

	resp, err := a.processor.SubmitAndWait(ctx, inferReq)
	if err != nil {
		return nil, err
	}

	return &server.InferenceResponse{
		ID:           resp.RequestID,
		Output:       resp.Output,
		TokensUsed:   resp.TokensUsed,
		LatencyMs:    resp.LatencyMs,
		ModelID:      "truthgpt-v1",
		FinishReason: resp.FinishReason,
	}, nil
}

func (a *InferenceServiceAdapter) InferStream(ctx context.Context, req *server.InferenceRequest, stream func(chunk string) error) error {
	// Simplified streaming - in production, this would stream actual tokens
	for _, char := range req.Input {
		if err := stream(string(char)); err != nil {
			return err
		}
		time.Sleep(30 * time.Millisecond)
	}
	return nil
}

// ════════════════════════════════════════════════════════════════════════════════
// BANNER & VERSION
// ════════════════════════════════════════════════════════════════════════════════

func printVersion() {
	fmt.Printf(`TruthGPT Go Core Inference Server
  Version:    %s
  Git Commit: %s
  Build Time: %s
  Go Version: %s
  OS/Arch:    %s/%s
`, Version, GitCommit, BuildTime, GoVersion, runtime.GOOS, runtime.GOARCH)
}

func printBanner(logger *zap.Logger) {
	banner := `
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ██████╗ ████████╗ ║
║   ╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║██╔════╝ ██╔══██╗╚══██╔══╝ ║
║      ██║   ██████╔╝██║   ██║   ██║   ███████║██║  ███╗██████╔╝   ██║    ║
║      ██║   ██╔══██╗██║   ██║   ██║   ██╔══██║██║   ██║██╔═══╝    ██║    ║
║      ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║╚██████╔╝██║        ██║    ║
║      ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝    ║
║                                                                      ║
║                    Go Core Inference Server v2.0                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
`
	fmt.Println(banner)
}
