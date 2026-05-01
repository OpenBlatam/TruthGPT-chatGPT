// Package main implements the cache service.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/truthgpt/optimization_core/go_core/internal/cache"
)

var (
	Version   = "1.0.0"
	GitCommit = "unknown"
	BuildTime = "unknown"
)

func main() {
	var (
		badgerPath        = flag.String("badger-path", "/tmp/truthgpt_cache", "BadgerDB storage path")
		fastcacheSize     = flag.Int("fastcache-size", 32, "Fastcache size in GB")
		port              = flag.Int("port", 8081, "HTTP port")
		debug             = flag.Bool("debug", false, "Enable debug logging")
		version           = flag.Bool("version", false, "Print version and exit")
	)
	flag.Parse()
	
	if *version {
		fmt.Printf("TruthGPT Go Core Cache Service\n")
		fmt.Printf("  Version:    %s\n", Version)
		fmt.Printf("  Git Commit: %s\n", GitCommit)
		fmt.Printf("  Build Time: %s\n", BuildTime)
		os.Exit(0)
	}
	
	logger := setupLogger(*debug)
	defer logger.Sync()
	
	logger.Info("Starting TruthGPT Cache Service",
		zap.String("version", Version),
		zap.String("badger_path", *badgerPath),
		zap.Int("fastcache_size_gb", *fastcacheSize),
		zap.Int("port", *port),
	)
	
	// Create cache config
	config := cache.DefaultConfig()
	config.BadgerPath = *badgerPath
	config.FastCacheMaxBytes = *fastcacheSize << 30 // Convert GB to bytes
	
	// Create cache
	kvCache, err := cache.New(config, logger)
	if err != nil {
		logger.Fatal("Failed to create cache", zap.Error(err))
	}
	defer kvCache.Close()
	
	logger.Info("Cache initialized successfully")
	
	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	
	// Simple test operations
	ctx := context.Background()
	
	// Example: Store and retrieve
	testKey := []byte("test-key")
	testValue := []byte("Hello, TruthGPT Cache!")
	
	if err := kvCache.Put(ctx, testKey, testValue); err != nil {
		logger.Error("Failed to put test value", zap.Error(err))
	}
	
	if value, err := kvCache.Get(ctx, testKey); err != nil {
		logger.Error("Failed to get test value", zap.Error(err))
	} else {
		logger.Info("Test retrieval successful", zap.String("value", string(value)))
	}
	
	// Print stats periodically
	stats := kvCache.Stats()
	logger.Info("Cache stats",
		zap.Uint64("hits", stats.Hits),
		zap.Uint64("misses", stats.Misses),
		zap.Float64("hit_rate", stats.HitRate()),
	)
	
	logger.Info("Cache service running. Press Ctrl+C to stop.")
	<-quit
	
	logger.Info("Shutting down cache service...")
}

func setupLogger(debug bool) *zap.Logger {
	var config zap.Config
	
	if debug {
		config = zap.NewDevelopmentConfig()
		config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
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












