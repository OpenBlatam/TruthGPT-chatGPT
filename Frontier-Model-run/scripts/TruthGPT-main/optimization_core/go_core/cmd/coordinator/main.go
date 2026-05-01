// Package main implements the training coordinator.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/truthgpt/optimization_core/go_core/internal/messaging"
)

var (
	Version   = "1.0.0"
	GitCommit = "unknown"
	BuildTime = "unknown"
)

func main() {
	var (
		natsURL  = flag.String("nats-url", "nats://localhost:4222", "NATS server URL")
		clientID = flag.String("client-id", "truthgpt-coordinator", "NATS client ID")
		debug    = flag.Bool("debug", false, "Enable debug logging")
		version  = flag.Bool("version", false, "Print version and exit")
	)
	flag.Parse()
	
	if *version {
		fmt.Printf("TruthGPT Go Core Coordinator\n")
		fmt.Printf("  Version:    %s\n", Version)
		fmt.Printf("  Git Commit: %s\n", GitCommit)
		fmt.Printf("  Build Time: %s\n", BuildTime)
		os.Exit(0)
	}
	
	logger := setupLogger(*debug)
	defer logger.Sync()
	
	logger.Info("Starting TruthGPT Coordinator",
		zap.String("version", Version),
		zap.String("nats_url", *natsURL),
		zap.String("client_id", *clientID),
	)
	
	// Create NATS config
	config := messaging.DefaultConfig()
	config.URLs = []string{*natsURL}
	config.ClientID = *clientID
	
	// Create NATS client
	client, err := messaging.NewClient(config, logger)
	if err != nil {
		logger.Fatal("Failed to connect to NATS", zap.Error(err))
	}
	defer client.Close()
	
	logger.Info("Connected to NATS successfully")
	
	// Subscribe to gradient updates
	err = client.SubscribeGradients(func(subject string, data []byte) error {
		logger.Debug("Received gradients",
			zap.String("subject", subject),
			zap.Int("size", len(data)),
		)
		// TODO: Aggregate gradients and publish back
		return nil
	})
	if err != nil {
		logger.Fatal("Failed to subscribe to gradients", zap.Error(err))
	}
	
	// Subscribe to metrics
	err = client.Subscribe(messaging.TopicTrainingMetrics, func(subject string, data []byte) error {
		logger.Debug("Received metrics",
			zap.String("subject", subject),
			zap.Int("size", len(data)),
		)
		return nil
	})
	if err != nil {
		logger.Fatal("Failed to subscribe to metrics", zap.Error(err))
	}
	
	logger.Info("Coordinator running. Press Ctrl+C to stop.")
	
	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	
	logger.Info("Shutting down coordinator...")
	
	// Print final stats
	stats := client.Stats()
	logger.Info("Final messaging stats",
		zap.Uint64("messages_published", stats.MessagesPublished),
		zap.Uint64("messages_received", stats.MessagesReceived),
		zap.Uint64("bytes_sent", stats.BytesSent),
		zap.Uint64("bytes_received", stats.BytesReceived),
	)
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












