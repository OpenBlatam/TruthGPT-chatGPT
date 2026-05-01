// Package messaging provides high-performance messaging using NATS and JetStream.
//
// Architecture:
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │                         NATS Messaging System                               │
// ├─────────────────────────────────────────────────────────────────────────────┤
// │  ┌─────────────────────────────────────────────────────────────────────────┐│
// │  │                    NATS Core (At-Most-Once)                             ││
// │  │  - Pub/Sub for real-time events                                         ││
// │  │  - Request/Reply for sync operations                                    ││
// │  │  - Queue groups for load balancing                                      ││
// │  └─────────────────────────────────────────────────────────────────────────┘│
// │  ┌─────────────────────────────────────────────────────────────────────────┐│
// │  │                    JetStream (Exactly-Once)                             ││
// │  │  - Persistent streams for durability                                    ││
// │  │  - Consumer groups for parallel processing                              ││
// │  │  - Message acknowledgment and redelivery                                ││
// │  │  - Message deduplication                                                ││
// │  └─────────────────────────────────────────────────────────────────────────┘│
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Features:
// - JetStream for persistent messaging with exactly-once delivery
// - Message deduplication
// - Consumer groups with automatic rebalancing
// - Dead letter queues for failed messages
// - Comprehensive metrics and tracing
package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// ════════════════════════════════════════════════════════════════════════════════
// PROMETHEUS METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	messagesPublished = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_nats_messages_published_total",
		Help: "Total number of messages published",
	}, []string{"subject", "stream"})

	messagesReceived = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_nats_messages_received_total",
		Help: "Total number of messages received",
	}, []string{"subject", "consumer"})

	messageLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "truthgpt_nats_message_latency_seconds",
		Help:    "Message processing latency",
		Buckets: prometheus.ExponentialBuckets(0.0001, 2, 15),
	}, []string{"subject"})

	messagesAcked = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_nats_messages_acked_total",
		Help: "Total number of messages acknowledged",
	}, []string{"consumer"})

	messagesNacked = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_nats_messages_nacked_total",
		Help: "Total number of messages negatively acknowledged",
	}, []string{"consumer"})

	connectionStatus = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "truthgpt_nats_connection_status",
		Help: "NATS connection status (1=connected, 0=disconnected)",
	})
)

// ════════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// Config holds the NATS configuration.
type Config struct {
	// Connection
	URLs           []string      `yaml:"urls"`
	ClusterID      string        `yaml:"cluster_id"`
	ClientID       string        `yaml:"client_id"`
	ConnectTimeout time.Duration `yaml:"connect_timeout"`
	ReconnectWait  time.Duration `yaml:"reconnect_wait"`
	MaxReconnects  int           `yaml:"max_reconnects"`

	// JetStream
	EnableJetStream bool   `yaml:"enable_jetstream"`
	JetStreamDomain string `yaml:"jetstream_domain"`

	// Default stream settings
	DefaultStreamReplicas   int           `yaml:"default_stream_replicas"`
	DefaultStreamMaxAge     time.Duration `yaml:"default_stream_max_age"`
	DefaultStreamMaxBytes   int64         `yaml:"default_stream_max_bytes"`
	DefaultStreamMaxMsgs    int64         `yaml:"default_stream_max_msgs"`

	// Consumer settings
	DefaultAckWait         time.Duration `yaml:"default_ack_wait"`
	DefaultMaxDeliver      int           `yaml:"default_max_deliver"`
	DefaultMaxAckPending   int           `yaml:"default_max_ack_pending"`

	// Features
	EnableMetrics   bool `yaml:"enable_metrics"`
	EnableTracing   bool `yaml:"enable_tracing"`
	EnableDedupe    bool `yaml:"enable_dedupe"`
	DedupeWindow    time.Duration `yaml:"dedupe_window"`
}

// DefaultConfig returns the default NATS configuration.
func DefaultConfig() Config {
	return Config{
		URLs:                    []string{nats.DefaultURL},
		ClusterID:              "truthgpt-cluster",
		ClientID:               "truthgpt-client",
		ConnectTimeout:          10 * time.Second,
		ReconnectWait:           1 * time.Second,
		MaxReconnects:           -1,
		EnableJetStream:         true,
		DefaultStreamReplicas:   1,
		DefaultStreamMaxAge:     24 * time.Hour,
		DefaultStreamMaxBytes:   1 << 30, // 1GB
		DefaultStreamMaxMsgs:    1_000_000,
		DefaultAckWait:          30 * time.Second,
		DefaultMaxDeliver:       5,
		DefaultMaxAckPending:    1000,
		EnableMetrics:           true,
		EnableTracing:           true,
		EnableDedupe:            true,
		DedupeWindow:            5 * time.Minute,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// STATS
// ════════════════════════════════════════════════════════════════════════════════

// Stats holds messaging statistics.
type Stats struct {
	MessagesPublished     uint64
	MessagesReceived      uint64
	MessagesAcked         uint64
	MessagesNacked        uint64
	BytesSent             uint64
	BytesReceived         uint64
	Errors                uint64
	Reconnects            uint64
	PendingMessages       int64
	StreamMessagesTotal   int64
	ConsumersActive       int64
}

// ════════════════════════════════════════════════════════════════════════════════
// MESSAGE TYPES
// ════════════════════════════════════════════════════════════════════════════════

// Message represents a message with metadata.
type Message struct {
	ID          string                 `json:"id"`
	Subject     string                 `json:"subject"`
	Data        []byte                 `json:"data"`
	Headers     map[string]string      `json:"headers,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	DedupeKey   string                 `json:"dedupe_key,omitempty"`
	ReplyTo     string                 `json:"reply_to,omitempty"`
}

// NewMessage creates a new message.
func NewMessage(subject string, data []byte) *Message {
	return &Message{
		ID:        generateMsgID(),
		Subject:   subject,
		Data:      data,
		Headers:   make(map[string]string),
		Timestamp: time.Now(),
	}
}

// WithDedupeKey sets the deduplication key.
func (m *Message) WithDedupeKey(key string) *Message {
	m.DedupeKey = key
	return m
}

// WithHeader adds a header.
func (m *Message) WithHeader(key, value string) *Message {
	m.Headers[key] = value
	return m
}

// ════════════════════════════════════════════════════════════════════════════════
// STREAM CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// StreamConfig holds stream configuration.
type StreamConfig struct {
	Name        string        `yaml:"name"`
	Description string        `yaml:"description"`
	Subjects    []string      `yaml:"subjects"`
	Replicas    int           `yaml:"replicas"`
	MaxAge      time.Duration `yaml:"max_age"`
	MaxBytes    int64         `yaml:"max_bytes"`
	MaxMsgs     int64         `yaml:"max_msgs"`
	Retention   string        `yaml:"retention"` // limits, interest, workqueue
	Storage     string        `yaml:"storage"`   // file, memory
	Discard     string        `yaml:"discard"`   // old, new
}

// ConsumerConfig holds consumer configuration.
type ConsumerConfig struct {
	Name           string        `yaml:"name"`
	Durable        string        `yaml:"durable"`
	FilterSubject  string        `yaml:"filter_subject"`
	DeliverPolicy  string        `yaml:"deliver_policy"` // all, last, new, by_start_sequence, by_start_time
	AckPolicy      string        `yaml:"ack_policy"`     // none, all, explicit
	AckWait        time.Duration `yaml:"ack_wait"`
	MaxDeliver     int           `yaml:"max_deliver"`
	MaxAckPending  int           `yaml:"max_ack_pending"`
	MaxWaiting     int           `yaml:"max_waiting"`
	MaxBatch       int           `yaml:"max_batch"`
}

// ════════════════════════════════════════════════════════════════════════════════
// CLIENT
// ════════════════════════════════════════════════════════════════════════════════

// Client wraps NATS connection with JetStream support.
type Client struct {
	config    Config
	conn      *nats.Conn
	js        jetstream.JetStream
	logger    *zap.Logger
	tracer    trace.Tracer

	streams   map[string]jetstream.Stream
	consumers map[string]jetstream.Consumer

	stats     Stats
	mu        sync.RWMutex
	subs      map[string]*nats.Subscription
	closed    atomic.Bool
}

// NewClient creates a new NATS client with JetStream support.
func NewClient(config Config, logger *zap.Logger) (*Client, error) {
	opts := []nats.Option{
		nats.Name(config.ClientID),
		nats.Timeout(config.ConnectTimeout),
		nats.ReconnectWait(config.ReconnectWait),
		nats.MaxReconnects(config.MaxReconnects),
		nats.ReconnectBufSize(8 * 1024 * 1024), // 8MB reconnect buffer
		nats.PingInterval(20 * time.Second),
		nats.MaxPingsOutstanding(3),
	}

	client := &Client{
		config:    config,
		logger:    logger,
		tracer:    otel.Tracer("messaging.Client"),
		streams:   make(map[string]jetstream.Stream),
		consumers: make(map[string]jetstream.Consumer),
		subs:      make(map[string]*nats.Subscription),
	}

	// Connection handlers
	opts = append(opts,
		nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
			logger.Warn("NATS disconnected", zap.Error(err))
			connectionStatus.Set(0)
		}),
		nats.ReconnectHandler(func(nc *nats.Conn) {
			logger.Info("NATS reconnected",
				zap.String("url", nc.ConnectedUrl()),
				zap.Uint64("reconnects", nc.Reconnects),
			)
			atomic.AddUint64(&client.stats.Reconnects, 1)
			connectionStatus.Set(1)
		}),
		nats.ClosedHandler(func(nc *nats.Conn) {
			logger.Info("NATS connection closed")
			connectionStatus.Set(0)
		}),
		nats.ErrorHandler(func(nc *nats.Conn, sub *nats.Subscription, err error) {
			logger.Error("NATS error",
				zap.Error(err),
				zap.String("subject", sub.Subject),
			)
			atomic.AddUint64(&client.stats.Errors, 1)
		}),
	)

	// Connect
	conn, err := nats.Connect(config.URLs[0], opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %w", err)
	}
	client.conn = conn
	connectionStatus.Set(1)

	// Initialize JetStream
	if config.EnableJetStream {
		jsOpts := []jetstream.JetStreamOpt{}
		if config.JetStreamDomain != "" {
			jsOpts = append(jsOpts, jetstream.WithDomain(config.JetStreamDomain))
		}

		js, err := jetstream.New(conn, jsOpts...)
		if err != nil {
			conn.Close()
			return nil, fmt.Errorf("failed to initialize JetStream: %w", err)
		}
		client.js = js
	}

	logger.Info("NATS client initialized",
		zap.Strings("urls", config.URLs),
		zap.Bool("jetstream", config.EnableJetStream),
	)

	return client, nil
}

// ════════════════════════════════════════════════════════════════════════════════
// STREAM MANAGEMENT
// ════════════════════════════════════════════════════════════════════════════════

// CreateStream creates or updates a JetStream stream.
func (c *Client) CreateStream(ctx context.Context, cfg StreamConfig) (jetstream.Stream, error) {
	if c.js == nil {
		return nil, fmt.Errorf("JetStream not enabled")
	}

	retention := jetstream.LimitsPolicy
	switch cfg.Retention {
	case "interest":
		retention = jetstream.InterestPolicy
	case "workqueue":
		retention = jetstream.WorkQueuePolicy
	}

	storage := jetstream.FileStorage
	if cfg.Storage == "memory" {
		storage = jetstream.MemoryStorage
	}

	discard := jetstream.DiscardOld
	if cfg.Discard == "new" {
		discard = jetstream.DiscardNew
	}

	replicas := cfg.Replicas
	if replicas == 0 {
		replicas = c.config.DefaultStreamReplicas
	}

	stream, err := c.js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:        cfg.Name,
		Description: cfg.Description,
		Subjects:    cfg.Subjects,
		Retention:   retention,
		MaxMsgs:     cfg.MaxMsgs,
		MaxBytes:    cfg.MaxBytes,
		MaxAge:      cfg.MaxAge,
		Storage:     storage,
		Replicas:    replicas,
		Discard:     discard,
		Duplicates:  c.config.DedupeWindow,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create stream %s: %w", cfg.Name, err)
	}

	c.mu.Lock()
	c.streams[cfg.Name] = stream
	c.mu.Unlock()

	c.logger.Info("Stream created/updated",
		zap.String("name", cfg.Name),
		zap.Strings("subjects", cfg.Subjects),
	)

	return stream, nil
}

// DeleteStream deletes a stream.
func (c *Client) DeleteStream(ctx context.Context, name string) error {
	if c.js == nil {
		return fmt.Errorf("JetStream not enabled")
	}

	if err := c.js.DeleteStream(ctx, name); err != nil {
		return fmt.Errorf("failed to delete stream %s: %w", name, err)
	}

	c.mu.Lock()
	delete(c.streams, name)
	c.mu.Unlock()

	return nil
}

// GetStream returns a stream by name.
func (c *Client) GetStream(ctx context.Context, name string) (jetstream.Stream, error) {
	if c.js == nil {
		return nil, fmt.Errorf("JetStream not enabled")
	}

	c.mu.RLock()
	stream, ok := c.streams[name]
	c.mu.RUnlock()

	if ok {
		return stream, nil
	}

	stream, err := c.js.Stream(ctx, name)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.streams[name] = stream
	c.mu.Unlock()

	return stream, nil
}

// ════════════════════════════════════════════════════════════════════════════════
// CONSUMER MANAGEMENT
// ════════════════════════════════════════════════════════════════════════════════

// CreateConsumer creates a consumer on a stream.
func (c *Client) CreateConsumer(ctx context.Context, streamName string, cfg ConsumerConfig) (jetstream.Consumer, error) {
	stream, err := c.GetStream(ctx, streamName)
	if err != nil {
		return nil, err
	}

	deliverPolicy := jetstream.DeliverAllPolicy
	switch cfg.DeliverPolicy {
	case "last":
		deliverPolicy = jetstream.DeliverLastPolicy
	case "new":
		deliverPolicy = jetstream.DeliverNewPolicy
	case "last_per_subject":
		deliverPolicy = jetstream.DeliverLastPerSubjectPolicy
	}

	ackPolicy := jetstream.AckExplicitPolicy
	switch cfg.AckPolicy {
	case "none":
		ackPolicy = jetstream.AckNonePolicy
	case "all":
		ackPolicy = jetstream.AckAllPolicy
	}

	ackWait := cfg.AckWait
	if ackWait == 0 {
		ackWait = c.config.DefaultAckWait
	}

	maxDeliver := cfg.MaxDeliver
	if maxDeliver == 0 {
		maxDeliver = c.config.DefaultMaxDeliver
	}

	maxAckPending := cfg.MaxAckPending
	if maxAckPending == 0 {
		maxAckPending = c.config.DefaultMaxAckPending
	}

	consumer, err := stream.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
		Name:          cfg.Name,
		Durable:       cfg.Durable,
		FilterSubject: cfg.FilterSubject,
		DeliverPolicy: deliverPolicy,
		AckPolicy:     ackPolicy,
		AckWait:       ackWait,
		MaxDeliver:    maxDeliver,
		MaxAckPending: maxAckPending,
		MaxWaiting:    cfg.MaxWaiting,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create consumer %s: %w", cfg.Name, err)
	}

	key := fmt.Sprintf("%s:%s", streamName, cfg.Name)
	c.mu.Lock()
	c.consumers[key] = consumer
	c.mu.Unlock()

	c.logger.Info("Consumer created",
		zap.String("stream", streamName),
		zap.String("name", cfg.Name),
		zap.String("filter", cfg.FilterSubject),
	)

	return consumer, nil
}

// ════════════════════════════════════════════════════════════════════════════════
// PUBLISH
// ════════════════════════════════════════════════════════════════════════════════

// Publish publishes a message to a subject (NATS Core).
func (c *Client) Publish(subject string, data []byte) error {
	if c.closed.Load() {
		return fmt.Errorf("client is closed")
	}

	err := c.conn.Publish(subject, data)
	if err != nil {
		atomic.AddUint64(&c.stats.Errors, 1)
		return fmt.Errorf("publish error: %w", err)
	}

	atomic.AddUint64(&c.stats.MessagesPublished, 1)
	atomic.AddUint64(&c.stats.BytesSent, uint64(len(data)))

	if c.config.EnableMetrics {
		messagesPublished.WithLabelValues(subject, "").Inc()
	}

	return nil
}

// PublishMsg publishes a Message with headers and metadata.
func (c *Client) PublishMsg(msg *Message) error {
	if c.closed.Load() {
		return fmt.Errorf("client is closed")
	}

	natsMsg := &nats.Msg{
		Subject: msg.Subject,
		Data:    msg.Data,
		Header:  make(nats.Header),
	}

	// Add headers
	for k, v := range msg.Headers {
		natsMsg.Header.Set(k, v)
	}
	natsMsg.Header.Set("Nats-Msg-Id", msg.ID)
	natsMsg.Header.Set("Msg-Timestamp", msg.Timestamp.Format(time.RFC3339Nano))

	if msg.DedupeKey != "" {
		natsMsg.Header.Set("Nats-Msg-Id", msg.DedupeKey)
	}

	err := c.conn.PublishMsg(natsMsg)
	if err != nil {
		atomic.AddUint64(&c.stats.Errors, 1)
		return fmt.Errorf("publish error: %w", err)
	}

	atomic.AddUint64(&c.stats.MessagesPublished, 1)
	atomic.AddUint64(&c.stats.BytesSent, uint64(len(msg.Data)))

	return nil
}

// PublishAsync publishes a message asynchronously.
func (c *Client) PublishAsync(subject string, data []byte) {
	go func() {
		if err := c.Publish(subject, data); err != nil {
			c.logger.Warn("Async publish failed",
				zap.String("subject", subject),
				zap.Error(err),
			)
		}
	}()
}

// PublishToStream publishes a message to a JetStream stream.
func (c *Client) PublishToStream(ctx context.Context, subject string, data []byte) (*jetstream.PubAck, error) {
	if c.js == nil {
		return nil, fmt.Errorf("JetStream not enabled")
	}
	if c.closed.Load() {
		return nil, fmt.Errorf("client is closed")
	}

	var span trace.Span
	if c.config.EnableTracing {
		ctx, span = c.tracer.Start(ctx, "PublishToStream",
			trace.WithAttributes(attribute.String("subject", subject)))
		defer span.End()
	}

	ack, err := c.js.Publish(ctx, subject, data)
	if err != nil {
		atomic.AddUint64(&c.stats.Errors, 1)
		return nil, fmt.Errorf("JetStream publish error: %w", err)
	}

	atomic.AddUint64(&c.stats.MessagesPublished, 1)
	atomic.AddUint64(&c.stats.BytesSent, uint64(len(data)))

	if c.config.EnableMetrics {
		messagesPublished.WithLabelValues(subject, ack.Stream).Inc()
	}

	return ack, nil
}

// PublishMsgToStream publishes a Message to JetStream with deduplication.
func (c *Client) PublishMsgToStream(ctx context.Context, msg *Message) (*jetstream.PubAck, error) {
	if c.js == nil {
		return nil, fmt.Errorf("JetStream not enabled")
	}

	opts := []jetstream.PublishOpt{}
	if msg.DedupeKey != "" && c.config.EnableDedupe {
		opts = append(opts, jetstream.WithMsgID(msg.DedupeKey))
	}

	ack, err := c.js.Publish(ctx, msg.Subject, msg.Data, opts...)
	if err != nil {
		return nil, fmt.Errorf("JetStream publish error: %w", err)
	}

	atomic.AddUint64(&c.stats.MessagesPublished, 1)

	return ack, nil
}

// Request sends a request and waits for a response.
func (c *Client) Request(ctx context.Context, subject string, data []byte) ([]byte, error) {
	if c.closed.Load() {
		return nil, fmt.Errorf("client is closed")
	}

	timeout := 30 * time.Second
	if deadline, ok := ctx.Deadline(); ok {
		timeout = time.Until(deadline)
	}

	msg, err := c.conn.Request(subject, data, timeout)
	if err != nil {
		atomic.AddUint64(&c.stats.Errors, 1)
		return nil, fmt.Errorf("request error: %w", err)
	}

	atomic.AddUint64(&c.stats.MessagesPublished, 1)
	atomic.AddUint64(&c.stats.MessagesReceived, 1)
	atomic.AddUint64(&c.stats.BytesSent, uint64(len(data)))
	atomic.AddUint64(&c.stats.BytesReceived, uint64(len(msg.Data)))

	return msg.Data, nil
}

// ════════════════════════════════════════════════════════════════════════════════
// SUBSCRIBE
// ════════════════════════════════════════════════════════════════════════════════

// MessageHandler is a callback for received messages.
type MessageHandler func(subject string, data []byte) error

// Subscribe subscribes to a subject (NATS Core).
func (c *Client) Subscribe(subject string, handler MessageHandler) error {
	if c.closed.Load() {
		return fmt.Errorf("client is closed")
	}

	sub, err := c.conn.Subscribe(subject, func(msg *nats.Msg) {
		start := time.Now()

		atomic.AddUint64(&c.stats.MessagesReceived, 1)
		atomic.AddUint64(&c.stats.BytesReceived, uint64(len(msg.Data)))

		if c.config.EnableMetrics {
			messagesReceived.WithLabelValues(msg.Subject, "").Inc()
		}

		if err := handler(msg.Subject, msg.Data); err != nil {
			c.logger.Warn("Handler error",
				zap.String("subject", msg.Subject),
				zap.Error(err))
			atomic.AddUint64(&c.stats.Errors, 1)
		}

		if c.config.EnableMetrics {
			messageLatency.WithLabelValues(msg.Subject).Observe(time.Since(start).Seconds())
		}
	})

	if err != nil {
		return fmt.Errorf("subscribe error: %w", err)
	}

	c.mu.Lock()
	c.subs[subject] = sub
	c.mu.Unlock()

	return nil
}

// QueueSubscribe subscribes to a subject with a queue group.
func (c *Client) QueueSubscribe(subject, queue string, handler MessageHandler) error {
	if c.closed.Load() {
		return fmt.Errorf("client is closed")
	}

	sub, err := c.conn.QueueSubscribe(subject, queue, func(msg *nats.Msg) {
		atomic.AddUint64(&c.stats.MessagesReceived, 1)
		atomic.AddUint64(&c.stats.BytesReceived, uint64(len(msg.Data)))

		if err := handler(msg.Subject, msg.Data); err != nil {
			c.logger.Warn("Handler error",
				zap.String("subject", msg.Subject),
				zap.Error(err))
			atomic.AddUint64(&c.stats.Errors, 1)
		}
	})

	if err != nil {
		return fmt.Errorf("queue subscribe error: %w", err)
	}

	c.mu.Lock()
	c.subs[subject+":"+queue] = sub
	c.mu.Unlock()

	return nil
}

// JetStreamHandler handles JetStream messages with acknowledgment.
type JetStreamHandler func(msg jetstream.Msg) error

// ConsumeFromStream consumes messages from a JetStream consumer.
func (c *Client) ConsumeFromStream(ctx context.Context, streamName, consumerName string, handler JetStreamHandler) error {
	key := fmt.Sprintf("%s:%s", streamName, consumerName)

	c.mu.RLock()
	consumer, ok := c.consumers[key]
	c.mu.RUnlock()

	if !ok {
		return fmt.Errorf("consumer %s not found", key)
	}

	// Start consuming
	cons, err := consumer.Consume(func(msg jetstream.Msg) {
		start := time.Now()

		atomic.AddUint64(&c.stats.MessagesReceived, 1)
		atomic.AddUint64(&c.stats.BytesReceived, uint64(len(msg.Data())))

		if c.config.EnableMetrics {
			messagesReceived.WithLabelValues(msg.Subject(), consumerName).Inc()
		}

		if err := handler(msg); err != nil {
			c.logger.Warn("JetStream handler error",
				zap.String("subject", msg.Subject()),
				zap.Error(err),
			)
			msg.Nak()
			atomic.AddUint64(&c.stats.MessagesNacked, 1)
			if c.config.EnableMetrics {
				messagesNacked.WithLabelValues(consumerName).Inc()
			}
		} else {
			msg.Ack()
			atomic.AddUint64(&c.stats.MessagesAcked, 1)
			if c.config.EnableMetrics {
				messagesAcked.WithLabelValues(consumerName).Inc()
			}
		}

		if c.config.EnableMetrics {
			messageLatency.WithLabelValues(msg.Subject()).Observe(time.Since(start).Seconds())
		}
	})

	if err != nil {
		return fmt.Errorf("failed to start consumer: %w", err)
	}

	// Wait for context cancellation
	<-ctx.Done()
	cons.Stop()

	return nil
}

// FetchMessages fetches a batch of messages from a consumer.
func (c *Client) FetchMessages(ctx context.Context, streamName, consumerName string, batch int) ([]jetstream.Msg, error) {
	key := fmt.Sprintf("%s:%s", streamName, consumerName)

	c.mu.RLock()
	consumer, ok := c.consumers[key]
	c.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("consumer %s not found", key)
	}

	msgs, err := consumer.Fetch(batch, jetstream.FetchMaxWait(5*time.Second))
	if err != nil {
		return nil, fmt.Errorf("fetch error: %w", err)
	}

	var result []jetstream.Msg
	for msg := range msgs.Messages() {
		result = append(result, msg)
		atomic.AddUint64(&c.stats.MessagesReceived, 1)
	}

	return result, msgs.Error()
}

// Unsubscribe unsubscribes from a subject.
func (c *Client) Unsubscribe(subject string) error {
	c.mu.Lock()
	sub, ok := c.subs[subject]
	if ok {
		delete(c.subs, subject)
	}
	c.mu.Unlock()

	if !ok {
		return fmt.Errorf("subscription not found: %s", subject)
	}

	return sub.Unsubscribe()
}

// ════════════════════════════════════════════════════════════════════════════════
// TRAINING SPECIFIC TOPICS
// ════════════════════════════════════════════════════════════════════════════════

const (
	TopicTrainingBatch      = "training.batch"
	TopicTrainingGradient   = "training.gradient"
	TopicTrainingCheckpoint = "training.checkpoint"
	TopicTrainingMetrics    = "training.metrics"
	TopicInferenceRequest   = "inference.request"
	TopicInferenceResponse  = "inference.response"
	TopicCacheInvalidate    = "cache.invalidate"
	TopicModelUpdate        = "model.update"
)

// TrainingMessage represents a training-related message.
type TrainingMessage struct {
	Type      string                 `json:"type"`
	WorkerID  string                 `json:"worker_id"`
	BatchID   string                 `json:"batch_id,omitempty"`
	Epoch     int                    `json:"epoch,omitempty"`
	Step      int                    `json:"step,omitempty"`
	Loss      float64                `json:"loss,omitempty"`
	Metrics   map[string]float64     `json:"metrics,omitempty"`
	Data      []byte                 `json:"data,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// PublishBatch publishes a training batch.
func (c *Client) PublishBatch(ctx context.Context, batchID string, data []byte) error {
	msg := TrainingMessage{
		Type:      "batch",
		BatchID:   batchID,
		Data:      data,
		Timestamp: time.Now(),
	}
	encoded, _ := json.Marshal(msg)
	_, err := c.PublishToStream(ctx, TopicTrainingBatch+"."+batchID, encoded)
	return err
}

// PublishGradient publishes gradients.
func (c *Client) PublishGradient(ctx context.Context, workerID string, data []byte) error {
	msg := TrainingMessage{
		Type:      "gradient",
		WorkerID:  workerID,
		Data:      data,
		Timestamp: time.Now(),
	}
	encoded, _ := json.Marshal(msg)
	_, err := c.PublishToStream(ctx, TopicTrainingGradient+"."+workerID, encoded)
	return err
}

// PublishMetrics publishes training metrics.
func (c *Client) PublishMetrics(ctx context.Context, workerID string, epoch, step int, loss float64, metrics map[string]float64) error {
	msg := TrainingMessage{
		Type:      "metrics",
		WorkerID:  workerID,
		Epoch:     epoch,
		Step:      step,
		Loss:      loss,
		Metrics:   metrics,
		Timestamp: time.Now(),
	}
	encoded, _ := json.Marshal(msg)
	_, err := c.PublishToStream(ctx, TopicTrainingMetrics, encoded)
	return err
}

// SubscribeGradients subscribes to gradient updates from all workers.
func (c *Client) SubscribeGradients(handler MessageHandler) error {
	return c.Subscribe(TopicTrainingGradient+".>", handler)
}

// ════════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ════════════════════════════════════════════════════════════════════════════════

// Stats returns the current messaging statistics.
func (c *Client) Stats() Stats {
	return Stats{
		MessagesPublished:   atomic.LoadUint64(&c.stats.MessagesPublished),
		MessagesReceived:    atomic.LoadUint64(&c.stats.MessagesReceived),
		MessagesAcked:       atomic.LoadUint64(&c.stats.MessagesAcked),
		MessagesNacked:      atomic.LoadUint64(&c.stats.MessagesNacked),
		BytesSent:           atomic.LoadUint64(&c.stats.BytesSent),
		BytesReceived:       atomic.LoadUint64(&c.stats.BytesReceived),
		Errors:              atomic.LoadUint64(&c.stats.Errors),
		Reconnects:          atomic.LoadUint64(&c.stats.Reconnects),
		PendingMessages:     atomic.LoadInt64(&c.stats.PendingMessages),
		ConsumersActive:     int64(len(c.consumers)),
	}
}

// Flush flushes the connection.
func (c *Client) Flush() error {
	return c.conn.Flush()
}

// IsConnected returns whether the client is connected.
func (c *Client) IsConnected() bool {
	return c.conn.IsConnected()
}

// Close closes the client.
func (c *Client) Close() error {
	if c.closed.Swap(true) {
		return nil
	}

	// Unsubscribe all
	c.mu.Lock()
	for _, sub := range c.subs {
		sub.Unsubscribe()
	}
	c.subs = make(map[string]*nats.Subscription)
	c.mu.Unlock()

	// Drain and close
	if err := c.conn.Drain(); err != nil {
		c.logger.Warn("Error draining connection", zap.Error(err))
	}

	c.logger.Info("NATS client closed")
	return nil
}

// ════════════════════════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════════════════════════

func generateMsgID() string {
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), time.Now().Nanosecond())
}
