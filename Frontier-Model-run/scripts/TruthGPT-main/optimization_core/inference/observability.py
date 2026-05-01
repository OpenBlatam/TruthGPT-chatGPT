"""
🔍 OpenTelemetry Tracing & Structured Logging
Enterprise-grade observability for inference API
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None


@dataclass
class LogContext:
    """Structured log context"""
    request_id: str
    user_id: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    latency_ms: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "model"):
            log_data["model"] = record.model
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ObservabilityManager:
    """Centralized observability manager"""
    
    def __init__(
        self,
        enable_tracing: bool = True,
        enable_structured_logging: bool = True,
        otlp_endpoint: Optional[str] = None
    ):
        self.enable_tracing = enable_tracing and OPENTELEMETRY_AVAILABLE
        self.enable_structured_logging = enable_structured_logging
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT")
        
        # Setup tracing
        if self.enable_tracing:
            self._setup_tracing()
        
        # Setup logging
        if self.enable_structured_logging:
            self._setup_logging()
        
        self.logger = logging.getLogger("inference_api")
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        resource = Resource.create({
            "service.name": "frontier-model-run-inference",
            "service.version": "1.0.0"
        })
        
        provider = TracerProvider(resource=resource)
        
        if self.otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            # Use console exporter for development
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_logging(self):
        """Setup structured logging"""
        logger = logging.getLogger("inference_api")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Add JSON formatter handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span"""
        if self.enable_tracing and OPENTELEMETRY_AVAILABLE:
            span = self.tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                span.end()
        else:
            yield None
    
    def log_request(
        self,
        level: int,
        message: str,
        context: LogContext,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log with structured context"""
        log_extra = {
            "request_id": context.request_id,
            "endpoint": context.endpoint,
        }
        
        if context.user_id:
            log_extra["user_id"] = context.user_id
        if context.model:
            log_extra["model"] = context.model
        if context.latency_ms:
            log_extra["latency_ms"] = context.latency_ms
        if context.status_code:
            log_extra["status_code"] = context.status_code
        if context.error:
            log_extra["error"] = context.error
        
        if extra:
            log_extra.update(extra)
        
        self.logger.log(level, message, extra=log_extra)
    
    def log_inference_start(self, context: LogContext, prompt_length: int):
        """Log inference start"""
        self.log_request(
            logging.INFO,
            "Inference started",
            context,
            {"prompt_length": prompt_length}
        )
    
    def log_inference_complete(
        self,
        context: LogContext,
        output_length: int,
        tokens_generated: int,
        cached: bool = False
    ):
        """Log inference completion"""
        self.log_request(
            logging.INFO,
            "Inference completed",
            context,
            {
                "output_length": output_length,
                "tokens_generated": tokens_generated,
                "cached": cached
            }
        )
    
    def log_error(self, context: LogContext, error: Exception, error_type: str = "unknown"):
        """Log error"""
        context.error = str(error)
        self.log_request(
            logging.ERROR,
            f"Inference error: {error_type}",
            context,
            {"error_type": error_type, "error_details": str(error)}
        )
    
    def log_rate_limit(self, context: LogContext, retry_after: int):
        """Log rate limit hit"""
        self.log_request(
            logging.WARNING,
            "Rate limit exceeded",
            context,
            {"retry_after": retry_after}
        )
    
    def log_circuit_breaker(self, context: LogContext, model: str, state: str):
        """Log circuit breaker state change"""
        self.log_request(
            logging.WARNING,
            f"Circuit breaker {state} for model {model}",
            context,
            {"circuit_breaker_state": state, "model": model}
        )


# Initialize observability manager
import os
observability_manager = ObservabilityManager(
    enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
    enable_structured_logging=os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT")
)


