"""
Base Advanced System for TruthGPT
Consolidates monitoring, metrics, and boilerplate for Ultra-Advanced systems.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class BaseAdvancedMetrics(BaseModel):
    """Base metrics for all advanced systems."""
    throughput: float = 0.0
    efficiency: float = 0.0
    stability: float = 1.0
    execution_time: float = 0.0
    last_update: float = Field(default_factory=time.time)

class BaseAdvancedSystem:
    """
    Base class for Ultra-Advanced systems to reduce redundancy.
    Handles monitoring threads, baseline metrics, and lifecycle.
    """
    
    def __init__(self, config: Any, system_name: str):
        self.config = config
        self.system_name = system_name
        self.logger = logging.getLogger(f"{system_name}_{id(self)}")
        
        self.metrics = BaseAdvancedMetrics()
        self.history: List[Dict[str, Any]] = []
        self._is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start background monitoring."""
        if self._is_running:
            return
        self._is_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"Monitoring started for {self.system_name}")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info(f"Monitoring stopped for {self.system_name}")

    def _monitor_loop(self):
        """Standard monitoring loop."""
        interval = getattr(self.config, 'monitoring_interval', 1.0)
        while self._is_running:
            try:
                self.update_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error in {self.system_name}: {e}")
                break

    def update_metrics(self):
        """To be overridden by subclasses."""
        pass

    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a system event with timestamp."""
        record = {
            "timestamp": time.time(),
            "event": event_type,
            "data": data,
            "system": self.system_name
        }
        self.history.append(record)
        if len(self.history) > 1000:
            self.history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get standard system statistics."""
        return {
            "system": self.system_name,
            "metrics": self.metrics.model_dump(),
            "history_size": len(self.history),
            "is_running": self._is_running
        }
