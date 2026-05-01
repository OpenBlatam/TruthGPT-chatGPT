"""
Monitoring Modules
"""

from .imports import *
from .core import BaseModule

class Alert:
    """Alert class"""
    
    def __init__(self, name: str, condition: Callable, message: str):
        self.name = name
        self.condition = condition
        self.message = message
    
    def check(self, data: Dict[str, Any]) -> bool:
        """Check if alert condition is met"""
        return self.condition(data)

class MonitoringModule(BaseModule):
    """Base monitoring module"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics = {}
        self.alerts = []
    
    def _setup(self):
        """Setup monitoring components"""
        self._setup_metrics()
        self._setup_alerts()
    
    @abstractmethod
    def _setup_metrics(self):
        """Setup metrics"""
        pass
    
    @abstractmethod
    def _setup_alerts(self):
        """Setup alerts"""
        pass
    
    def monitor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor data"""
        results = {}
        
        # Collect metrics
        for name, metric in self.metrics.items():
            try:
                results[name] = metric(data)
            except Exception as e:
                self.logger.warning(f"Could not compute metric {name}: {e}")
                results[name] = None
        
        # Check alerts
        for alert in self.alerts:
            if alert.check(results):
                self.logger.warning(f"Alert triggered: {alert.message}")
                results["alerts"] = results.get("alerts", []) + [alert.message]
        
        return results

class PerformanceMonitoringModule(MonitoringModule):
    """Performance monitoring module"""
    
    def _setup_metrics(self):
        """Setup performance metrics"""
        self.metrics = {
            "cpu_usage": self._get_cpu_usage,
            "memory_usage": self._get_memory_usage,
            "gpu_usage": self._get_gpu_usage,
            "inference_time": self._get_inference_time
        }
    
    def _setup_alerts(self):
        """Setup performance alerts"""
        self.alerts = [
            Alert("high_cpu", lambda x: x.get("cpu_usage", 0) > 90, "High CPU usage"),
            Alert("high_memory", lambda x: x.get("memory_usage", 0) > 90, "High memory usage"),
            Alert("slow_inference", lambda x: x.get("inference_time", 0) > 1.0, "Slow inference")
        ]
    
    def _get_cpu_usage(self, data: Dict[str, Any]) -> float:
        """Get CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self, data: Dict[str, Any]) -> float:
        """Get memory usage"""
        try:
             import psutil
             return psutil.virtual_memory().percent
        except ImportError:
             return 0.0
    
    def _get_gpu_usage(self, data: Dict[str, Any]) -> float:
        """Get GPU usage"""
        if torch.cuda.is_available():
            # Basic approximation or requires pynvml
            return 0.0 
        return 0.0
    
    def _get_inference_time(self, data: Dict[str, Any]) -> float:
        """Get inference time"""
        return data.get("inference_time", 0.0)

def create_monitoring_module(monitoring_type: str, config: Dict[str, Any]) -> MonitoringModule:
    """Create monitoring module"""
    if monitoring_type == "performance":
        return PerformanceMonitoringModule(config)
    else:
        raise ValueError(f"Unknown monitoring type: {monitoring_type}")

