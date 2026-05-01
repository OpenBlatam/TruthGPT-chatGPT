"""
Cost optimization utilities for polyglot_core.

Provides cost tracking, optimization, and budget management.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class CostEntry:
    """Cost entry."""
    service: str
    resource: str
    cost: float
    unit: str = "USD"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimization:
    """Cost optimization recommendation."""
    service: str
    current_cost: float
    potential_savings: float
    recommendation: str
    priority: str = "medium"  # "low", "medium", "high"
    implementation_effort: str = "low"  # "low", "medium", "high"


class CostTracker:
    """
    Cost tracker for polyglot_core.
    
    Tracks costs across services and resources.
    """
    
    def __init__(self):
        self._costs: List[CostEntry] = []
        self._max_entries: int = 100000
    
    def record_cost(
        self,
        service: str,
        resource: str,
        cost: float,
        unit: str = "USD",
        **metadata
    ):
        """
        Record cost.
        
        Args:
            service: Service name
            resource: Resource identifier
            cost: Cost amount
            unit: Cost unit
            **metadata: Additional metadata
        """
        entry = CostEntry(
            service=service,
            resource=resource,
            cost=cost,
            unit=unit,
            metadata=metadata
        )
        
        self._costs.append(entry)
        
        # Keep only recent entries
        if len(self._costs) > self._max_entries:
            self._costs = self._costs[-self._max_entries:]
    
    def get_total_cost(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        service: Optional[str] = None
    ) -> float:
        """
        Get total cost.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            service: Service filter
            
        Returns:
            Total cost
        """
        costs = self._costs
        
        if start_time:
            costs = [c for c in costs if c.timestamp >= start_time]
        
        if end_time:
            costs = [c for c in costs if c.timestamp <= end_time]
        
        if service:
            costs = [c for c in costs if c.service == service]
        
        return sum(c.cost for c in costs)
    
    def get_cost_by_service(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get costs grouped by service.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Dictionary mapping service to total cost
        """
        costs = self._costs
        
        if start_time:
            costs = [c for c in costs if c.timestamp >= start_time]
        
        if end_time:
            costs = [c for c in costs if c.timestamp <= end_time]
        
        service_costs = defaultdict(float)
        for cost in costs:
            service_costs[cost.service] += cost.cost
        
        return dict(service_costs)
    
    def get_cost_trend(
        self,
        days: int = 30,
        service: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get cost trend over time.
        
        Args:
            days: Number of days
            service: Optional service filter
            
        Returns:
            Dictionary mapping date to cost
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        costs = self._costs
        
        if start_time:
            costs = [c for c in costs if c.timestamp >= start_time]
        
        if service:
            costs = [c for c in costs if c.service == service]
        
        daily_costs = defaultdict(float)
        for cost in costs:
            date_key = cost.timestamp.date().isoformat()
            daily_costs[date_key] += cost.cost
        
        return dict(daily_costs)


class CostOptimizer:
    """
    Cost optimizer for polyglot_core.
    
    Analyzes costs and provides optimization recommendations.
    """
    
    def __init__(self, cost_tracker: CostTracker):
        """
        Initialize cost optimizer.
        
        Args:
            cost_tracker: Cost tracker instance
        """
        self.cost_tracker = cost_tracker
    
    def analyze_costs(self) -> List[CostOptimization]:
        """
        Analyze costs and generate optimization recommendations.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze by service
        service_costs = self.cost_tracker.get_cost_by_service()
        total_cost = sum(service_costs.values())
        
        for service, cost in service_costs.items():
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            
            # Recommend optimization for high-cost services
            if percentage > 30:
                potential_savings = cost * 0.2  # Assume 20% savings possible
                
                recommendations.append(CostOptimization(
                    service=service,
                    current_cost=cost,
                    potential_savings=potential_savings,
                    recommendation=f"Optimize {service} (accounts for {percentage:.1f}% of total cost)",
                    priority="high" if percentage > 50 else "medium",
                    implementation_effort="medium"
                ))
        
        # Analyze trends
        trend = self.cost_tracker.get_cost_trend(days=7)
        if trend:
            recent_costs = list(trend.values())[-7:]
            if len(recent_costs) >= 2:
                avg_recent = sum(recent_costs) / len(recent_costs)
                avg_older = sum(recent_costs[:-3]) / len(recent_costs[:-3]) if len(recent_costs) > 3 else avg_recent
                
                if avg_recent > avg_older * 1.2:  # 20% increase
                    recommendations.append(CostOptimization(
                        service="all",
                        current_cost=avg_recent,
                        potential_savings=(avg_recent - avg_older),
                        recommendation="Costs are increasing - investigate recent changes",
                        priority="high",
                        implementation_effort="low"
                    ))
        
        return recommendations


# Global cost tracker
_global_cost_tracker = CostTracker()
_global_cost_optimizer = CostOptimizer(_global_cost_tracker)


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker."""
    return _global_cost_tracker


def get_cost_optimizer() -> CostOptimizer:
    """Get global cost optimizer."""
    return _global_cost_optimizer


def record_cost(service: str, resource: str, cost: float, **kwargs):
    """Convenience function to record cost."""
    _global_cost_tracker.record_cost(service, resource, cost, **kwargs)













