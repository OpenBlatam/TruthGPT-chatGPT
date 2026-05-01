"""
Compliance and audit utilities for polyglot_core.

Provides compliance checking, audit logging, and governance.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ComplianceLevel(str, Enum):
    """Compliance level."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceCheck:
    """Compliance check result."""
    check_id: str
    name: str
    description: str
    level: ComplianceLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AuditLog:
    """Audit log entry."""
    event_type: str
    user: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None


class ComplianceChecker:
    """
    Compliance checker for polyglot_core.
    
    Checks compliance with policies and standards.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._results: List[ComplianceCheck] = []
    
    def register_check(self, check_id: str, name: str, check_fn: Callable[[], ComplianceCheck]):
        """
        Register compliance check.
        
        Args:
            check_id: Unique check ID
            name: Check name
            check_fn: Check function
        """
        self._checks[check_id] = {'name': name, 'fn': check_fn}
    
    def run_check(self, check_id: str) -> ComplianceCheck:
        """
        Run a specific check.
        
        Args:
            check_id: Check ID
            
        Returns:
            Compliance check result
        """
        if check_id not in self._checks:
            return ComplianceCheck(
                check_id=check_id,
                name="Unknown",
                description="Check not found",
                level=ComplianceLevel.FAIL,
                message=f"Check {check_id} not registered"
            )
        
        check_info = self._checks[check_id]
        result = check_info['fn']()
        self._results.append(result)
        return result
    
    def run_all_checks(self) -> List[ComplianceCheck]:
        """Run all registered checks."""
        results = []
        for check_id in self._checks:
            result = self.run_check(check_id)
            results.append(result)
        return results
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        if not self._results:
            return {'status': 'unknown', 'total': 0}
        
        recent_results = self._results[-100:] if len(self._results) > 100 else self._results
        
        pass_count = sum(1 for r in recent_results if r.level == ComplianceLevel.PASS)
        warning_count = sum(1 for r in recent_results if r.level == ComplianceLevel.WARNING)
        fail_count = sum(1 for r in recent_results if r.level == ComplianceLevel.FAIL)
        
        total = len(recent_results)
        
        if fail_count > 0:
            status = "non_compliant"
        elif warning_count > 0:
            status = "warning"
        else:
            status = "compliant"
        
        return {
            'status': status,
            'total': total,
            'pass': pass_count,
            'warning': warning_count,
            'fail': fail_count,
            'compliance_rate': (pass_count / total * 100) if total > 0 else 0
        }


class AuditLogger:
    """
    Audit logger for polyglot_core.
    
    Logs all security-relevant events.
    """
    
    def __init__(self, max_entries: int = 100000):
        """
        Initialize audit logger.
        
        Args:
            max_entries: Maximum number of audit log entries
        """
        self.max_entries = max_entries
        self._logs: List[AuditLog] = []
    
    def log(
        self,
        event_type: str,
        user: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """
        Log audit event.
        
        Args:
            event_type: Event type
            user: User identifier
            resource: Resource identifier
            action: Action performed
            result: Result (success/failure)
            details: Additional details
            ip_address: IP address
        """
        log_entry = AuditLog(
            event_type=event_type,
            user=user,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            ip_address=ip_address
        )
        
        self._logs.append(log_entry)
        
        # Keep only recent logs
        if len(self._logs) > self.max_entries:
            self._logs = self._logs[-self.max_entries:]
    
    def query(
        self,
        event_type: Optional[str] = None,
        user: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Query audit logs.
        
        Args:
            event_type: Filter by event type
            user: Filter by user
            resource: Filter by resource
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results
            
        Returns:
            List of matching audit logs
        """
        results = self._logs
        
        if event_type:
            results = [r for r in results if r.event_type == event_type]
        
        if user:
            results = [r for r in results if r.user == user]
        
        if resource:
            results = [r for r in results if r.resource == resource]
        
        if start_time:
            results = [r for r in results if r.timestamp >= start_time]
        
        if end_time:
            results = [r for r in results if r.timestamp <= end_time]
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def export(self, format: str = "json") -> str:
        """
        Export audit logs.
        
        Args:
            format: Export format (json)
            
        Returns:
            Exported logs as string
        """
        if format == "json":
            logs_data = [
                {
                    'event_type': log.event_type,
                    'user': log.user,
                    'resource': log.resource,
                    'action': log.action,
                    'result': log.result,
                    'timestamp': log.timestamp.isoformat(),
                    'details': log.details
                }
                for log in self._logs
            ]
            return json.dumps(logs_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global managers
_global_compliance_checker = ComplianceChecker()
_global_audit_logger = AuditLogger()


def get_compliance_checker() -> ComplianceChecker:
    """Get global compliance checker."""
    return _global_compliance_checker


def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    return _global_audit_logger


def log_audit_event(
    event_type: str,
    user: Optional[str] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
    **kwargs
):
    """Convenience function to log audit event."""
    _global_audit_logger.log(event_type, user, resource, action, **kwargs)


