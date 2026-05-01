"""
Notification utilities for optimization_core.

Provides utilities for sending notifications.
"""
import logging
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class Notification(BaseModel):
    """Notification data structure."""
    title: str
    message: str
    level: NotificationLevel
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NotificationManager:
    """Manager for notifications."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.handlers: List[Callable[[Notification], None]] = []
        self.notifications: List[Notification] = []
        self.max_history: int = 1000
    
    def register_handler(
        self,
        handler: Callable[[Notification], None]
    ):
        """
        Register a notification handler.
        
        Args:
            handler: Handler function
        """
        self.handlers.append(handler)
        logger.debug("Notification handler registered")
    
    def notify(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send a notification.
        
        Args:
            title: Notification title
            message: Notification message
            level: Notification level
            metadata: Optional metadata
        """
        notification = Notification(
            title=title,
            message=message,
            level=level,
            metadata=metadata or {}
        )
        
        # Add to history
        self.notifications.append(notification)
        if len(self.notifications) > self.max_history:
            self.notifications.pop(0)
        
        # Call handlers
        for handler in self.handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}", exc_info=True)
    
    def info(self, title: str, message: str, **metadata):
        """Send info notification."""
        self.notify(title, message, NotificationLevel.INFO, metadata)
    
    def warning(self, title: str, message: str, **metadata):
        """Send warning notification."""
        self.notify(title, message, NotificationLevel.WARNING, metadata)
    
    def error(self, title: str, message: str, **metadata):
        """Send error notification."""
        self.notify(title, message, NotificationLevel.ERROR, metadata)
    
    def success(self, title: str, message: str, **metadata):
        """Send success notification."""
        self.notify(title, message, NotificationLevel.SUCCESS, metadata)
    
    def get_recent(
        self,
        level: Optional[NotificationLevel] = None,
        limit: int = 100
    ) -> List[Notification]:
        """
        Get recent notifications.
        
        Args:
            level: Filter by level (all if None)
            limit: Maximum number of notifications
        
        Returns:
            List of notifications
        """
        notifications = list(self.notifications)
        
        if level:
            notifications = [n for n in notifications if n.level == level]
        
        return notifications[-limit:]


# Global notification manager
_global_notification_manager = NotificationManager()


def get_notification_manager() -> NotificationManager:
    """Get global notification manager."""
    return _global_notification_manager


def notify(
    title: str,
    message: str,
    level: NotificationLevel = NotificationLevel.INFO,
    **metadata
):
    """
    Send a notification.
    
    Args:
        title: Notification title
        message: Notification message
        level: Notification level
        **metadata: Optional metadata
    """
    _global_notification_manager.notify(title, message, level, metadata)













