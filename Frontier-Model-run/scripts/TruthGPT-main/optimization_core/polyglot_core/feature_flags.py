"""
Feature flags for polyglot_core.

Provides feature flag management and A/B testing capabilities.
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random


class FlagType(str, Enum):
    """Feature flag type."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER = "user"
    CUSTOM = "custom"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    flag_type: FlagType
    enabled: bool = True
    percentage: float = 100.0  # 0-100
    user_filter: Optional[Callable[[str], bool]] = None
    custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Feature flag manager for polyglot_core.
    
    Manages feature flags and A/B testing.
    """
    
    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
    
    def register_flag(
        self,
        name: str,
        flag_type: FlagType = FlagType.BOOLEAN,
        enabled: bool = True,
        percentage: float = 100.0,
        user_filter: Optional[Callable[[str], bool]] = None,
        custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **metadata
    ):
        """
        Register a feature flag.
        
        Args:
            name: Flag name
            flag_type: Flag type
            enabled: Whether flag is enabled
            percentage: Percentage for percentage-based flags
            user_filter: User-based filter function
            custom_filter: Custom filter function
            **metadata: Additional metadata
        """
        flag = FeatureFlag(
            name=name,
            flag_type=flag_type,
            enabled=enabled,
            percentage=percentage,
            user_filter=user_filter,
            custom_filter=custom_filter,
            metadata=metadata
        )
        
        self._flags[name] = flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature flag is enabled.
        
        Args:
            flag_name: Flag name
            user_id: Optional user ID
            context: Optional context dictionary
            
        Returns:
            True if enabled, False otherwise
        """
        if flag_name not in self._flags:
            return False
        
        flag = self._flags[flag_name]
        
        if not flag.enabled:
            return False
        
        if flag.flag_type == FlagType.BOOLEAN:
            return True
        
        elif flag.flag_type == FlagType.PERCENTAGE:
            return random.random() * 100 < flag.percentage
        
        elif flag.flag_type == FlagType.USER:
            if user_id and flag.user_filter:
                return flag.user_filter(user_id)
            return False
        
        elif flag.flag_type == FlagType.CUSTOM:
            if flag.custom_filter and context:
                return flag.custom_filter(context)
            return False
        
        return False
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get flag by name."""
        return self._flags.get(flag_name)
    
    def list_flags(self) -> Dict[str, FeatureFlag]:
        """List all flags."""
        return self._flags.copy()
    
    def enable_flag(self, flag_name: str):
        """Enable a flag."""
        if flag_name in self._flags:
            self._flags[flag_name].enabled = True
    
    def disable_flag(self, flag_name: str):
        """Disable a flag."""
        if flag_name in self._flags:
            self._flags[flag_name].enabled = False


# Global feature flag manager
_global_flag_manager = FeatureFlagManager()


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get global feature flag manager."""
    return _global_flag_manager


def is_feature_enabled(
    flag_name: str,
    user_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience function to check feature flag."""
    return _global_flag_manager.is_enabled(flag_name, user_id, context)


def register_feature_flag(
    name: str,
    flag_type: FlagType = FlagType.BOOLEAN,
    enabled: bool = True,
    **kwargs
):
    """Convenience function to register feature flag."""
    _global_flag_manager.register_flag(name, flag_type, enabled, **kwargs)













