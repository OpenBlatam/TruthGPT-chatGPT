"""
Common datetime utilities for optimization_core.

Provides reusable date and time manipulation functions.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import dateutil for flexible parsing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    logger.debug("python-dateutil not available, using basic date parsing")


# ════════════════════════════════════════════════════════════════════════════════
# CURRENT TIME
# ════════════════════════════════════════════════════════════════════════════════

def now_utc() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current UTC datetime
    
    Example:
        >>> dt = now_utc()
        >>> dt.tzinfo == timezone.utc
        True
    """
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    """
    Get current local datetime.
    
    Returns:
        Current local datetime
    """
    return datetime.now()


def now_iso() -> str:
    """
    Get current UTC datetime as ISO string.
    
    Returns:
        ISO format string
    
    Example:
        >>> iso = now_iso()
        >>> 'T' in iso
        True
    """
    return now_utc().isoformat()


# ════════════════════════════════════════════════════════════════════════════════
# PARSING
# ════════════════════════════════════════════════════════════════════════════════

def parse_datetime(
    date_string: str,
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parse date string to datetime (flexible parsing with fallback).
    
    Args:
        date_string: Date string to parse (must be a string)
        default: Default value if parsing fails (default: None)
    
    Returns:
        Parsed datetime or default if parsing fails
    
    Raises:
        TypeError: If date_string is not a string
    
    Examples:
        >>> parse_datetime("2024-01-01T12:00:00Z")
        datetime.datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        >>> parse_datetime("invalid", default=datetime(2024, 1, 1))
        datetime.datetime(2024, 1, 1, 0, 0)
    """
    # Validate input
    if not isinstance(date_string, str):
        raise TypeError(f"date_string must be a string, got {type(date_string).__name__}")
    
    # Handle empty string
    if not date_string:
        return default
    
    try:
        if HAS_DATEUTIL:
            # Use dateutil for flexible parsing (handles many formats)
            return date_parser.parse(date_string)
        else:
            # Basic ISO format parsing (fallback when dateutil not available)
            # Replace 'Z' with '+00:00' for ISO format compatibility
            normalized_string = date_string.replace('Z', '+00:00')
            return datetime.fromisoformat(normalized_string)
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Failed to parse date '{date_string}': {e}")
        return default


def parse_datetime_format(
    date_string: str,
    format_string: str,
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parse date string using specific strftime format.
    
    Args:
        date_string: Date string to parse (must be a string)
        format_string: strftime format string (must be a string)
        default: Default value if parsing fails (default: None)
    
    Returns:
        Parsed datetime or default if parsing fails
    
    Raises:
        TypeError: If date_string or format_string is not a string
    
    Examples:
        >>> parse_datetime_format("2024-01-01", "%Y-%m-%d")
        datetime.datetime(2024, 1, 1, 0, 0)
        >>> parse_datetime_format("invalid", "%Y-%m-%d", default=datetime(2024, 1, 1))
        datetime.datetime(2024, 1, 1, 0, 0)
    """
    # Validate inputs
    if not isinstance(date_string, str):
        raise TypeError(f"date_string must be a string, got {type(date_string).__name__}")
    if not isinstance(format_string, str):
        raise TypeError(f"format_string must be a string, got {type(format_string).__name__}")
    
    # Handle empty date_string
    if not date_string:
        return default
    
    try:
        return datetime.strptime(date_string, format_string)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse date '{date_string}' with format '{format_string}': {e}")
        return default


# ════════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ════════════════════════════════════════════════════════════════════════════════

def format_datetime(
    dt: datetime,
    format_string: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format datetime to string using strftime format.
    
    Args:
        dt: Datetime to format (must be a datetime object)
        format_string: strftime format string (default: "%Y-%m-%d %H:%M:%S")
    
    Returns:
        Formatted string
    
    Raises:
        TypeError: If dt is not a datetime object or format_string is not a string
    
    Examples:
        >>> format_datetime(datetime(2024, 1, 1, 12, 0, 0))
        '2024-01-01 12:00:00'
        >>> format_datetime(datetime(2024, 1, 1), "%Y-%m-%d")
        '2024-01-01'
    """
    # Validate inputs
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    if not isinstance(format_string, str):
        raise TypeError(f"format_string must be a string, got {type(format_string).__name__}")
    
    return dt.strftime(format_string)


def format_datetime_iso(dt: Optional[datetime] = None) -> str:
    """
    Format datetime to ISO string.
    
    Args:
        dt: Datetime to format (defaults to now_utc)
    
    Returns:
        ISO format string
    
    Example:
        >>> format_datetime_iso()
        '2024-01-01T12:00:00+00:00'
    """
    if dt is None:
        dt = now_utc()
    return dt.isoformat()


def format_relative_time(dt: datetime) -> str:
    """
    Format datetime as relative time (e.g., "2 hours ago" or "in 3 days").
    
    Args:
        dt: Datetime to format (must be a datetime object)
    
    Returns:
        Relative time string (e.g., "2 hours ago", "in 3 days", "just now")
    
    Raises:
        TypeError: If dt is not a datetime object
    
    Examples:
        >>> format_relative_time(now_utc() - timedelta(hours=2))
        '2 hours ago'
        >>> format_relative_time(now_utc() + timedelta(days=1))
        'in 1 day'
    """
    # Validate input
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    
    # Get current UTC time and calculate difference
    now = now_utc()
    # Normalize timezone: if dt has no timezone, assume UTC
    if dt.tzinfo is None:
        dt_normalized = dt.replace(tzinfo=timezone.utc)
    else:
        dt_normalized = dt
    diff = now - dt_normalized
    
    # Handle future times (negative difference)
    if diff.total_seconds() < 0:
        diff = abs(diff)
        if diff.days > 0:
            return f"in {diff.days} day{'s' if diff.days != 1 else ''}"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"in {hours} hour{'s' if hours != 1 else ''}"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"in {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            return "in a moment"
    else:
        # Handle past times (positive difference)
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"


# ════════════════════════════════════════════════════════════════════════════════
# TIMEZONE CONVERSION
# ════════════════════════════════════════════════════════════════════════════════

def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.
    
    Args:
        dt: Datetime to convert
    
    Returns:
        UTC datetime
    
    Example:
        >>> to_utc(datetime(2024, 1, 1, 12, 0, 0))
        datetime.datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ════════════════════════════════════════════════════════════════════════════════
# DATE ARITHMETIC
# ════════════════════════════════════════════════════════════════════════════════

def add_days(dt: datetime, days: int) -> datetime:
    """
    Add days to datetime (supports negative values for subtraction).
    
    Args:
        dt: Datetime (must be a datetime object)
        days: Number of days to add (can be negative for subtraction)
    
    Returns:
        New datetime
    
    Raises:
        TypeError: If dt is not a datetime object or days is not an int
    
    Examples:
        >>> add_days(datetime(2024, 1, 1), 5)
        datetime.datetime(2024, 1, 6, 0, 0)
        >>> add_days(datetime(2024, 1, 6), -5)
        datetime.datetime(2024, 1, 1, 0, 0)
    """
    # Validate inputs
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    if not isinstance(days, int):
        raise TypeError(f"days must be an integer, got {type(days).__name__}")
    
    return dt + timedelta(days=days)


def add_hours(dt: datetime, hours: int) -> datetime:
    """
    Add hours to datetime (supports negative values for subtraction).
    
    Args:
        dt: Datetime (must be a datetime object)
        hours: Number of hours to add (can be negative for subtraction)
    
    Returns:
        New datetime
    
    Raises:
        TypeError: If dt is not a datetime object or hours is not an int
    
    Examples:
        >>> add_hours(datetime(2024, 1, 1, 12, 0), 2)
        datetime.datetime(2024, 1, 1, 14, 0)
        >>> add_hours(datetime(2024, 1, 1, 14, 0), -2)
        datetime.datetime(2024, 1, 1, 12, 0)
    """
    # Validate inputs
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    if not isinstance(hours, int):
        raise TypeError(f"hours must be an integer, got {type(hours).__name__}")
    
    return dt + timedelta(hours=hours)


def add_minutes(dt: datetime, minutes: int) -> datetime:
    """
    Add minutes to datetime (supports negative values for subtraction).
    
    Args:
        dt: Datetime (must be a datetime object)
        minutes: Number of minutes to add (can be negative for subtraction)
    
    Returns:
        New datetime
    
    Raises:
        TypeError: If dt is not a datetime object or minutes is not an int
    
    Examples:
        >>> add_minutes(datetime(2024, 1, 1, 12, 0), 30)
        datetime.datetime(2024, 1, 1, 12, 30)
        >>> add_minutes(datetime(2024, 1, 1, 12, 30), -30)
        datetime.datetime(2024, 1, 1, 12, 0)
    """
    # Validate inputs
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    if not isinstance(minutes, int):
        raise TypeError(f"minutes must be an integer, got {type(minutes).__name__}")
    
    return dt + timedelta(minutes=minutes)


def add_seconds(dt: datetime, seconds: int) -> datetime:
    """
    Add seconds to datetime (supports negative values for subtraction).
    
    Args:
        dt: Datetime (must be a datetime object)
        seconds: Number of seconds to add (can be negative for subtraction)
    
    Returns:
        New datetime
    
    Raises:
        TypeError: If dt is not a datetime object or seconds is not an int
    
    Examples:
        >>> add_seconds(datetime(2024, 1, 1, 12, 0, 0), 30)
        datetime.datetime(2024, 1, 1, 12, 0, 30)
        >>> add_seconds(datetime(2024, 1, 1, 12, 0, 30), -30)
        datetime.datetime(2024, 1, 1, 12, 0, 0)
    """
    # Validate inputs
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be a datetime object, got {type(dt).__name__}")
    if not isinstance(seconds, int):
        raise TypeError(f"seconds must be an integer, got {type(seconds).__name__}")
    
    return dt + timedelta(seconds=seconds)


# ════════════════════════════════════════════════════════════════════════════════
# DATE BOUNDARIES
# ════════════════════════════════════════════════════════════════════════════════

def start_of_day(dt: datetime) -> datetime:
    """
    Get start of day (00:00:00).
    
    Args:
        dt: Datetime
    
    Returns:
        Datetime at start of day
    
    Example:
        >>> start_of_day(datetime(2024, 1, 1, 12, 30, 45))
        datetime.datetime(2024, 1, 1, 0, 0, 0)
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: datetime) -> datetime:
    """
    Get end of day (23:59:59.999999).
    
    Args:
        dt: Datetime
    
    Returns:
        Datetime at end of day
    """
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Current time
    "now_utc",
    "now_local",
    "now_iso",
    # Parsing
    "parse_datetime",
    "parse_datetime_format",
    # Formatting
    "format_datetime",
    "format_datetime_iso",
    "format_relative_time",
    # Timezone
    "to_utc",
    # Arithmetic
    "add_days",
    "add_hours",
    "add_minutes",
    "add_seconds",
    # Boundaries
    "start_of_day",
    "end_of_day",
]




