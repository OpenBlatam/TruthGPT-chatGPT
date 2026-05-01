"""
Commit Tracking Models
Data structures for commit metadata and analytics
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import json

class CommitType(Enum):
    """Types of commits"""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    HOTFIX = "hotfix"
    REFACTOR = "refactor"
    DOCS = "docs"
    STYLE = "style"
    TEST = "test"
    CHORE = "chore"
    MERGE = "merge"
    REVERT = "revert"

class FileChangeType(Enum):
    """Types of file changes"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"

@dataclass
class FileChange:
    """Represents a file change in a commit"""
    file_path: str
    change_type: FileChangeType
    lines_added: int = 0
    lines_deleted: int = 0
    lines_changed: int = 0
    binary: bool = False
    old_path: Optional[str] = None
    new_path: Optional[str] = None

@dataclass
class CommitInfo:
    """Basic commit information"""
    hash: str
    short_hash: str
    author: str
    email: str
    date: datetime
    message: str
    commit_type: Optional[CommitType] = None
    files_changed: List[FileChange] = field(default_factory=list)
    parent_hashes: List[str] = field(default_factory=list)
    merge: bool = False

@dataclass
class CommitMetadata:
    """Extended commit metadata"""
    commit_info: CommitInfo
    branch: str
    tags: List[str] = field(default_factory=list)
    pull_request: Optional[int] = None
    issue_numbers: List[int] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    ci_status: Optional[str] = None
    deployment_status: Optional[str] = None

@dataclass
class CommitStats:
    """Commit statistics"""
    total_commits: int
    commits_by_author: Dict[str, int]
    commits_by_type: Dict[CommitType, int]
    commits_by_month: Dict[str, int]
    files_changed: int
    lines_added: int
    lines_deleted: int
    net_lines: int
    average_files_per_commit: float
    average_lines_per_commit: float

@dataclass
class CommitHistory:
    """Complete commit history"""
    commits: List[CommitInfo]
    metadata: List[CommitMetadata]
    stats: CommitStats
    start_date: datetime
    end_date: datetime
    total_days: int

@dataclass
class CommitAnalytics:
    """Advanced commit analytics"""
    commit_velocity: float  # commits per day
    author_activity: Dict[str, Dict[str, Any]]
    file_activity: Dict[str, Dict[str, Any]]
    commit_patterns: Dict[str, Any]
    code_quality_metrics: Dict[str, float]
    technical_debt_indicators: Dict[str, Any]

@dataclass
class CommitFilter:
    """Filter criteria for commits"""
    authors: Optional[List[str]] = None
    date_range: Optional[tuple] = None
    commit_types: Optional[List[CommitType]] = None
    files: Optional[List[str]] = None
    branches: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    message_keywords: Optional[List[str]] = None
    min_lines_changed: Optional[int] = None
    max_lines_changed: Optional[int] = None

class CommitQuery:
    """Query builder for commits"""
    
    def __init__(self):
        self.filters: List[CommitFilter] = []
        self.sort_by: Optional[str] = None
        self.sort_order: str = "desc"
        self.limit: Optional[int] = None
        self.offset: int = 0
    
    def add_filter(self, filter_criteria: CommitFilter) -> 'CommitQuery':
        """Add filter criteria"""
        self.filters.append(filter_criteria)
        return self
    
    def sort(self, field: str, order: str = "desc") -> 'CommitQuery':
        """Set sorting criteria"""
        self.sort_by = field
        self.sort_order = order
        return self
    
    def limit_results(self, count: int) -> 'CommitQuery':
        """Limit number of results"""
        self.limit = count
        return self
    
    def paginate(self, page: int, per_page: int) -> 'CommitQuery':
        """Set pagination"""
        self.offset = (page - 1) * per_page
        self.limit = per_page
        return self




