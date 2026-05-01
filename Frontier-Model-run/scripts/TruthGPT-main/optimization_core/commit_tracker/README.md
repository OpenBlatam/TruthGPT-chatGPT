# Commit Tracking System

A comprehensive system for tracking, analyzing, and managing commit metadata with advanced analytics and reporting capabilities.

## Features

- **Commit Import**: Import commits from git repositories or structured data
- **Data Storage**: SQLite database with file-based backup storage
- **Advanced Analytics**: Comprehensive statistics and trend analysis
- **Filtering & Search**: Powerful query capabilities with multiple criteria
- **Reporting**: Generate detailed reports in multiple formats
- **Visualization**: Dashboard-ready data for commit analytics

## Quick Start

```python
from commit_tracker import CommitSystem

# Initialize the system
commit_system = CommitSystem()

# Import from git repository
commits = commit_system.setup_from_git(branch="main", limit=100)

# Or import from structured data
sample_data = [
    {
        'hash': 'abc123def456',
        'author': 'Developer A',
        'date': '2023-11-24T10:30:00',
        'message': 'Update main branch post 24.11 (#7829)',
        'commit_type': 'chore',
        'files_changed': [
            {'file_path': 'CMakeLists.txt', 'change_type': 'modified', 'lines_added': 5, 'lines_deleted': 2}
        ]
    }
]
commits = commit_system.setup_from_data(sample_data)

# Get statistics
stats = commit_system.tracker.get_statistics()
print(f"Total commits: {stats.total_commits}")

# Get analytics
analytics = commit_system.tracker.get_analytics()
print(f"Commit velocity: {analytics.commit_velocity}")
```

## Core Components

### Models

- **CommitInfo**: Basic commit information (hash, author, date, message, etc.)
- **CommitMetadata**: Extended metadata (branch, tags, PR numbers, etc.)
- **FileChange**: File change details (path, type, line counts)
- **CommitStats**: Statistical summaries
- **CommitAnalytics**: Advanced analytics and trends

### Parser

- **GitLogParser**: Parse git log output
- **CommitParser**: Parse individual commit data
- **CommitHistoryParser**: Parse complete commit history

### Database

- **CommitDatabase**: SQLite storage with full schema
- **CommitStorage**: File-based JSON storage
- **CommitRepository**: Repository pattern for data access

### Analytics

- **CommitStatistics**: Calculate comprehensive statistics
- **CommitTrends**: Analyze trends and patterns
- **CommitAnalytics**: Generate advanced analytics

### Core System

- **CommitTracker**: Main tracking interface
- **CommitManager**: Advanced management with filtering
- **CommitSystem**: Complete system with all features

## Usage Examples

### Basic Statistics

```python
# Get basic statistics
stats = commit_system.tracker.get_statistics()
print(f"Total commits: {stats.total_commits}")
print(f"Files changed: {stats.files_changed}")
print(f"Lines added: {stats.lines_added}")
print(f"Lines deleted: {stats.lines_deleted}")
```

### Filtering Commits

```python
from commit_tracker.models import CommitFilter, CommitType

# Filter by author
author_filter = CommitFilter(authors=['Developer A'])
author_commits = commit_system.manager.filter_commits(author_filter)

# Filter by commit type
feature_filter = CommitFilter(commit_types=[CommitType.FEATURE])
feature_commits = commit_system.manager.filter_commits(feature_filter)

# Filter by date range
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
date_filter = CommitFilter(date_range=(start_date, end_date))
recent_commits = commit_system.manager.filter_commits(date_filter)
```

### Advanced Queries

```python
from commit_tracker.models import CommitQuery

# Build complex queries
query = (CommitQuery()
    .add_filter(CommitFilter(authors=['Developer A']))
    .add_filter(CommitFilter(commit_types=[CommitType.FEATURE]))
    .sort('date', 'desc')
    .limit_results(10))

results = commit_system.manager.query_commits(query)
```

### Analytics and Trends

```python
# Get comprehensive analytics
analytics = commit_system.tracker.get_analytics()

# Author activity
for author, activity in analytics.author_activity.items():
    print(f"{author}: {activity['total_commits']} commits")
    print(f"  Files changed: {activity['files_changed']}")
    print(f"  Lines added: {activity['lines_added']}")
    print(f"  Commit frequency: {activity['commit_frequency']:.2f} commits/day")

# File activity
for file_path, activity in analytics.file_activity.items():
    print(f"{file_path}: {activity['commits']} commits")
    print(f"  Authors: {activity['unique_authors']}")
    print(f"  Lines added: {activity['lines_added']}")
```

### Author and File Summaries

```python
# Get author summary
author_summary = commit_system.manager.get_author_summary('Developer A')
print(f"Total commits: {author_summary['total_commits']}")
print(f"Total files: {author_summary['total_files']}")
print(f"Active days: {author_summary['active_days']}")

# Get file summary
file_summary = commit_system.manager.get_file_summary('grpc_server.cc')
print(f"Total commits: {file_summary['total_commits']}")
print(f"Net lines: {file_summary['net_lines']}")
print(f"Authors: {file_summary['authors']}")
```

### Export and Reporting

```python
# Generate comprehensive report
report = commit_system.generate_report()
print(report)

# Export data in different formats
json_data = commit_system.export_data('json')
csv_data = commit_system.export_data('csv')

# Get dashboard data
dashboard_data = commit_system.get_dashboard_data()
```

## Data Structure

### Commit Information

```python
@dataclass
class CommitInfo:
    hash: str
    short_hash: str
    author: str
    email: str
    date: datetime
    message: str
    commit_type: Optional[CommitType]
    files_changed: List[FileChange]
    parent_hashes: List[str]
    merge: bool
```

### File Changes

```python
@dataclass
class FileChange:
    file_path: str
    change_type: FileChangeType
    lines_added: int
    lines_deleted: int
    lines_changed: int
    binary: bool
    old_path: Optional[str]
    new_path: Optional[str]
```

### Commit Types

- `FEATURE`: New features
- `BUGFIX`: Bug fixes
- `HOTFIX`: Critical fixes
- `REFACTOR`: Code refactoring
- `DOCS`: Documentation
- `STYLE`: Code style changes
- `TEST`: Test-related changes
- `CHORE`: Maintenance tasks
- `MERGE`: Merge commits
- `REVERT`: Revert commits

### File Change Types

- `ADDED`: New files
- `MODIFIED`: Modified files
- `DELETED`: Deleted files
- `RENAMED`: Renamed files
- `COPIED`: Copied files

## Database Schema

The system uses SQLite with the following tables:

- **commits**: Main commit information
- **file_changes**: File change details
- **commit_metadata**: Extended metadata

## Configuration

### Database Path
```python
commit_system = CommitSystem(db_path="custom_commits.db")
```

### Repository Path
```python
commit_system = CommitSystem(repo_path="/path/to/repo")
```

## Advanced Features

### Custom Analytics

```python
# Calculate custom metrics
stats_calculator = CommitStatistics(commits)
stats = stats_calculator.calculate_stats()

# Get author activity
author_activity = stats_calculator.get_author_activity()

# Get file activity
file_activity = stats_calculator.get_file_activity()
```

### Trend Analysis

```python
# Analyze trends
trends = CommitTrends(commits)

# Get commit velocity
velocity = trends.get_commit_velocity(days=30)

# Get commit patterns
patterns = trends.get_commit_patterns()

# Detect anomalies
anomalies = trends.detect_anomalies()
```

### Technical Debt Indicators

```python
analytics = commit_system.tracker.get_analytics()
tech_debt = analytics.technical_debt_indicators

print(f"Large commits: {tech_debt['large_commits']}")
print(f"Refactoring ratio: {tech_debt['refactoring_ratio']:.2f}")
print(f"Frequently changed files: {len(tech_debt['frequent_changes'])}")
```

## Performance Considerations

- Database operations are optimized with proper indexing
- Large commit histories are handled efficiently with pagination
- Memory usage is optimized for large datasets
- Export operations support streaming for large datasets

## Error Handling

The system includes comprehensive error handling:

- Invalid commit data is skipped with warnings
- Database operations include transaction safety
- File operations include proper error recovery
- Git operations include fallback mechanisms

## Extensibility

The system is designed for extensibility:

- Custom commit types can be added
- New analytics can be implemented
- Additional storage backends can be added
- Custom export formats can be supported

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Create an issue in the repository
- Check the documentation
- Review the examples in the demo script



