"""
Core Commit Tracking System
Main interface for commit tracking functionality
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from .models import CommitInfo, CommitMetadata, CommitStats, CommitAnalytics, CommitFilter, CommitQuery
from .parser import CommitHistoryParser
from .database import CommitDatabase, CommitStorage, CommitRepository
from .analytics import CommitStatistics, CommitTrends, CommitAnalytics as Analytics

class CommitTracker:
    """Main commit tracking interface"""
    
    def __init__(self, repo_path: str = ".", db_path: str = "commits.db"):
        self.repo_path = repo_path
        self.parser = CommitHistoryParser(repo_path)
        self.db = CommitDatabase(db_path)
        self.storage = CommitStorage()
        self.repository = CommitRepository(self.db, self.storage)
    
    def import_commits(self, branch: str = "main", limit: Optional[int] = None) -> List[CommitInfo]:
        """Import commits from git repository"""
        commits = self.parser.parse_history(branch, limit)
        
        # Store in database
        for commit in commits:
            self.db.store_commit(commit)
        
        # Also save to file storage
        self.storage.save_commits(commits)
        
        return commits
    
    def import_from_data(self, commit_data: List[Dict]) -> List[CommitInfo]:
        """Import commits from provided data structure"""
        commits = self.parser.parse_from_data(commit_data)
        
        # Store in database
        for commit in commits:
            self.db.store_commit(commit)
        
        return commits
    
    def get_commits(self, limit: Optional[int] = None, offset: int = 0) -> List[CommitInfo]:
        """Get commits from storage"""
        return self.db.get_commits(limit, offset)
    
    def get_commit(self, commit_hash: str) -> Optional[CommitInfo]:
        """Get specific commit by hash"""
        return self.db.get_commit(commit_hash)
    
    def search_commits(self, query: str) -> List[CommitInfo]:
        """Search commits by message or author"""
        return self.repository.search_commits(query)
    
    def get_statistics(self) -> CommitStats:
        """Get commit statistics"""
        commits = self.get_commits()
        stats_calculator = CommitStatistics(commits)
        return stats_calculator.calculate_stats()
    
    def get_analytics(self) -> Analytics:
        """Get comprehensive analytics"""
        commits = self.get_commits()
        analytics = Analytics(commits)
        return analytics.generate_analytics()

class CommitManager:
    """Advanced commit management with filtering and querying"""
    
    def __init__(self, tracker: CommitTracker):
        self.tracker = tracker
    
    def filter_commits(self, filter_criteria: CommitFilter) -> List[CommitInfo]:
        """Filter commits based on criteria"""
        all_commits = self.tracker.get_commits()
        filtered_commits = []
        
        for commit in all_commits:
            # Author filter
            if filter_criteria.authors and commit.author not in filter_criteria.authors:
                continue
            
            # Date range filter
            if filter_criteria.date_range:
                start_date, end_date = filter_criteria.date_range
                if not (start_date <= commit.date <= end_date):
                    continue
            
            # Commit type filter
            if filter_criteria.commit_types and commit.commit_type not in filter_criteria.commit_types:
                continue
            
            # File filter
            if filter_criteria.files:
                commit_files = [fc.file_path for fc in commit.files_changed]
                if not any(file in commit_files for file in filter_criteria.files):
                    continue
            
            # Message keywords filter
            if filter_criteria.message_keywords:
                message_lower = commit.message.lower()
                if not any(keyword.lower() in message_lower for keyword in filter_criteria.message_keywords):
                    continue
            
            # Lines changed filter
            total_changes = sum(fc.lines_changed for fc in commit.files_changed)
            if filter_criteria.min_lines_changed and total_changes < filter_criteria.min_lines_changed:
                continue
            if filter_criteria.max_lines_changed and total_changes > filter_criteria.max_lines_changed:
                continue
            
            filtered_commits.append(commit)
        
        return filtered_commits
    
    def query_commits(self, query: CommitQuery) -> List[CommitInfo]:
        """Execute complex query on commits"""
        commits = self.tracker.get_commits()
        
        # Apply filters
        for filter_criteria in query.filters:
            commits = self.filter_commits(filter_criteria)
        
        # Apply sorting
        if query.sort_by:
            reverse = query.sort_order == "desc"
            if query.sort_by == "date":
                commits.sort(key=lambda x: x.date, reverse=reverse)
            elif query.sort_by == "author":
                commits.sort(key=lambda x: x.author, reverse=reverse)
            elif query.sort_by == "message":
                commits.sort(key=lambda x: x.message, reverse=reverse)
        
        # Apply pagination
        if query.offset:
            commits = commits[query.offset:]
        if query.limit:
            commits = commits[:query.limit]
        
        return commits
    
    def get_author_summary(self, author: str) -> Dict[str, Any]:
        """Get summary for specific author"""
        author_commits = self.filter_commits(CommitFilter(authors=[author]))
        
        if not author_commits:
            return {}
        
        total_commits = len(author_commits)
        total_files = sum(len(commit.files_changed) for commit in author_commits)
        total_lines = sum(
            sum(fc.lines_changed for fc in commit.files_changed) 
            for commit in author_commits
        )
        
        # Commit types
        commit_types = {}
        for commit in author_commits:
            if commit.commit_type:
                commit_types[commit.commit_type.value] = commit_types.get(commit.commit_type.value, 0) + 1
        
        # Date range
        dates = [commit.date for commit in author_commits]
        first_commit = min(dates)
        last_commit = max(dates)
        
        return {
            'author': author,
            'total_commits': total_commits,
            'total_files': total_files,
            'total_lines': total_lines,
            'commit_types': commit_types,
            'first_commit': first_commit,
            'last_commit': last_commit,
            'active_days': (last_commit - first_commit).days + 1 if first_commit != last_commit else 1
        }
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """Get summary for specific file"""
        file_commits = []
        for commit in self.tracker.get_commits():
            for fc in commit.files_changed:
                if fc.file_path == file_path:
                    file_commits.append((commit, fc))
                    break
        
        if not file_commits:
            return {}
        
        total_commits = len(file_commits)
        total_lines_added = sum(fc.lines_added for _, fc in file_commits)
        total_lines_deleted = sum(fc.lines_deleted for _, fc in file_commits)
        
        # Authors who modified this file
        authors = set(commit.author for commit, _ in file_commits)
        
        # Change types
        change_types = {}
        for _, fc in file_commits:
            change_types[fc.change_type.value] = change_types.get(fc.change_type.value, 0) + 1
        
        return {
            'file_path': file_path,
            'total_commits': total_commits,
            'total_lines_added': total_lines_added,
            'total_lines_deleted': total_lines_deleted,
            'net_lines': total_lines_added - total_lines_deleted,
            'authors': list(authors),
            'change_types': change_types
        }

class CommitSystem:
    """Complete commit tracking system with all features"""
    
    def __init__(self, repo_path: str = ".", db_path: str = "commits.db"):
        self.tracker = CommitTracker(repo_path, db_path)
        self.manager = CommitManager(self.tracker)
    
    def setup_from_git(self, branch: str = "main", limit: Optional[int] = None) -> List[CommitInfo]:
        """Setup system by importing from git repository"""
        return self.tracker.import_commits(branch, limit)
    
    def setup_from_data(self, commit_data: List[Dict]) -> List[CommitInfo]:
        """Setup system from provided data"""
        return self.tracker.import_from_data(commit_data)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for commit dashboard"""
        stats = self.tracker.get_statistics()
        analytics = self.tracker.get_analytics()
        
        return {
            'statistics': {
                'total_commits': stats.total_commits,
                'total_files': stats.files_changed,
                'total_lines_added': stats.lines_added,
                'total_lines_deleted': stats.lines_deleted,
                'net_lines': stats.net_lines,
                'average_files_per_commit': stats.average_files_per_commit,
                'average_lines_per_commit': stats.average_lines_per_commit
            },
            'analytics': {
                'commit_velocity': analytics.commit_velocity,
                'author_activity': analytics.author_activity,
                'file_activity': analytics.file_activity,
                'commit_patterns': analytics.commit_patterns,
                'code_quality_metrics': analytics.code_quality_metrics,
                'technical_debt_indicators': analytics.technical_debt_indicators
            },
            'commits_by_author': stats.commits_by_author,
            'commits_by_type': {k.value: v for k, v in stats.commits_by_type.items()},
            'commits_by_month': stats.commits_by_month
        }
    
    def export_data(self, format: str = "json") -> str:
        """Export commit data in specified format"""
        commits = self.tracker.get_commits()
        
        if format == "json":
            import json
            data = []
            for commit in commits:
                data.append({
                    'hash': commit.hash,
                    'short_hash': commit.short_hash,
                    'author': commit.author,
                    'email': commit.email,
                    'date': commit.date.isoformat(),
                    'message': commit.message,
                    'commit_type': commit.commit_type.value if commit.commit_type else None,
                    'files_changed': [
                        {
                            'file_path': fc.file_path,
                            'change_type': fc.change_type.value,
                            'lines_added': fc.lines_added,
                            'lines_deleted': fc.lines_deleted,
                            'lines_changed': fc.lines_changed
                        } for fc in commit.files_changed
                    ]
                })
            return json.dumps(data, indent=2)
        
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['Hash', 'Author', 'Date', 'Message', 'Type', 'Files Changed', 'Lines Added', 'Lines Deleted'])
            
            # Data
            for commit in commits:
                total_files = len(commit.files_changed)
                total_added = sum(fc.lines_added for fc in commit.files_changed)
                total_deleted = sum(fc.lines_deleted for fc in commit.files_changed)
                
                writer.writerow([
                    commit.hash,
                    commit.author,
                    commit.date.isoformat(),
                    commit.message,
                    commit.commit_type.value if commit.commit_type else '',
                    total_files,
                    total_added,
                    total_deleted
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_report(self) -> str:
        """Generate comprehensive commit report"""
        dashboard_data = self.get_dashboard_data()
        
        report = f"""
# Commit Tracking Report
Generated on: {datetime.now().isoformat()}

## Summary Statistics
- Total Commits: {dashboard_data['statistics']['total_commits']}
- Total Files Changed: {dashboard_data['statistics']['total_files']}
- Total Lines Added: {dashboard_data['statistics']['total_lines_added']}
- Total Lines Deleted: {dashboard_data['statistics']['total_lines_deleted']}
- Net Lines: {dashboard_data['statistics']['net_lines']}

## Commit Velocity
- Commits per day (30-day average): {dashboard_data['analytics']['commit_velocity']:.2f}

## Top Contributors
"""
        
        # Top contributors
        author_stats = sorted(
            dashboard_data['commits_by_author'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for author, count in author_stats:
            report += f"- {author}: {count} commits\n"
        
        report += "\n## Commit Types\n"
        for commit_type, count in dashboard_data['commits_by_type'].items():
            report += f"- {commit_type}: {count} commits\n"
        
        return report




