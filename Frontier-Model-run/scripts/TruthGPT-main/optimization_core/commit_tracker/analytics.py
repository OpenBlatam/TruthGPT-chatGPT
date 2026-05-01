"""
Commit Analytics and Statistics
Advanced analytics for commit data
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from .models import CommitInfo, CommitStats, CommitAnalytics, CommitType

class CommitStatistics:
    """Calculate commit statistics"""
    
    def __init__(self, commits: List[CommitInfo]):
        self.commits = commits
    
    def calculate_stats(self) -> CommitStats:
        """Calculate comprehensive commit statistics"""
        if not self.commits:
            return CommitStats(
                total_commits=0,
                commits_by_author={},
                commits_by_type={},
                commits_by_month={},
                files_changed=0,
                lines_added=0,
                lines_deleted=0,
                net_lines=0,
                average_files_per_commit=0.0,
                average_lines_per_commit=0.0
            )
        
        # Basic counts
        total_commits = len(self.commits)
        
        # Author statistics
        commits_by_author = Counter(commit.author for commit in self.commits)
        
        # Type statistics
        commits_by_type = Counter(
            commit.commit_type for commit in self.commits 
            if commit.commit_type
        )
        
        # Monthly statistics
        commits_by_month = defaultdict(int)
        for commit in self.commits:
            month_key = commit.date.strftime('%Y-%m')
            commits_by_month[month_key] += 1
        
        # File and line statistics
        total_files = sum(len(commit.files_changed) for commit in self.commits)
        total_lines_added = sum(
            sum(fc.lines_added for fc in commit.files_changed) 
            for commit in self.commits
        )
        total_lines_deleted = sum(
            sum(fc.lines_deleted for fc in commit.files_changed) 
            for commit in self.commits
        )
        net_lines = total_lines_added - total_lines_deleted
        
        return CommitStats(
            total_commits=total_commits,
            commits_by_author=dict(commits_by_author),
            commits_by_type=dict(commits_by_type),
            commits_by_month=dict(commits_by_month),
            files_changed=total_files,
            lines_added=total_lines_added,
            lines_deleted=total_lines_deleted,
            net_lines=net_lines,
            average_files_per_commit=total_files / total_commits if total_commits > 0 else 0,
            average_lines_per_commit=(total_lines_added + total_lines_deleted) / total_commits if total_commits > 0 else 0
        )
    
    def get_author_activity(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed author activity"""
        author_activity = defaultdict(lambda: {
            'total_commits': 0,
            'commits_by_type': defaultdict(int),
            'files_changed': 0,
            'lines_added': 0,
            'lines_deleted': 0,
            'first_commit': None,
            'last_commit': None,
            'commit_frequency': 0.0
        })
        
        for commit in self.commits:
            author = commit.author
            activity = author_activity[author]
            
            activity['total_commits'] += 1
            activity['files_changed'] += len(commit.files_changed)
            
            if commit.commit_type:
                activity['commits_by_type'][commit.commit_type] += 1
            
            # Line counts
            for fc in commit.files_changed:
                activity['lines_added'] += fc.lines_added
                activity['lines_deleted'] += fc.lines_deleted
            
            # Date tracking
            if not activity['first_commit'] or commit.date < activity['first_commit']:
                activity['first_commit'] = commit.date
            if not activity['last_commit'] or commit.date > activity['last_commit']:
                activity['last_commit'] = commit.date
        
        # Calculate frequency
        for author, activity in author_activity.items():
            if activity['first_commit'] and activity['last_commit']:
                days = (activity['last_commit'] - activity['first_commit']).days + 1
                activity['commit_frequency'] = activity['total_commits'] / days if days > 0 else 0
        
        return dict(author_activity)
    
    def get_file_activity(self) -> Dict[str, Dict[str, Any]]:
        """Get file activity statistics"""
        file_activity = defaultdict(lambda: {
            'commits': 0,
            'authors': set(),
            'lines_added': 0,
            'lines_deleted': 0,
            'change_types': defaultdict(int),
            'first_modified': None,
            'last_modified': None
        })
        
        for commit in self.commits:
            for fc in commit.files_changed:
                file_path = fc.file_path
                activity = file_activity[file_path]
                
                activity['commits'] += 1
                activity['authors'].add(commit.author)
                activity['lines_added'] += fc.lines_added
                activity['lines_deleted'] += fc.lines_deleted
                activity['change_types'][fc.change_type] += 1
                
                # Date tracking
                if not activity['first_modified'] or commit.date < activity['first_modified']:
                    activity['first_modified'] = commit.date
                if not activity['last_modified'] or commit.date > activity['last_modified']:
                    activity['last_modified'] = commit.date
        
        # Convert sets to counts
        for file_path, activity in file_activity.items():
            activity['unique_authors'] = len(activity['authors'])
            activity['authors'] = list(activity['authors'])
        
        return dict(file_activity)

class CommitTrends:
    """Analyze commit trends over time"""
    
    def __init__(self, commits: List[CommitInfo]):
        self.commits = commits
    
    def get_commit_velocity(self, days: int = 30) -> float:
        """Calculate commits per day over specified period"""
        if not self.commits:
            return 0.0
        
        end_date = max(commit.date for commit in self.commits)
        start_date = end_date - timedelta(days=days)
        
        recent_commits = [
            commit for commit in self.commits 
            if start_date <= commit.date <= end_date
        ]
        
        return len(recent_commits) / days
    
    def get_commit_patterns(self) -> Dict[str, Any]:
        """Analyze commit patterns"""
        patterns = {
            'hourly_distribution': defaultdict(int),
            'daily_distribution': defaultdict(int),
            'weekly_distribution': defaultdict(int),
            'commit_size_distribution': defaultdict(int),
            'message_length_distribution': defaultdict(int)
        }
        
        for commit in self.commits:
            # Time patterns
            patterns['hourly_distribution'][commit.date.hour] += 1
            patterns['daily_distribution'][commit.date.weekday()] += 1
            patterns['weekly_distribution'][commit.date.isocalendar()[1]] += 1
            
            # Size patterns
            total_changes = sum(fc.lines_changed for fc in commit.files_changed)
            if total_changes < 10:
                patterns['commit_size_distribution']['small'] += 1
            elif total_changes < 100:
                patterns['commit_size_distribution']['medium'] += 1
            else:
                patterns['commit_size_distribution']['large'] += 1
            
            # Message length
            msg_len = len(commit.message)
            if msg_len < 50:
                patterns['message_length_distribution']['short'] += 1
            elif msg_len < 200:
                patterns['message_length_distribution']['medium'] += 1
            else:
                patterns['message_length_distribution']['long'] += 1
        
        return dict(patterns)
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous commit patterns"""
        anomalies = []
        
        # Large commits
        for commit in self.commits:
            total_changes = sum(fc.lines_changed for fc in commit.files_changed)
            if total_changes > 1000:  # Threshold for large commits
                anomalies.append({
                    'type': 'large_commit',
                    'commit_hash': commit.hash,
                    'author': commit.author,
                    'date': commit.date,
                    'total_changes': total_changes,
                    'message': commit.message[:100]
                })
        
        # Frequent commits by same author
        author_commits = defaultdict(list)
        for commit in self.commits:
            author_commits[commit.author].append(commit)
        
        for author, commits in author_commits.items():
            if len(commits) > 10:  # Threshold for frequent commits
                # Check for commits within short time periods
                commits.sort(key=lambda x: x.date)
                for i in range(len(commits) - 1):
                    time_diff = commits[i+1].date - commits[i].date
                    if time_diff.total_seconds() < 300:  # 5 minutes
                        anomalies.append({
                            'type': 'rapid_commits',
                            'author': author,
                            'commit1': commits[i].hash,
                            'commit2': commits[i+1].hash,
                            'time_diff_seconds': time_diff.total_seconds()
                        })
        
        return anomalies

class CommitAnalytics:
    """Comprehensive commit analytics"""
    
    def __init__(self, commits: List[CommitInfo]):
        self.commits = commits
        self.stats = CommitStatistics(commits)
        self.trends = CommitTrends(commits)
    
    def generate_analytics(self) -> CommitAnalytics:
        """Generate comprehensive analytics"""
        stats = self.stats.calculate_stats()
        author_activity = self.stats.get_author_activity()
        file_activity = self.stats.get_file_activity()
        patterns = self.trends.get_commit_patterns()
        anomalies = self.trends.detect_anomalies()
        
        # Calculate velocity
        velocity_30d = self.trends.get_commit_velocity(30)
        velocity_7d = self.trends.get_commit_velocity(7)
        
        # Code quality metrics
        code_quality = self._calculate_code_quality_metrics()
        
        # Technical debt indicators
        tech_debt = self._calculate_technical_debt_indicators()
        
        return CommitAnalytics(
            commit_velocity=velocity_30d,
            author_activity=author_activity,
            file_activity=file_activity,
            commit_patterns=patterns,
            code_quality_metrics=code_quality,
            technical_debt_indicators=tech_debt
        )
    
    def _calculate_code_quality_metrics(self) -> Dict[str, float]:
        """Calculate code quality metrics"""
        metrics = {
            'average_commit_size': 0.0,
            'commit_frequency': 0.0,
            'author_diversity': 0.0,
            'file_concentration': 0.0
        }
        
        if not self.commits:
            return metrics
        
        # Average commit size
        total_changes = sum(
            sum(fc.lines_changed for fc in commit.files_changed) 
            for commit in self.commits
        )
        metrics['average_commit_size'] = total_changes / len(self.commits)
        
        # Commit frequency
        if len(self.commits) > 1:
            date_range = max(c.date for c in self.commits) - min(c.date for c in self.commits)
            metrics['commit_frequency'] = len(self.commits) / max(date_range.days, 1)
        
        # Author diversity
        unique_authors = len(set(commit.author for commit in self.commits))
        metrics['author_diversity'] = unique_authors / len(self.commits)
        
        # File concentration (how many files are frequently changed)
        file_changes = defaultdict(int)
        for commit in self.commits:
            for fc in commit.files_changed:
                file_changes[fc.file_path] += 1
        
        if file_changes:
            max_changes = max(file_changes.values())
            total_files = len(file_changes)
            metrics['file_concentration'] = max_changes / total_files if total_files > 0 else 0
        
        return metrics
    
    def _calculate_technical_debt_indicators(self) -> Dict[str, Any]:
        """Calculate technical debt indicators"""
        indicators = {
            'large_commits': 0,
            'frequent_changes': [],
            'complex_files': [],
            'refactoring_ratio': 0.0
        }
        
        # Large commits
        for commit in self.commits:
            total_changes = sum(fc.lines_changed for fc in commit.files_changed)
            if total_changes > 500:
                indicators['large_commits'] += 1
        
        # Frequently changed files
        file_changes = defaultdict(int)
        for commit in self.commits:
            for fc in commit.files_changed:
                file_changes[fc.file_path] += 1
        
        indicators['frequent_changes'] = [
            {'file': file, 'changes': count} 
            for file, count in file_changes.items() 
            if count > 10
        ]
        
        # Refactoring ratio
        refactor_commits = sum(
            1 for commit in self.commits 
            if commit.commit_type == CommitType.REFACTOR
        )
        indicators['refactoring_ratio'] = refactor_commits / len(self.commits) if self.commits else 0
        
        return indicators




