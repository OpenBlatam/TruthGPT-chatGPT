"""
Commit History Parser
Parse git log and commit data into structured format
"""

import re
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from .models import CommitInfo, CommitMetadata, FileChange, CommitType, FileChangeType

class GitLogParser:
    """Parse git log output"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
    
    def get_commit_log(self, branch: str = "main", limit: Optional[int] = None) -> str:
        """Get git log output"""
        cmd = ["git", "log", "--pretty=format:%H|%h|%an|%ae|%ad|%s", "--date=iso"]
        if limit:
            cmd.extend(["-n", str(limit)])
        cmd.append(branch)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            return result.stdout
        except Exception as e:
            raise Exception(f"Failed to get git log: {e}")
    
    def get_file_changes(self, commit_hash: str) -> List[FileChange]:
        """Get file changes for a specific commit"""
        cmd = ["git", "show", "--name-status", "--pretty=format:", commit_hash]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            return self._parse_file_changes(result.stdout)
        except Exception as e:
            return []
    
    def _parse_file_changes(self, output: str) -> List[FileChange]:
        """Parse file changes from git show output"""
        changes = []
        for line in output.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            status, file_path = parts
            change_type = self._map_status_to_change_type(status)
            
            # Get line counts
            lines_added, lines_deleted = self._get_line_counts(file_path)
            
            changes.append(FileChange(
                file_path=file_path,
                change_type=change_type,
                lines_added=lines_added,
                lines_deleted=lines_deleted,
                lines_changed=lines_added + lines_deleted
            ))
        
        return changes
    
    def _map_status_to_change_type(self, status: str) -> FileChangeType:
        """Map git status to change type"""
        if status.startswith('A'):
            return FileChangeType.ADDED
        elif status.startswith('M'):
            return FileChangeType.MODIFIED
        elif status.startswith('D'):
            return FileChangeType.DELETED
        elif status.startswith('R'):
            return FileChangeType.RENAMED
        elif status.startswith('C'):
            return FileChangeType.COPIED
        else:
            return FileChangeType.MODIFIED
    
    def _get_line_counts(self, file_path: str) -> Tuple[int, int]:
        """Get line counts for a file"""
        # This is a simplified implementation
        # In practice, you'd use git diff --numstat
        return 0, 0

class CommitParser:
    """Parse commit information from various sources"""
    
    def __init__(self):
        self.git_parser = GitLogParser()
    
    def parse_commit_line(self, line: str) -> Optional[CommitInfo]:
        """Parse a single commit line"""
        parts = line.split('|')
        if len(parts) < 6:
            return None
        
        try:
            hash_full, hash_short, author, email, date_str, message = parts[:6]
            
            # Parse date
            date = datetime.fromisoformat(date_str.replace(' ', 'T'))
            
            # Determine commit type from message
            commit_type = self._classify_commit_type(message)
            
            return CommitInfo(
                hash=hash_full,
                short_hash=hash_short,
                author=author,
                email=email,
                date=date,
                message=message,
                commit_type=commit_type
            )
        except Exception as e:
            print(f"Error parsing commit line: {e}")
            return None
    
    def _classify_commit_type(self, message: str) -> CommitType:
        """Classify commit type based on message"""
        message_lower = message.lower()
        
        if message_lower.startswith('feat'):
            return CommitType.FEATURE
        elif message_lower.startswith('fix'):
            return CommitType.BUGFIX
        elif message_lower.startswith('hotfix'):
            return CommitType.HOTFIX
        elif message_lower.startswith('refactor'):
            return CommitType.REFACTOR
        elif message_lower.startswith('docs'):
            return CommitType.DOCS
        elif message_lower.startswith('style'):
            return CommitType.STYLE
        elif message_lower.startswith('test'):
            return CommitType.TEST
        elif message_lower.startswith('chore'):
            return CommitType.CHORE
        elif message_lower.startswith('merge'):
            return CommitType.MERGE
        elif message_lower.startswith('revert'):
            return CommitType.REVERT
        else:
            return CommitType.CHORE

class CommitHistoryParser:
    """Parse complete commit history"""
    
    def __init__(self, repo_path: str = "."):
        self.parser = CommitParser()
        self.git_parser = GitLogParser(repo_path)
    
    def parse_history(self, branch: str = "main", limit: Optional[int] = None) -> List[CommitInfo]:
        """Parse complete commit history"""
        log_output = self.git_parser.get_commit_log(branch, limit)
        commits = []
        
        for line in log_output.strip().split('\n'):
            if not line:
                continue
            
            commit = self.parser.parse_commit_line(line)
            if commit:
                # Get file changes
                commit.files_changed = self.git_parser.get_file_changes(commit.hash)
                commits.append(commit)
        
        return commits
    
    def parse_from_data(self, commit_data: List[Dict]) -> List[CommitInfo]:
        """Parse commits from provided data structure"""
        commits = []
        
        for data in commit_data:
            try:
                commit = CommitInfo(
                    hash=data.get('hash', ''),
                    short_hash=data.get('short_hash', ''),
                    author=data.get('author', ''),
                    email=data.get('email', ''),
                    date=datetime.fromisoformat(data.get('date', '')),
                    message=data.get('message', ''),
                    commit_type=CommitType(data.get('commit_type', 'chore')) if data.get('commit_type') else None
                )
                commits.append(commit)
            except Exception as e:
                print(f"Error parsing commit data: {e}")
                continue
        
        return commits




