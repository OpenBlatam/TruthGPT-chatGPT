"""
Commit Database and Storage
Persistent storage for commit data
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from .models import CommitInfo, CommitMetadata, CommitStats, CommitHistory, FileChange, FileChangeType

class CommitDatabase:
    """SQLite database for commit storage"""
    
    def __init__(self, db_path: str = "commits.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Commits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS commits (
                    hash TEXT PRIMARY KEY,
                    short_hash TEXT,
                    author TEXT,
                    email TEXT,
                    date TEXT,
                    message TEXT,
                    commit_type TEXT,
                    branch TEXT,
                    merge BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # File changes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    commit_hash TEXT,
                    file_path TEXT,
                    change_type TEXT,
                    lines_added INTEGER DEFAULT 0,
                    lines_deleted INTEGER DEFAULT 0,
                    lines_changed INTEGER DEFAULT 0,
                    binary BOOLEAN DEFAULT FALSE,
                    old_path TEXT,
                    new_path TEXT,
                    FOREIGN KEY (commit_hash) REFERENCES commits (hash)
                )
            ''')
            
            # Commit metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS commit_metadata (
                    commit_hash TEXT PRIMARY KEY,
                    tags TEXT,
                    pull_request INTEGER,
                    issue_numbers TEXT,
                    reviewers TEXT,
                    ci_status TEXT,
                    deployment_status TEXT,
                    FOREIGN KEY (commit_hash) REFERENCES commits (hash)
                )
            ''')
            
            conn.commit()
    
    def store_commit(self, commit: CommitInfo, metadata: Optional[CommitMetadata] = None):
        """Store a commit in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store commit
            cursor.execute('''
                INSERT OR REPLACE INTO commits 
                (hash, short_hash, author, email, date, message, commit_type, branch, merge)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                commit.hash,
                commit.short_hash,
                commit.author,
                commit.email,
                commit.date.isoformat(),
                commit.message,
                commit.commit_type.value if commit.commit_type else None,
                metadata.branch if metadata else None,
                commit.merge
            ))
            
            # Store file changes
            cursor.execute('DELETE FROM file_changes WHERE commit_hash = ?', (commit.hash,))
            for file_change in commit.files_changed:
                cursor.execute('''
                    INSERT INTO file_changes 
                    (commit_hash, file_path, change_type, lines_added, lines_deleted, 
                     lines_changed, binary, old_path, new_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    commit.hash,
                    file_change.file_path,
                    file_change.change_type.value,
                    file_change.lines_added,
                    file_change.lines_deleted,
                    file_change.lines_changed,
                    file_change.binary,
                    file_change.old_path,
                    file_change.new_path
                ))
            
            # Store metadata if provided
            if metadata:
                cursor.execute('''
                    INSERT OR REPLACE INTO commit_metadata 
                    (commit_hash, tags, pull_request, issue_numbers, reviewers, 
                     ci_status, deployment_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    commit.hash,
                    json.dumps(metadata.tags),
                    metadata.pull_request,
                    json.dumps(metadata.issue_numbers),
                    json.dumps(metadata.reviewers),
                    metadata.ci_status,
                    metadata.deployment_status
                ))
            
            conn.commit()
    
    def get_commit(self, commit_hash: str) -> Optional[CommitInfo]:
        """Get a commit by hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM commits WHERE hash = ?', (commit_hash,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get file changes
            cursor.execute('SELECT * FROM file_changes WHERE commit_hash = ?', (commit_hash,))
            file_changes = []
            for fc_row in cursor.fetchall():
                file_changes.append(FileChange(
                    file_path=fc_row[2],
                    change_type=FileChangeType(fc_row[3]),
                    lines_added=fc_row[4],
                    lines_deleted=fc_row[5],
                    lines_changed=fc_row[6],
                    binary=bool(fc_row[7]),
                    old_path=fc_row[8],
                    new_path=fc_row[9]
                ))
            
            return CommitInfo(
                hash=row[0],
                short_hash=row[1],
                author=row[2],
                email=row[3],
                date=datetime.fromisoformat(row[4]),
                message=row[5],
                commit_type=CommitType(row[6]) if row[6] else None,
                files_changed=file_changes,
                merge=bool(row[8])
            )
    
    def get_commits(self, limit: Optional[int] = None, offset: int = 0) -> List[CommitInfo]:
        """Get commits with pagination"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM commits ORDER BY date DESC'
            params = []
            
            if limit:
                query += ' LIMIT ? OFFSET ?'
                params.extend([limit, offset])
            
            cursor.execute(query, params)
            commits = []
            
            for row in cursor.fetchall():
                # Get file changes for each commit
                cursor.execute('SELECT * FROM file_changes WHERE commit_hash = ?', (row[0],))
                file_changes = []
                for fc_row in cursor.fetchall():
                    file_changes.append(FileChange(
                        file_path=fc_row[2],
                        change_type=FileChangeType(fc_row[3]),
                        lines_added=fc_row[4],
                        lines_deleted=fc_row[5],
                        lines_changed=fc_row[6],
                        binary=bool(fc_row[7]),
                        old_path=fc_row[8],
                        new_path=fc_row[9]
                    ))
                
                commits.append(CommitInfo(
                    hash=row[0],
                    short_hash=row[1],
                    author=row[2],
                    email=row[3],
                    date=datetime.fromisoformat(row[4]),
                    message=row[5],
                    commit_type=CommitType(row[6]) if row[6] else None,
                    files_changed=file_changes,
                    merge=bool(row[8])
                ))
            
            return commits

class CommitStorage:
    """File-based storage for commit data"""
    
    def __init__(self, storage_path: str = "commit_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_commits(self, commits: List[CommitInfo], filename: str = "commits.json"):
        """Save commits to JSON file"""
        file_path = self.storage_path / filename
        
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
                        'lines_changed': fc.lines_changed,
                        'binary': fc.binary,
                        'old_path': fc.old_path,
                        'new_path': fc.new_path
                    } for fc in commit.files_changed
                ],
                'merge': commit.merge
            })
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_commits(self, filename: str = "commits.json") -> List[CommitInfo]:
        """Load commits from JSON file"""
        file_path = self.storage_path / filename
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        commits = []
        for item in data:
            commits.append(CommitInfo(
                hash=item['hash'],
                short_hash=item['short_hash'],
                author=item['author'],
                email=item['email'],
                date=datetime.fromisoformat(item['date']),
                message=item['message'],
                commit_type=CommitType(item['commit_type']) if item['commit_type'] else None,
                files_changed=[
                    FileChange(
                        file_path=fc['file_path'],
                        change_type=FileChangeType(fc['change_type']),
                        lines_added=fc['lines_added'],
                        lines_deleted=fc['lines_deleted'],
                        lines_changed=fc['lines_changed'],
                        binary=fc['binary'],
                        old_path=fc.get('old_path'),
                        new_path=fc.get('new_path')
                    ) for fc in item['files_changed']
                ],
                merge=item['merge']
            ))
        
        return commits

class CommitRepository:
    """Repository pattern for commit data access"""
    
    def __init__(self, db: CommitDatabase, storage: CommitStorage):
        self.db = db
        self.storage = storage
    
    def store_commit_history(self, commits: List[CommitInfo], metadata: List[CommitMetadata] = None):
        """Store complete commit history"""
        for i, commit in enumerate(commits):
            meta = metadata[i] if metadata and i < len(metadata) else None
            self.db.store_commit(commit, meta)
        
        # Also save to file storage
        self.storage.save_commits(commits)
    
    def get_commit_history(self, use_db: bool = True) -> List[CommitInfo]:
        """Get commit history from storage"""
        if use_db:
            return self.db.get_commits()
        else:
            return self.storage.load_commits()
    
    def search_commits(self, query: str) -> List[CommitInfo]:
        """Search commits by message or author"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM commits 
                WHERE message LIKE ? OR author LIKE ?
                ORDER BY date DESC
            ''', (f'%{query}%', f'%{query}%'))
            
            # Process results similar to get_commits
            # ... (implementation details)
            return []

