"""
Commit Tracking System Demo
Demonstrates usage with sample commit data
"""

from datetime import datetime
from .core import CommitSystem
from .models import CommitInfo, CommitType, FileChange, FileChangeType, CommitFilter

def create_sample_commits():
    """Create sample commit data based on the provided information"""
    sample_data = [
        {
            'hash': 'abc123def456',
            'short_hash': 'abc123d',
            'author': 'Developer A',
            'email': 'dev-a@example.com',
            'date': '2023-11-24T10:30:00',
            'message': 'Update main branch post 24.11 (#7829)',
            'commit_type': 'chore',
            'files_changed': [
                {'file_path': 'CMakeLists.txt', 'change_type': 'modified', 'lines_added': 5, 'lines_deleted': 2}
            ]
        },
        {
            'hash': 'def456ghi789',
            'short_hash': 'def456g',
            'author': 'Developer B',
            'email': 'dev-b@example.com',
            'date': '2022-01-15T14:20:00',
            'message': 'Add support for forwarding HTTP/GRPC headers (#5575)',
            'commit_type': 'feature',
            'files_changed': [
                {'file_path': 'grpc_handler.h', 'change_type': 'modified', 'lines_added': 15, 'lines_deleted': 3}
            ]
        },
        {
            'hash': 'ghi789jkl012',
            'short_hash': 'ghi789j',
            'author': 'Developer C',
            'email': 'dev-c@example.com',
            'date': '2023-05-20T09:15:00',
            'message': 'feat: Add graceful shutdown timer to GRPC frontend (#7969)',
            'commit_type': 'feature',
            'files_changed': [
                {'file_path': 'grpc_server.cc', 'change_type': 'modified', 'lines_added': 25, 'lines_deleted': 8},
                {'file_path': 'grpc_server.h', 'change_type': 'modified', 'lines_added': 10, 'lines_deleted': 2}
            ]
        },
        {
            'hash': 'jkl012mno345',
            'short_hash': 'jkl012m',
            'author': 'Developer A',
            'email': 'dev-a@example.com',
            'date': '2022-03-10T16:45:00',
            'message': 'Adding client-side request cancellation support and testing (#6383)',
            'commit_type': 'feature',
            'files_changed': [
                {'file_path': 'grpc_utils.cc', 'change_type': 'modified', 'lines_added': 30, 'lines_deleted': 5}
            ]
        },
        {
            'hash': 'mno345pqr678',
            'short_hash': 'mno345p',
            'author': 'Developer D',
            'email': 'dev-d@example.com',
            'date': '2023-01-15T11:30:00',
            'message': 'feat: Add GRPC error codes to GRPC streaming if enabled by user. (#7499)',
            'commit_type': 'feature',
            'files_changed': [
                {'file_path': 'grpc_utils.h', 'change_type': 'modified', 'lines_added': 12, 'lines_deleted': 1}
            ]
        },
        {
            'hash': 'pqr678stu901',
            'short_hash': 'pqr678s',
            'author': 'Developer B',
            'email': 'dev-b@example.com',
            'date': '2023-08-15T13:20:00',
            'message': 'fix: Improve cancellation handling for gRPC non-decoupled inference (#...)',
            'commit_type': 'bugfix',
            'files_changed': [
                {'file_path': 'infer_handler.cc', 'change_type': 'modified', 'lines_added': 8, 'lines_deleted': 3}
            ]
        },
        {
            'hash': 'stu901vwx234',
            'short_hash': 'stu901v',
            'author': 'Developer C',
            'email': 'dev-c@example.com',
            'date': '2023-05-20T09:15:00',
            'message': 'feat: Add graceful shutdown timer to GRPC frontend (#7969)',
            'commit_type': 'feature',
            'files_changed': [
                {'file_path': 'infer_handler.h', 'change_type': 'modified', 'lines_added': 6, 'lines_deleted': 1}
            ]
        },
        {
            'hash': 'vwx234yza567',
            'short_hash': 'vwx234y',
            'author': 'Developer A',
            'email': 'dev-a@example.com',
            'date': '2023-05-20T09:15:00',
            'message': 'feat: Add graceful shutdown timer to GRPC frontend (#7969)',
            'commit_type': 'feature',
            'files_changed': [
                {'file_path': 'stream_infer_handler.cc', 'change_type': 'modified', 'lines_added': 20, 'lines_deleted': 4},
                {'file_path': 'stream_infer_handler.h', 'change_type': 'modified', 'lines_added': 8, 'lines_deleted': 2}
            ]
        }
    ]
    
    return sample_data

def demo_commit_tracking():
    """Demonstrate the commit tracking system"""
    print("=== Commit Tracking System Demo ===\n")
    
    # Initialize the system
    commit_system = CommitSystem()
    
    # Import sample data
    sample_data = create_sample_commits()
    commits = commit_system.setup_from_data(sample_data)
    
    print(f"Imported {len(commits)} commits\n")
    
    # Get basic statistics
    print("=== Basic Statistics ===")
    stats = commit_system.tracker.get_statistics()
    print(f"Total commits: {stats.total_commits}")
    print(f"Total files changed: {stats.files_changed}")
    print(f"Total lines added: {stats.lines_added}")
    print(f"Total lines deleted: {stats.lines_deleted}")
    print(f"Net lines: {stats.net_lines}")
    print(f"Average files per commit: {stats.average_files_per_commit:.2f}")
    print(f"Average lines per commit: {stats.average_lines_per_commit:.2f}\n")
    
    # Show commits by author
    print("=== Commits by Author ===")
    for author, count in stats.commits_by_author.items():
        print(f"{author}: {count} commits")
    print()
    
    # Show commits by type
    print("=== Commits by Type ===")
    for commit_type, count in stats.commits_by_type.items():
        print(f"{commit_type.value}: {count} commits")
    print()
    
    # Get analytics
    print("=== Advanced Analytics ===")
    analytics = commit_system.tracker.get_analytics()
    print(f"Commit velocity (30-day average): {analytics.commit_velocity:.2f} commits/day")
    print()
    
    # Show author activity
    print("=== Author Activity Details ===")
    for author, activity in analytics.author_activity.items():
        print(f"Author: {author}")
        print(f"  Total commits: {activity['total_commits']}")
        print(f"  Files changed: {activity['files_changed']}")
        print(f"  Lines added: {activity['lines_added']}")
        print(f"  Lines deleted: {activity['lines_deleted']}")
        print(f"  Commit frequency: {activity['commit_frequency']:.2f} commits/day")
        print()
    
    # Show file activity
    print("=== File Activity ===")
    for file_path, activity in analytics.file_activity.items():
        print(f"File: {file_path}")
        print(f"  Commits: {activity['commits']}")
        print(f"  Authors: {activity['unique_authors']}")
        print(f"  Lines added: {activity['lines_added']}")
        print(f"  Lines deleted: {activity['lines_deleted']}")
        print()
    
    # Demonstrate filtering
    print("=== Filtering Examples ===")
    
    # Filter by author
    author_filter = CommitFilter(authors=['Developer A'])
    author_commits = commit_system.manager.filter_commits(author_filter)
    print(f"Commits by Developer A: {len(author_commits)}")
    
    # Filter by commit type
    feature_filter = CommitFilter(commit_types=[CommitType.FEATURE])
    feature_commits = commit_system.manager.filter_commits(feature_filter)
    print(f"Feature commits: {len(feature_commits)}")
    
    # Filter by file
    grpc_filter = CommitFilter(files=['grpc_server.cc'])
    grpc_commits = commit_system.manager.filter_commits(grpc_filter)
    print(f"Commits affecting grpc_server.cc: {len(grpc_commits)}")
    print()
    
    # Show specific author summary
    print("=== Developer A Summary ===")
    author_summary = commit_system.manager.get_author_summary('Developer A')
    if author_summary:
        print(f"Total commits: {author_summary['total_commits']}")
        print(f"Total files: {author_summary['total_files']}")
        print(f"Total lines: {author_summary['total_lines']}")
        print(f"Active days: {author_summary['active_days']}")
        print(f"Commit types: {author_summary['commit_types']}")
    print()
    
    # Show specific file summary
    print("=== grpc_server.cc Summary ===")
    file_summary = commit_system.manager.get_file_summary('grpc_server.cc')
    if file_summary:
        print(f"Total commits: {file_summary['total_commits']}")
        print(f"Lines added: {file_summary['total_lines_added']}")
        print(f"Lines deleted: {file_summary['total_lines_deleted']}")
        print(f"Net lines: {file_summary['net_lines']}")
        print(f"Authors: {file_summary['authors']}")
    print()
    
    # Generate report
    print("=== Generated Report ===")
    report = commit_system.generate_report()
    print(report)
    
    # Export data
    print("=== Export Data ===")
    json_export = commit_system.export_data('json')
    print(f"JSON export length: {len(json_export)} characters")
    
    csv_export = commit_system.export_data('csv')
    print(f"CSV export length: {len(csv_export)} characters")
    print("\nCSV Preview:")
    print(csv_export[:500] + "..." if len(csv_export) > 500 else csv_export)

if __name__ == "__main__":
    demo_commit_tracking()

