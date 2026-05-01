"""
Simple test script for the commit tracking system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import CommitSystem
from models import CommitType, CommitFilter

def test_basic_functionality():
    """Test basic functionality of the commit tracking system"""
    print("Testing Commit Tracking System...")
    
    # Sample commit data
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
        }
    ]
    
    try:
        # Initialize system
        commit_system = CommitSystem()
        
        # Import data
        commits = commit_system.setup_from_data(sample_data)
        print(f"✓ Imported {len(commits)} commits")
        
        # Test statistics
        stats = commit_system.tracker.get_statistics()
        print(f"✓ Total commits: {stats.total_commits}")
        print(f"✓ Total files: {stats.files_changed}")
        print(f"✓ Lines added: {stats.lines_added}")
        
        # Test filtering
        author_filter = CommitFilter(authors=['Developer A'])
        filtered_commits = commit_system.manager.filter_commits(author_filter)
        print(f"✓ Filtered commits by author: {len(filtered_commits)}")
        
        # Test analytics
        analytics = commit_system.tracker.get_analytics()
        print(f"✓ Commit velocity: {analytics.commit_velocity:.2f}")
        
        # Test export
        json_export = commit_system.export_data('json')
        print(f"✓ JSON export: {len(json_export)} characters")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)




