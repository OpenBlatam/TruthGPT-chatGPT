"""
Unit tests for the CommandManager class.
"""
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from abc import ABC, abstractmethod

# Add the path to import the manager module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture', 'NLP'))

# Create a mock command interface for testing
class Command(ABC):
    """Abstract base class for commands."""
    
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass


class MockCommand(Command):
    """Mock command for testing."""
    
    def __init__(self, name="test_command"):
        self.name = name
        self.executed = False
        self.undone = False
        self.execute_count = 0
        self.undo_count = 0
    
    def execute(self):
        self.executed = True
        self.execute_count += 1
    
    def undo(self):
        self.undone = True
        self.undo_count += 1


class FailingCommand(Command):
    """Command that fails during execution for testing error handling."""
    
    def execute(self):
        raise Exception("Execution failed")
    
    def undo(self):
        raise Exception("Undo failed")


class TestCommandManager:
    """Test cases for the CommandManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Import here to avoid import issues
        from manager import CommandManager
        self.command_manager = CommandManager()
        self.temp_dir = tempfile.mkdtemp()
        
        # Change to temp directory for pickle files
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Change back to original directory and clean up temp files
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_command_manager_initialization(self):
        """Test that CommandManager initializes correctly."""
        assert self.command_manager.command_queue is not None
        assert self.command_manager.command_stack is not None
        assert self.command_manager.command_queue.empty()
        assert self.command_manager.command_stack.empty()
    
    def test_add_single_command(self):
        """Test adding a single command to the manager."""
        command = MockCommand("test1")
        
        with patch.object(self.command_manager, 'save_command_state') as mock_save:
            self.command_manager.add_command(command)
            
            # Check that command was added to queue
            assert not self.command_manager.command_queue.empty()
            
            # Check that save_command_state was called
            mock_save.assert_called_once_with(command)
    
    def test_add_multiple_commands(self):
        """Test adding multiple commands to the manager."""
        commands = [MockCommand(f"test{i}") for i in range(3)]
        
        with patch.object(self.command_manager, 'save_command_state'):
            for command in commands:
                self.command_manager.add_command(command)
            
            # Check that all commands were added
            assert self.command_manager.command_queue.qsize() == 3
    
    def test_execute_single_command(self):
        """Test executing a single command."""
        command = MockCommand("test1")
        
        with patch.object(self.command_manager, 'save_command_state'):
            self.command_manager.add_command(command)
            self.command_manager.execute_commands()
            
            # Check that command was executed
            assert command.executed
            assert command.execute_count == 1
            
            # Check that command was moved to stack
            assert self.command_manager.command_queue.empty()
            assert not self.command_manager.command_stack.empty()
    
    def test_execute_multiple_commands(self):
        """Test executing multiple commands in order."""
        commands = [MockCommand(f"test{i}") for i in range(3)]
        
        with patch.object(self.command_manager, 'save_command_state'):
            for command in commands:
                self.command_manager.add_command(command)
            
            self.command_manager.execute_commands()
            
            # Check that all commands were executed
            for command in commands:
                assert command.executed
                assert command.execute_count == 1
            
            # Check that all commands were moved to stack
            assert self.command_manager.command_queue.empty()
            assert self.command_manager.command_stack.qsize() == 3
    
    def test_execute_command_with_exception(self):
        """Test handling of command execution exceptions."""
        failing_command = FailingCommand()
        normal_command = MockCommand("normal")
        
        with patch.object(self.command_manager, 'save_command_state'):
            self.command_manager.add_command(failing_command)
            self.command_manager.add_command(normal_command)
            
            # Execute commands - should handle exception gracefully
            with patch('logging.error') as mock_log:
                self.command_manager.execute_commands()
                
                # Check that error was logged
                mock_log.assert_called()
                
                # Check that normal command was still executed
                assert normal_command.executed
    
    def test_undo_last_command(self):
        """Test undoing the last executed command."""
        command = MockCommand("test1")
        
        with patch.object(self.command_manager, 'save_command_state'), \
             patch.object(self.command_manager, 'load_command_state') as mock_load:
            
            self.command_manager.add_command(command)
            self.command_manager.execute_commands()
            
            # Undo the last command
            self.command_manager.undo_last_command()
            
            # Check that command was undone
            assert command.undone
            assert command.undo_count == 1
            
            # Check that load_command_state was called
            mock_load.assert_called_once_with(command)
    
    def test_undo_with_empty_stack(self):
        """Test undoing when no commands have been executed."""
        # Should handle empty stack gracefully
        self.command_manager.undo_last_command()
        
        # No exception should be raised
        assert self.command_manager.command_stack.empty()
    
    def test_undo_with_exception(self):
        """Test handling of undo exceptions."""
        failing_command = FailingCommand()
        
        with patch.object(self.command_manager, 'save_command_state'), \
             patch.object(self.command_manager, 'load_command_state'):
            
            self.command_manager.add_command(failing_command)
            
            # Manually add to stack to simulate executed command
            self.command_manager.command_stack.put(failing_command)
            
            # Undo should handle exception gracefully
            with patch('logging.error') as mock_log:
                self.command_manager.undo_last_command()
                
                # Check that error was logged
                mock_log.assert_called()
    
    def test_save_command_state(self):
        """Test saving command state to pickle file."""
        command = MockCommand("test_save")
        
        # Save command state
        self.command_manager.save_command_state(command)
        
        # Check that pickle file was created
        expected_file = f"{command.__class__.__name__}_state.pickle"
        assert os.path.exists(expected_file)
    
    def test_load_command_state(self):
        """Test loading command state from pickle file."""
        command = MockCommand("test_load")
        command.executed = True
        command.execute_count = 5
        
        # Save and then load command state
        self.command_manager.save_command_state(command)
        
        # Create a new command and load state into it
        new_command = MockCommand("new_command")
        self.command_manager.load_command_state(new_command)
        
        # Check that state was loaded correctly
        assert new_command.executed == True
        assert new_command.execute_count == 5
    
    def test_command_queue_fifo_order(self):
        """Test that commands are executed in FIFO order."""
        execution_order = []
        
        class OrderTrackingCommand(Command):
            def __init__(self, name):
                self.name = name
            
            def execute(self):
                execution_order.append(self.name)
            
            def undo(self):
                pass
        
        commands = [OrderTrackingCommand(f"cmd{i}") for i in range(3)]
        
        with patch.object(self.command_manager, 'save_command_state'):
            for command in commands:
                self.command_manager.add_command(command)
            
            self.command_manager.execute_commands()
            
            # Check execution order
            assert execution_order == ["cmd0", "cmd1", "cmd2"]
    
    def test_command_stack_lifo_order(self):
        """Test that commands are undone in LIFO order."""
        undo_order = []
        
        class UndoTrackingCommand(Command):
            def __init__(self, name):
                self.name = name
            
            def execute(self):
                pass
            
            def undo(self):
                undo_order.append(self.name)
        
        commands = [UndoTrackingCommand(f"cmd{i}") for i in range(3)]
        
        with patch.object(self.command_manager, 'save_command_state'), \
             patch.object(self.command_manager, 'load_command_state'):
            
            for command in commands:
                self.command_manager.add_command(command)
            
            self.command_manager.execute_commands()
            
            # Undo all commands
            for _ in range(3):
                self.command_manager.undo_last_command()
            
            # Check undo order (should be reverse of execution order)
            assert undo_order == ["cmd2", "cmd1", "cmd0"]


if __name__ == "__main__":
    pytest.main([__file__])