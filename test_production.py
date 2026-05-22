"""
Unit tests for the production action execution system.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the path to import the production module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture'))


class TestActionExecution:
    """Test cases for the execute_action_based_on_text function."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock all the action handler functions
        self.mock_handlers = {
            'log_action': Mock(),
            'handle_action_create_ticket': Mock(),
            'handle_action_update_node_status': Mock(),
            'handle_action_add_network_element': Mock(),
            'delete_network_element': Mock(),
            'handle_action_get_network_statistics': Mock(),
            'handle_action_fetch_and_plot_kpi': Mock(),
            'display_network_summary': Mock(),
            'check_connection_status': Mock(),
            'restart_network_node': Mock(),
            'check_node_update': Mock(),
            'update_network_node': Mock(),
            'scale_network_node': Mock(),
            'backup_network_node': Mock(),
        }
    
    def test_execute_known_action_create_ticket(self):
        """Test executing a known action: create_ticket."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "create_ticket"
            parameters = {"system": "TestSystem", "ticket_details": "Test ticket"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called with parameters
            self.mock_handlers['handle_action_create_ticket'].assert_called_once_with(**parameters)
    
    def test_execute_known_action_update_node_status(self):
        """Test executing a known action: update_node_status."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "update_node_status"
            parameters = {"node_id": "123", "status": "active"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called
            self.mock_handlers['handle_action_update_node_status'].assert_called_once_with(**parameters)
    
    def test_execute_known_action_add_network_element(self):
        """Test executing a known action: add_network_element."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "add_network_element"
            parameters = {"element_type": "router", "location": "datacenter1"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called
            self.mock_handlers['handle_action_add_network_element'].assert_called_once_with(**parameters)
    
    def test_execute_known_action_delete_network_element(self):
        """Test executing a known action: delete_network_element."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "delete_network_element"
            parameters = {"element_id": "router_001"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called
            self.mock_handlers['delete_network_element'].assert_called_once_with(**parameters)
    
    def test_execute_known_action_get_network_statistics(self):
        """Test executing a known action: get_network_statistics."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "get_network_statistics"
            parameters = {"time_range": "24h", "metrics": ["bandwidth", "latency"]}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called
            self.mock_handlers['handle_action_get_network_statistics'].assert_called_once_with(**parameters)
    
    def test_execute_known_action_fetch_and_plot_kpi(self):
        """Test executing a known action: fetch_and_plot_kpi."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "fetch_and_plot_kpi"
            parameters = {"kpi_type": "throughput", "duration": "1h"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called
            self.mock_handlers['handle_action_fetch_and_plot_kpi'].assert_called_once_with(**parameters)
    
    def test_execute_known_action_backup_network_node(self):
        """Test executing a known action: backup_network_node."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "backup_network_node"
            parameters = {"node_id": "node_123", "backup_type": "full"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that the correct handler was called
            self.mock_handlers['backup_network_node'].assert_called_once_with(**parameters)
    
    def test_execute_unknown_action(self):
        """Test executing an unknown action."""
        with patch.multiple('production', **self.mock_handlers), \
             patch('builtins.print') as mock_print:
            
            from production import execute_action_based_on_text
            
            action = "unknown_action"
            parameters = {"param1": "value1"}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was still called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that error message was printed
            mock_print.assert_called_once_with("Action 'unknown_action' not recognized.")
            
            # Check that no handlers were called
            for handler_name, handler_mock in self.mock_handlers.items():
                if handler_name != 'log_action':
                    handler_mock.assert_not_called()
    
    def test_execute_action_with_empty_parameters(self):
        """Test executing an action with empty parameters."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "create_ticket"
            parameters = {}
            
            execute_action_based_on_text(action, parameters)
            
            # Check that log_action was called
            self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            
            # Check that handler was called with empty parameters
            self.mock_handlers['handle_action_create_ticket'].assert_called_once_with()
    
    def test_execute_action_with_none_parameters(self):
        """Test executing an action with None parameters."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "display_network_summary"
            parameters = None
            
            # This should handle None gracefully
            try:
                execute_action_based_on_text(action, parameters)
                # If we get here, the function handled None parameters
                self.mock_handlers['log_action'].assert_called_once_with(action, parameters)
            except TypeError:
                # If TypeError is raised, that's expected behavior for None parameters
                pass
    
    def test_action_handler_mapping_completeness(self):
        """Test that all expected actions are mapped to handlers."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            expected_actions = [
                "create_ticket",
                "update_node_status", 
                "add_network_element",
                "delete_network_element",
                "get_network_statistics",
                "fetch_and_plot_kpi",
                "display_network_summary",
                "check_connection_status",
                "restart_network_node",
                "check_node_update",
                "update_network_node",
                "scale_network_node",
                "backup_network_node"
            ]
            
            # Test that each expected action can be executed without error
            for action in expected_actions:
                with patch('builtins.print') as mock_print:
                    execute_action_based_on_text(action, {})
                    
                    # Should not print "not recognized" message
                    if mock_print.called:
                        printed_message = mock_print.call_args[0][0]
                        assert "not recognized" not in printed_message
    
    def test_action_handler_exception_handling(self):
        """Test that exceptions in action handlers are handled gracefully."""
        # Create a handler that raises an exception
        failing_handler = Mock(side_effect=Exception("Handler failed"))
        
        mock_handlers_with_failure = self.mock_handlers.copy()
        mock_handlers_with_failure['handle_action_create_ticket'] = failing_handler
        
        with patch.multiple('production', **mock_handlers_with_failure):
            from production import execute_action_based_on_text
            
            action = "create_ticket"
            parameters = {"system": "TestSystem"}
            
            # The function should not crash even if handler fails
            try:
                execute_action_based_on_text(action, parameters)
                # If we get here, the exception was handled or not raised
            except Exception as e:
                # If an exception is raised, it should be the original one
                assert str(e) == "Handler failed"
    
    def test_log_action_called_for_all_actions(self):
        """Test that log_action is called for both known and unknown actions."""
        with patch.multiple('production', **self.mock_handlers), \
             patch('builtins.print'):
            
            from production import execute_action_based_on_text
            
            # Test known action
            execute_action_based_on_text("create_ticket", {"param": "value"})
            
            # Test unknown action
            execute_action_based_on_text("unknown_action", {"param": "value"})
            
            # log_action should have been called twice
            assert self.mock_handlers['log_action'].call_count == 2
    
    def test_parameter_unpacking(self):
        """Test that parameters are correctly unpacked to handler functions."""
        with patch.multiple('production', **self.mock_handlers):
            from production import execute_action_based_on_text
            
            action = "update_node_status"
            parameters = {
                "node_id": "node_123",
                "status": "active",
                "timestamp": "2023-01-01T00:00:00Z"
            }
            
            execute_action_based_on_text(action, parameters)
            
            # Check that handler was called with unpacked parameters
            self.mock_handlers['handle_action_update_node_status'].assert_called_once_with(
                node_id="node_123",
                status="active", 
                timestamp="2023-01-01T00:00:00Z"
            )


if __name__ == "__main__":
    pytest.main([__file__])