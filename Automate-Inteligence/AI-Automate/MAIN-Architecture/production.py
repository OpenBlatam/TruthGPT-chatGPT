
def execute_action_based_on_text(action, parameters):

    # Log the action and parameters
    log_action(action, parameters)

    # map the action to its function
    action_handler = {
        "create_ticket": handle_action_create_ticket,
        "update_node_status": handle_action_update_node_status,
        "add_network_element": handle_action_add_network_element,
        "delete_network_element": delete_network_element,
        "get_network_statistics": handle_action_get_network_statistics,
        "fetch_and_plot_kpi": handle_action_fetch_and_plot_kpi,
        "display_network_summary": display_network_summary,
        "check_connection_status": check_connection_status,
        "restart_network_node": restart_network_node,
        "check_node_update": check_node_update,
        "update_network_node": update_network_node,
        "scale_network_node": scale_network_node,
        "backup_network_node": backup_network_node,  # new action added
    }

    func = action_handler.get(action, None)
    if func is not None:
        func(**parameters)
    else:
        print(f"Action '{action}' not recognized.")