import datetime
import matplotlib.pyplot as plt
import requests

def fetch_kpi_data(system, site_id, date):
    if system == "KPIManager":
        api_endpoint = f"https://kpimanager.api.endpoint/site/{site_id}/{date}"
        response = requests.get(api_endpoint)
        if response.status_code == 200:
            return response.json()["kpiData"]
        else:
            print(f"Failed to fetch KPI data: {response.json()['error']}")
            return None

def plot_kpi_data(kpi_data, kpi):
    x_values = [datetime.datetime.strptime(item["timestamp"], "%Y-%m-%dT%H:%M:%S") for item in kpi_data]
    y_values = [item[kpi] for item in kpi_data]

    plt.plot(x_values, y_values)
    plt.xlabel('Timestamp')
    plt.ylabel(kpi)
    plt.title('KPI Graph')
    plt.show()

def execute_action_based_on_text(action, parameters):
    if action == "create_ticket":
        system = parameters.get("system")
        ticket_details = parameters.get("ticket_details")
        create_new_ticket(system, ticket_details)

    elif action == "update_node_status":
        system = parameters.get("system")
        node_id = parameters.get("node_id")
        new_status = parameters.get("new_status")
        update_node_status(system, node_id, new_status)

    elif action == "add_network_element":
        system = parameters.get("system")
        element_details = parameters.get("element_details")
        add_new_network_element(system, element_details)

    elif action == "get_network_statistics":
        system = parameters.get("system")
        parameters = parameters.get("parameters")
        get_network_statistics(system, parameters)

    # Add another action for "fetch_and_plot_kpi"
    elif action == "fetch_and_plot_kpi":
        system = parameters.get("system")
        site_id = parameters.get("site_id")
        date = parameters.get("date")
        kpi = parameters.get("kpi")
        kpi_data = fetch_kpi_data(system, site_id, date)
        if kpi_data is not None:
            plot_kpi_data(kpi_data, kpi)

    else:
        print(f"Action '{action}' not recognized.")