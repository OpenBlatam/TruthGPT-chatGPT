import requests
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Fetch network information
response = requests.get('http://your_network_api_endpoint')
data = response.json()

# Step 2: Process data and calculate KPIs
df = pd.DataFrame(data)
kpi = df['your_kpi_column'].mean()  # Replace with your actual KPI calculation

# Step 3: Create a graph of the KPI
plt.plot(df['time'], df['your_kpi_column'])
plt.title('KPI over Time')
plt.xlabel('Time')
plt.ylabel('KPI')
plt.show()

