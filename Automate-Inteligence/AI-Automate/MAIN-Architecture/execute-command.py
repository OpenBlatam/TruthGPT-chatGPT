import openai
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime

# Assume data is a dictionary with csv file names as keys and corresponding DataFrames as values
datasets = {}

def read_and_group_data(file_path, group_by_column_idx=1):
    df = pd.read_csv(file_path)
    return df.groupby(df.columns[group_by_column_idx])

def process_worksheet(data):
    # Replace 'kpi1' with some logic to retrieve/calculate the actual KPI name.
    kpi_name = 'kpi1'
    datasets[kpi_name] = data

xlsx_file = glob.glob('C:\\*\\*.xlsx')[0]
csv_file = os.path.splitext(xlsx_file)[0] + '.csv'

# Convert Excel file to CSV
# Using pandas to convert excel to csv, because pandas can handle large data and provide more functionality
data_xls = pd.read_excel(xlsx_file, index_col=None)
data_xls.to_csv(csv_file, encoding='utf-8', index=False)

# Read the csv file and group the data
grouped_data = read_and_group_data(csv_file)

# Process each dataframe
for name, dataframe in grouped_data:
    process_worksheet(dataframe)

# At this point, all datasets have been read and processed

def plot_data(data):
    plt.plot(data)
    plt.show()

def calculate_mean(data):
    print(f"Mean value: {data.mean().item()}")

def calculate_median(data):
    print(f"Median value: {data.median().item()}")

def calculate_std(data):
    print(f"Standard Deviation: {data.std().item()}")

def date_filter(data, start_date, end_date):
    return data[(data >= start_date) & (data <= end_date)]

def execute_command(prompt):

    action_map = {
        'generate_plot': plot_data,
        'calculate_mean': calculate_mean,
        'calculate_median': calculate_median,
        'calculate_std': calculate_std,
    }

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=50
    )

    action = response.choices[0].text.strip()

    if 'Generate a plot of' in action:
        words = action.split(' ')
        kpi_name = words[4]
        # Here, you would need to adjust this strptime formats to match the date format used in your command
        start_date = datetime.strptime(words[6], '%d %B')
        end_date = datetime.today() if words[8] == 'to date' else datetime.strptime(words[8], '%d %B')

        if kpi_name in datasets:
            data = datasets[kpi_name]
            # Assuming data['date'] contains the date. Replace 'date' with the date column name in your csv
            filtered_data = date_filter(data['date'], start_date, end_date)

            action_map['generate_plot'](filtered_data)
        else:
            print(f"Sorry, I don't have data for {kpi_name}.")
    elif 'Calculate the mean for' in action:
        words = action.split(' ')
        kpi_name = words[5]
        start_date = datetime.strptime(words[7], '%d %B')
        end_date = datetime.today() if words[9] == 'to date' else datetime.strptime(words[9], '%d %B')

        if kpi_name in datasets:
            data = datasets[kpi_name]
            filtered_data = date_filter(data['date'], start_date, end_date)

            action_map['calculate_mean'](filtered_data)
        else:
            print(f"Sorry, I don't have data for {kpi_name}.")
    else:
        print("Sorry, I could not understand your command.")

# Example usage
execute_command("I want a graph of kpi1 from 26 April to date")
execute_command("Calculate the mean for kpi1 from 26 April to date")