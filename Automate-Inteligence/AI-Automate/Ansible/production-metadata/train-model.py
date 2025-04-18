import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import argparse

# Define ArgumentParser
parser = argparse.ArgumentParser(description='Train a Linear Regression model.')
parser.add_argument('--url', type=str, help='URL of the data endpoint')

def train_model(url):
    # Assume data is a time series data with date and KPI value
    response = requests.get(url)
    data = response.json()

    # Processing data into a DataFrame
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['kpi'] = pd.to_numeric(df['kpi'])

    # Shifting KPI for prediction
    df['kpi_shifted'] = df['kpi'].shift(-1)
    df = df.dropna()

    # Preparing data for training
    x = df[['time', 'kpi']].values.reshape(-1, 1)
    y = df['kpi_shifted'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Training a model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Making predictions
    y_pred = model.predict(x_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')

if __name__ == '__main__':
    args = parser.parse_args()
    train_model(args.url)