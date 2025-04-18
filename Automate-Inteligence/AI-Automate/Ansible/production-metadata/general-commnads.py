import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import argparse
from joblib import dump, load
import matplotlib.pyplot as plt

# ArgumentParser Setup
parser = argparse.ArgumentParser(description='Train a Machine Learning model.')
parser.add_argument('--url', type=str, required=True, help='URL of the data endpoint.')
parser.add_argument('--model', type=str, default='linear', choices=['linear', 'randomforest'],
                    help='Type of machine learning model to use.')


def train_model(url, model_type):
    # Fetch data from the API
    response = requests.get(url)
    data = response.json()

    # Process data into DataFrame.
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['kpi'] = pd.to_numeric(df['kpi'])

    # Prepare data for prediction.
    df['kpi_shifted'] = df['kpi'].shift(-1)
    df = df.dropna()

    # Prepare training data.
    x = df[['time', 'kpi']].values.reshape(-1, 1)
    y = df['kpi_shifted'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model.
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'randomforest':
        model = RandomForestRegressor()

    model.fit(x_train, y_train)

    # Save the trained model.
    dump(model, f'{model_type}_model.joblib')

    # Perform predictions.
    y_pred = model.predict(x_test)

    # Evaluate model using MSE.
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')

    # Plot actual vs predicted values.
    plt.figure(figsize=(10, 5))
    plt.scatter(x_test, y_test, color='black', label='Actual')
    plt.plot(x_test, y_pred, color='blue', linewidth=2, linestyle='-', label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('KPI')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    train_model(args.url, args.model)