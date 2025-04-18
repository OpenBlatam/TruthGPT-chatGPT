import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def train_model(df: pd.DataFrame, model_type: str):
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    train = df[df.index.year < 2023]
    test = df[df.index.year >= 2023]

    if len(test) == 0:
        raise ValueError(
            "No data available for year 2023 and onwards for testing. Please adjust the year or provide more data.")

    if model_type == 'linear':
        model = LinearRegression()

    if model_type == 'randomforest':
        model = RandomForestRegressor()

    model.fit(train.index.astype('int64').values.reshape(-1, 1), train['kpi'].values.ravel())
    preds = model.predict(test.index.astype('int64').values.reshape(-1, 1))

    error = mean_absolute_error(test['kpi'], preds)

    # Plotting with Matplotlib instead of Bokeh
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train['kpi'], label='Train', color='navy')
    plt.plot(test.index, test['kpi'], label='Test', color='green')
    plt.plot(test.index, preds, label='Prediction', color='red')
    plt.title('KPI over time')
    plt.xlabel('Date')
    plt.ylabel('KPI')
    plt.legend()
    plt.show()


parser = argparse.ArgumentParser(description='Train a Machine Learning model.')
parser.add_argument('--model', type=str, default='linear', choices=['linear', 'randomforest'],
                    help='Type of machine learning model to use.')

if __name__ == '__main__':
    # Sample data
    data = [
        {"time": "2021-02-16T20:00:00", "kpi": 12300.0},
        {"time": "2021-03-16T20:00:00", "kpi": 12350.0},
        {"time": "2021-04-16T20:00:00", "kpi": 12335.0},
        {"time": "2021-04-17T20:00:00", "kpi": 12123.0},
        {"time": "2022-04-18T20:00:00", "kpi": 12765.0},
        {"time": "2023-04-19T20:00:00", "kpi": 13000.0},
        {"time": "2023-05-20T20:00:00", "kpi": 13200.0},
        {"time": "2023-06-21T20:00:00", "kpi": 13300.0}
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    args = parser.parse_args()
    train_model(df, args.model)