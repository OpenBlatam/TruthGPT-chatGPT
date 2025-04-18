import argparse
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def train_model(df, model_type):
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    train = df[df.index.year < 2023]
    test = df[df.index.year >= 2023]

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'randomforest':
        model = RandomForestRegressor()

    model.fit(train.index.astype('int64').values.reshape(-1, 1), train['kpi'])
    prediction = model.predict(test.index.astype('int64').values.reshape(-1, 1))

    error = mean_absolute_error(test['kpi'], prediction)
    print('Mean Absolute Error:', error)

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=train.index, y=train['kpi'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test['kpi'], mode='lines', name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=prediction, mode='lines', name='Prediction'))

    fig.update_layout(
        title="KPI Over Time",
        xaxis_title="Time",
        yaxis_title="KPI"
    )

    fig.show()


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', type=str, help='Type of model (linear or randomforest).', default='linear')

if __name__ == '__main__':
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
    df = pd.DataFrame(data)
    args = parser.parse_args()
    train_model(df, args.model)