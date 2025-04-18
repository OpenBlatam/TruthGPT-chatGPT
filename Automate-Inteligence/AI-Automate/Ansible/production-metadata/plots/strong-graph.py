import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


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
    else:
        model = RandomForestRegressor()

    model.fit(train.index.year.values.reshape(-1, 1), train['kpi'].values.ravel())
    preds = model.predict(test.index.year.values.reshape(-1, 1))

    error = mean_absolute_error(test['kpi'], preds)

    # Concat all data into a single DataFrame for plotting
    all_data = pd.concat([
        train.assign(Data='Train'),
        test.assign(Data='Test'),
        pd.DataFrame({'kpi': preds, 'Data': 'Prediction'}, index=test.index)])

    return all_data

parser = argparse.ArgumentParser(description='Train a Machine Learning model.')
parser.add_argument('--model', type=str, default='linear', choices=['linear', 'randomforest'],
                    help='Type of machine learning model to use.')

if __name__ == '__main__':
    # retrieve the command-line arguments
    args = parser.parse_args()

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
    all_data = train_model(df, args.model)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('KPI over Time'),
        dcc.Dropdown(id='model-choice',
                     options=[
                         {'label': 'Linear Regression', 'value': 'linear'},
                         {'label': 'Random Forest Regressor', 'value': 'randomforest'}
                     ],
                     value='linear'
                     ),
        dcc.Graph(id='live-graph', config={'displayModeBar': False}),
        html.Button('Run Model', id='run-button')
    ])


    @app.callback(
        Output('live-graph', 'figure'),
        Input('run-button', 'n_clicks'),
        State('model-choice', 'value')
    )
    def update_graph(clicks, model_choice):
        if clicks is not None:
            all_data = train_model(df, model_choice)
            fig = px.line(all_data, x=all_data.index, y='kpi', color='Data',
                          labels={'index': 'Date', 'kpi': 'KPI'},
                          template='plotly_dark')
            return fig


    app.run_server(debug=True)
