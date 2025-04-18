import openai
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import argparse
import matplotlib.pyplot as plt

# your OpenAI API key
openai.api_key = 'your-openai-api-key'


def interact_with_model(prompt):
    model_id = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def train_model(df, model_type):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    train = df[df.index.year < 2023]
    test = df[df.index.year >= 2023]

    if train.empty:
        print('Train DataFrame is empty.')
        print('Original DataFrame:')
        print(df)
        return
    if test.empty:
        print('Test DataFrame is empty.')
        print('Original DataFrame:')
        print(df)
        return

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'randomforest':
        model = RandomForestRegressor()

    model.fit(train.index.astype('int64').values.reshape(-1, 1), train['kpi'])
    prediction = model.predict(test.index.astype('int64').values.reshape(-1, 1))

    error = mean_absolute_error(test['kpi'], prediction)
    print('Mean Absolute Error:', error)

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['kpi'], label='Train')
    plt.plot(test.index, test['kpi'], label='Test')
    plt.plot(test.index, prediction, label='Prediction')
    plt.title('KPI Over Time')
    plt.legend(loc='upper left')

    plt.show()


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', type=str, help='Type of model (linear or randomforest).', default='linear')

if __name__ == '__main__':
    user_input = input("Please enter your request: ")

    user_command = user_input.split("with date ")

    if "of " in user_command[0]:
        network_name = str(user_command[0].split("of ")[1].strip())
    else:
        print("The command you entered is not in the correct format.")
        exit(1)

    date = str(user_command[1].strip())

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
    filtered_df = df[df['time'] == date]

    df = pd.DataFrame(data)
    filtered_df = df[df['time'] == user_command[1]]

    args = parser.parse_args()
    train_model(filtered_df, user_command[0])  # OpenAI's response is expected to be in the format "model date"
