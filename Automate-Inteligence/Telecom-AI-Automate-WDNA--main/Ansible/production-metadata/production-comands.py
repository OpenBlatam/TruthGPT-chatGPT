import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import argparse
import logging
from joblib import dump, load

logging.basicConfig(filename='training.log', level=logging.INFO)

parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--url', type=str, default='http://example.com', help='URL of the data endpoint')
parser.add_argument('--model', type=str, default='randomforest', choices=['linear', 'randomforest'],
                    help='Type of machine learning model to use')

def add_datetime_features(df):
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    return df

def normalize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def handle_missing_values(df):
    for column in df.columns:
        df[column] = df[column].fillna(df[column].mean())
    return df

def check_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse}')
    print(f'R2: {r2}')
    logging.info(f'MSE: {mse}')
    logging.info(f'R2: {r2}')

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10,5))
        sorted_idx = model.feature_importances_.argsort()
        plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.show()

def save_predictions(y_test, y_pred):
    df_pred = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
    df_pred.to_csv('predictions.csv', index=False)

def cross_validation(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    logging.info(f'CV scores: {cv_scores}')
    logging.info(f'CV average score: {cv_scores.mean()}')

def tune_hyperparameters(model, params, X, y):
    gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
    gs.fit(X, y)
    logging.info(f'Best parameters: {gs.best_params_}')
    return gs.best_estimator_

def train_model(url, model_type):
    # Fetch data
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)

    df['time'] = pd.to_datetime(df['time'])
    df['kpi'] = pd.to_numeric(df['kpi'])
    df = handle_missing_values(df)
    df = add_datetime_features(df)

    X = df.drop(columns=['time','kpi_shifted']).values
    y = df['kpi_shifted'].values
    X, scaler = normalize_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    forest_params = {
      'n_estimators': [50, 100, 200, 300],
      'max_depth': [None, 5, 10, 15],
    }

    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == 'randomforest':
        model = RandomForestRegressor()
        model = tune_hyperparameters(model, forest_params, X_train, y_train)

    model.fit(X_train, y_train)

    dump(model, f'{model_type}_model.joblib')
    dump(scaler, 'scaler.joblib')

    check_model_performance(model, X_test, y_test)
    plot_feature_importance(model, df.drop(columns=['time','kpi_shifted']).columns)

if __name__ == '__main__':
    args = parser.parse_args()
    train_model(args.url, args.model)