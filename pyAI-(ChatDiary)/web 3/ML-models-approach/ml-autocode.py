import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('smart_contract_data.csv')

# Extract features and target variable
X = df.drop('vulnerability', axis=1)
y = df['vulnerability']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize H2O
h2o.init()

# Convert pandas dataframe to H2O dataframe
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

# Set target variable
train['vulnerability'] = train['vulnerability'].asfactor()
test['vulnerability'] = test['vulnerability'].asfactor()

# Use AutoML to train and select the best model
automl = H2OAutoML(max_models=10, seed=1)
automl.train(x=X.columns.tolist(), y='vulnerability', training_frame=train)

# Get the best model and make predictions on the testing set
best_model = automl.leader
y_pred = best_model.predict(test)
y_pred = y_pred.as_data_frame()['predict'].values

# Evaluate the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
