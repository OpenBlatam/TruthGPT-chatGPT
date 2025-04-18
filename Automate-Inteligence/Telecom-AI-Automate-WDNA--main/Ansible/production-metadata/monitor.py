from sklearn.metrics import mean_squared_error

# Threshold for retraining
THRESHOLD = 10.0

def monitor_performance(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    if mse > THRESHOLD:
        return False  # performance is bad, consider retraining
    else:
        return True  # performance is acceptable

# Use the function in the pipeline after getting predictions
def serve_request(request):
    # Assign a model for the incoming request
    model = next(model_selector)

    # Preprocess the request, get data and perform prediction...
    # Let's assume the preprocess function return the features (X) and the actual value (y_test)
    X, y_test = preprocess(request)

    y_pred = model.predict(X)

    # Monitor performance
    if not monitor_performance(y_test, y_pred):
        logger.warning('Model performance is bad. Consider retraining.')

    # Further steps including prediction and response
    # ...