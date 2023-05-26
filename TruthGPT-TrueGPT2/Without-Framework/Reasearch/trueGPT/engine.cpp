#include <iostream>
#include <vector>
#include <memory>

using namespace std;

// Forward declarations
class BaseEstimator;
class DataPoint;
class StandardScaler;
class LinearRegression;

class Pipeline {
public:
  Pipeline() {}

  void addStep(string name, shared_ptr<BaseEstimator> estimator) {
    steps.push_back({name, estimator});
  }

  void fit(vector<DataPoint>& data) {
    for (auto& step : steps) {
      step.second->fit(data);
    }
  }

  vector<Prediction> predict(const vector<DataPoint>& data) {
    vector<Prediction> predictions;
    predictions.reserve(data.size());

    bool needsTransform = true;
    for (auto& step : steps) {
      if (needsTransform) {
        data = step.second->transform(data);
      }
      needsTransform = step.second->requiresTransform();

      for (const auto& point : data) {
        predictions.push_back(step.second->predict(point));
      }
    }

    return predictions;
  }

private:
  vector<pair<string, shared_ptr<BaseEstimator>>> steps;
};

int main() {
  vector<DataPoint> data = {
    {1.0, 2.0},
    {3.0, 4.0},
    {5.0, 6.0},
  };

  Pipeline pipeline;
  pipeline.addStep("scaler", make_shared<StandardScaler>());
  pipeline.addStep("regressor", make_shared<LinearRegression>());

  pipeline.fit(data);

  vector<Prediction> predictions = pipeline.predict(data);

  for (const auto& prediction : predictions) {
    cout << prediction.prediction << endl;
  }

  return 0;
}

