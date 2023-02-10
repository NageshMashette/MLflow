import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start a MLflow run
with mlflow.start_run():
    # Create and train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Log model parameters and metrics
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    
    # Log the model artifact in the run
    mlflow.sklearn.log_model(model, "model")
    
    # Get the run ID
    run_id = mlflow.active_run().info.run_id
    
# Load the model back from the run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Use the model to make predictions on new data
data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(data)
print(f"Predicted class: {prediction[0]}")
