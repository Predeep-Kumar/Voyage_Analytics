import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

models = ["FlightPriceModel", "GenderPredictionModel"]

for model in models:
    try:
        client.delete_registered_model(model)
        print(f"{model} deleted ✅")
    except Exception as e:
        print(f"Error deleting {model}: {e}")