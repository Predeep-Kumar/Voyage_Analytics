import mlflow
import os
from mlflow.tracking import MlflowClient

from regression import run_regression_pipeline
from classification import run_classification_pipeline

import warnings
warnings.filterwarnings("ignore")


PROCESSED_DATA_PATH = "./data/processed"

print("Initializing MLflow tracking...")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print("MLflow tracking URI set")


# RUN PIPELINES
print("\nStarting regression pipeline...")
run_regression_pipeline(PROCESSED_DATA_PATH)
print("Regression pipeline completed")

print("\nStarting classification pipeline...")
run_classification_pipeline(PROCESSED_DATA_PATH)
print("Classification pipeline completed")


client = MlflowClient()


# REGISTER BEST MODEL
def register_best_model(experiment_name, model_name):

    print(f"\nSearching best run for experiment: {experiment_name}")

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.score DESC"]
    )

    if not runs:
        raise ValueError(f"No runs found for experiment: {experiment_name}")

    best_run = runs[0]
    run_id = best_run.info.run_id
    score = best_run.data.metrics.get("score", None)

    print(f"Best run selected")
    print(f"Run ID: {run_id}")
    print(f"Score: {score}")

    model_uri = f"runs:/{run_id}/model"

    print(f"Registering model: {model_name}")
    registered_model = mlflow.register_model(model_uri, model_name)

    print(f"Model registered successfully")
    print(f"Model name: {model_name}")
    print(f"Version: {registered_model.version}")

    return model_name, registered_model.version


# PROMOTE MODEL
def promote_model(model_name, version):

    print(f"\nPromoting model {model_name} version {version}")

    print("Transitioning to Staging...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print("Stage updated to Staging")

    print("Transitioning to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print("Stage updated to Production")


# REGRESSION MODEL
print("\nProcessing regression model...")

reg_model_name, reg_version = register_best_model(
    experiment_name="Flight Price Prediction",
    model_name="FlightPriceModel"
)

promote_model(reg_model_name, reg_version)

print("Regression model is now in Production")


# CLASSIFICATION MODEL
print("\nProcessing classification model...")

clf_model_name, clf_version = register_best_model(
    experiment_name="User Gender Prediction",
    model_name="GenderPredictionModel"
)

promote_model(clf_model_name, clf_version)

print("Classification model is now in Production")


print("\nPipeline completed. All models are in Production")