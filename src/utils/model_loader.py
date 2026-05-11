import mlflow
import mlflow.pyfunc
import joblib
from mlflow.tracking import MlflowClient

from src.config.paths import REGRESSION_MODEL_URI, CLASSIFICATION_MODEL_URI

import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


client = MlflowClient()


def load_auto_model(task):
    if task == "regression":
        return mlflow.pyfunc.load_model(REGRESSION_MODEL_URI)
    elif task == "classification":
        return mlflow.pyfunc.load_model(CLASSIFICATION_MODEL_URI)
    else:
        raise ValueError(f"Invalid task: {task}")


def load_manual_model(task, model_name):

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    exp_map = {
        "regression": "Flight Price Prediction",
        "classification": "User Gender Prediction"
    }

    experiment_name = exp_map.get(task)

    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise Exception(f"Experiment not found: {experiment_name}")

    runs = client.search_runs([experiment.experiment_id])

    print(f"\n[DEBUG] Searching in Experiment: {experiment_name}")
    print(f"[DEBUG] Looking for model: {model_name}")

    for run in runs:

        run_model = run.data.params.get("model")

        print(f"[DEBUG] Found model: {run_model}")

        if run_model == model_name:

            print(f"[DEBUG] ✅ MATCH FOUND: {run_model}")

            model_uri = f"runs:/{run.info.run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)

            return model, {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "version": "manual",
                "params": run.data.params
            }

    raise Exception(f"Model '{model_name}' not found in {experiment_name}")


def load_auto_model_with_info(task):
    if task == "regression":
        model_uri = REGRESSION_MODEL_URI
    elif task == "classification":
        model_uri = CLASSIFICATION_MODEL_URI
    else:
        raise ValueError(f"Invalid task: {task}")

    model = mlflow.pyfunc.load_model(model_uri)

    parts = model_uri.split("/")
    model_name = parts[1]
    stage = parts[2]

    versions = client.get_latest_versions(model_name, stages=[stage])
    version = versions[0].version
    run_id = versions[0].run_id

    run = client.get_run(run_id)
    params = run.data.params

    return model, {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "params": params
    }