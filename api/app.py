import logging
from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src.config.paths import MLFLOW_TRACKING_URI
from src.pipeline.main_pipeline import MainPipeline
from src.utils.encoder_loader import load_gender_target_encoder


# LOGGING

logging.basicConfig(level=logging.INFO)


# MLFLOW SETUP

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()


# FASTAPI APP

app = FastAPI(
    title="Voyage Analytics API",
    version="5.0"
)


# CLEAN OUTPUT

def clean_output(data):

    if isinstance(data, dict):

        return {
            k: clean_output(v)
            for k, v in data.items()
        }

    elif isinstance(data, list):

        return [
            clean_output(v)
            for v in data
        ]

    elif isinstance(data, float):

        if np.isnan(data) or np.isinf(data):

            return 0.0

    return data


# LOAD DATA LAZILY

@lru_cache(maxsize=1)
def get_flights_df():

    logging.info("Loading flights dataset...")

    return pd.read_csv(
        "./data/processed/flights_clean.csv"
    )


# LOAD ENCODER LAZILY

@lru_cache(maxsize=1)
def get_target_encoder():

    logging.info("Loading gender target encoder...")

    return load_gender_target_encoder()


# PIPELINE CACHE

@lru_cache(maxsize=10)
def get_pipeline(
    task,
    mode="auto",
    model_name=None
):

    logging.info(
        f"Loading pipeline: {task} | {mode} | {model_name}"
    )

    return MainPipeline(
        task=task,
        mode=mode,
        model_name=model_name
    )


# HOME

@app.get("/")
def home():

    return {
        "status": "API is running 🚀"
    }


# HEALTH CHECK

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# FETCH METRICS

def get_metrics(run_id):

    try:

        run = client.get_run(run_id)

        return run.data.metrics

    except Exception as e:

        logging.error(f"Metrics fetch failed: {e}")

        return {}


# PRICE PREDICTION

@app.post("/predict/price")
def predict_price(data: dict):

    try:

        mode = data.get("mode", "auto")

        model_name = data.get(
            "model_name",
            None
        )

        source = data.get("from")

        destination = data.get("to")

        flight_type = data.get("flighttype")

        df = get_flights_df()

        if df.empty:

            raise ValueError(
                "Flight dataset not loaded"
            )

        route_df = df[
            (df['from'] == source) &
            (df['to'] == destination) &
            (df['flighttype'] == flight_type)
        ].copy()

        if route_df.empty:

            raise ValueError(
                "No flights found for this route"
            )

        route_df = route_df.drop_duplicates(
            subset=['agency']
        )

        pipeline = get_pipeline(
            task="regression",
            mode=mode,
            model_name=model_name
        )

        results = []

        info = {}

        for _, row in route_df.iterrows():

            input_data = {

                "from": source,

                "to": destination,

                "flighttype": flight_type,

                "distance": row.get(
                    "distance",
                    500
                ),

                "time": row.get(
                    "time",
                    2
                )
            }

            log_pred, info = pipeline.run(
                input_data
            )

            price = float(
                np.expm1(log_pred)
            )

            if np.isnan(price) or np.isinf(price):

                price = 0.0

            results.append({

                "agency": row.get("agency"),

                "price": round(price, 2),

                "distance": row.get("distance"),

                "time": row.get("time")
            })

        results = sorted(
            results,
            key=lambda x: x["price"]
        )

        metrics = get_metrics(
            info.get("run_id")
        )

        response = {

            "route_info": {

                "source": source,

                "destination": destination,

                "flight_type": flight_type,

                "distance": float(
                    route_df['distance'].mean()
                ),

                "time": float(
                    route_df['time'].mean()
                )
            },

            "results": results,

            "model_info": {

                "mode": mode,

                "model_name": info.get(
                    "model_name"
                ),

                "version": info.get(
                    "version"
                ),

                "run_id": info.get(
                    "run_id"
                ),

                "algorithm": info.get(
                    "params",
                    {}
                ).get(
                    "model",
                    "unknown"
                ),

                "type": info.get(
                    "params",
                    {}
                ).get(
                    "type",
                    "unknown"
                ),
            },

            "metrics": {

                "MAE": metrics.get("MAE"),

                "RMSE": metrics.get("RMSE"),

                "R2": metrics.get("R2"),

                "CV": metrics.get("CV"),

                "score": metrics.get("score")
            },

            "status": "success"
        }

        return clean_output(response)

    except Exception as e:

        logging.error(f"Price prediction failed: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# GENDER PREDICTION

@app.post("/predict/gender")
def predict_gender(data: dict):

    try:

        mode = data.get("mode", "auto")

        model_name = data.get(
            "model_name",
            "GenderPredictionModel"
        )

        pipeline = get_pipeline(
            task="classification",
            mode=mode,
            model_name=model_name
        )

        pred, info = pipeline.run(data)

        encoder = get_target_encoder()

        if encoder:

            pred = encoder.inverse_transform(
                [int(pred)]
            )[0]

        metrics = get_metrics(
            info.get("run_id")
        )

        response = {

            "prediction": pred,

            "model_info": {

                "mode": mode,

                "model_name": info.get(
                    "model_name"
                ),

                "version": info.get(
                    "version"
                ),

                "run_id": info.get(
                    "run_id"
                ),

                "algorithm": info.get(
                    "params",
                    {}
                ).get(
                    "model",
                    "unknown"
                ),

                "type": info.get(
                    "params",
                    {}
                ).get(
                    "type",
                    "unknown"
                ),
            },

            "metrics": {

                "accuracy": metrics.get(
                    "accuracy"
                ),

                "precision": metrics.get(
                    "precision"
                ),

                "recall": metrics.get(
                    "recall"
                ),

                "f1": metrics.get("f1"),

                "score": metrics.get(
                    "score"
                )
            },

            "status": "success"
        }

        return clean_output(response)

    except Exception as e:

        logging.error(f"Gender prediction failed: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# RECOMMENDATION

@app.post("/recommend")
def recommend(data: dict):

    try:

        pipeline = get_pipeline(
            task="recommendation"
        )

        result, info = pipeline.run(data)

        response = {

            "recommendations": result,

            "type": info.get("rec_type"),

            "status": "success"
        }

        return clean_output(response)

    except Exception as e:

        logging.error(f"Recommendation failed: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# METADATA

@app.get("/metadata/locations")
def get_locations():

    try:

        df = get_flights_df()

        if df.empty:

            raise ValueError(
                "Flight dataset not loaded"
            )

        return {

            "sources": sorted(
                df['from']
                .dropna()
                .unique()
                .tolist()
            ),

            "destinations": sorted(
                df['to']
                .dropna()
                .unique()
                .tolist()
            ),

            "flight_types": sorted(
                df['flighttype']
                .dropna()
                .unique()
                .tolist()
            )
        }

    except Exception as e:

        logging.error(f"Metadata fetch failed: {e}")

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )