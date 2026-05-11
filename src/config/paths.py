import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
ENCODERS_DIR = os.path.join(MODELS_DIR, "encoders")

FLIGHT_ENCODER_PATH = os.path.join(ENCODERS_DIR, "flight_encoders.pkl")
USER_ENCODER_PATH = os.path.join(ENCODERS_DIR, "user_encoders.pkl")
GENDER_TARGET_ENCODER_PATH = os.path.join(ENCODERS_DIR, "gender_target_encoder.pkl")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

REGRESSION_MODEL_URI = "models:/FlightPriceModel/Production"
CLASSIFICATION_MODEL_URI = "models:/GenderPredictionModel/Production"