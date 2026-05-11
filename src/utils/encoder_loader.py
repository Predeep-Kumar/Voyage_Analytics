import joblib
from src.config.paths import (
    FLIGHT_ENCODER_PATH,
    USER_ENCODER_PATH,
    GENDER_TARGET_ENCODER_PATH
)

def load_flight_encoders():
    return joblib.load(FLIGHT_ENCODER_PATH)

def load_user_encoders():
    return joblib.load(USER_ENCODER_PATH)

def load_gender_target_encoder():
    return joblib.load(GENDER_TARGET_ENCODER_PATH)