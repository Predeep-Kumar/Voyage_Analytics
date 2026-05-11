from src.pipeline.input_handler import InputHandler
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.prediction import Predictor

from src.utils.model_loader import load_auto_model_with_info, load_manual_model
from src.pipeline.recommender import Recommender
recommender_instance = None


def get_recommender():

    global recommender_instance

    if recommender_instance is None:

        recommender_instance = Recommender(
            data_path="./data/processed"
        )

    return recommender_instance

class MainPipeline:
    
    def __init__(self, task, mode="auto", model_name=None):
        self.task = task
        self.mode = mode
        self.model_name = model_name

        # HANDLE RECOMMENDER SEPARATELY
        if self.task == "recommendation":
            self.recommender = get_recommender()
            self.model = None
            self.model_info = {"task": "recommendation"}
            print("[PIPELINE] Recommender system initialized")
            return

        # ML MODELS (REG / CLF)
        if mode == "auto":
            self.model, self.model_info = load_auto_model_with_info(task)
            self.model_info["source"] = "mlflow_auto"

            print(f"[PIPELINE] AUTO MODE → Using MLflow Production model: "
                  f"{self.model_info['model_name']} (v{self.model_info['version']})")

        else:
            self.model, self.model_info = load_manual_model(task, model_name)

            if self.model_info.get("version") == "local_fallback":
                self.model_info["source"] = "local_fallback"
                print(f"[PIPELINE] MANUAL MODE → MLflow failed, using LOCAL model: {model_name}")
            else:
                self.model_info["source"] = "mlflow_manual"
                print(f"[PIPELINE] MANUAL MODE → Using MLflow run: "
                      f"{self.model_info['model_name']} (run_id={self.model_info['run_id']})")

    
    # MAIN RUN FUNCTION
    def run(self, data: dict):


        # RECOMMENDER LOGIC

        if self.task == "recommendation":

            print("[PIPELINE] Running Recommender System...")

            rec_type = data.get("rec_type", "all")

            if rec_type == "flights":
                result = self.recommender.recommend_flights(
                    user_id=data.get("user_id"),
                    source=data.get("source"),
                    destination=data.get("destination"),
                    flight_type=data.get("flight_type"),
                    max_price=data.get("max_price"),
                    top_n=data.get("top_n", 5)
                )

            elif rec_type == "hotels":
                result = self.recommender.recommend_hotels(
                    user_id=data.get("user_id"),
                    place=data.get("place"),
                    max_price=data.get("max_price"),
                    top_n=data.get("top_n", 5)
                )

            elif rec_type == "package":
                result = self.recommender.recommend_package(
                    user_id=data.get("user_id"),
                    source=data.get("source"),
                    destination=data.get("destination"),
                    place=data.get("place"),
                    max_price=data.get("max_price"),
                    top_n=data.get("top_n", 5)
                )

            else:
                result = self.recommender.recommend_all(
                    user_id=data.get("user_id"),
                    source=data.get("source"),
                    destination=data.get("destination"),
                    place=data.get("place"),
                    max_price=data.get("max_price"),
                    top_n=data.get("top_n", 5)
                )

            print(f"[PIPELINE] Recommendation completed → {rec_type}")

            return result, {
                "task": "recommendation",
                "rec_type": rec_type
            }


        # ML PIPELINE (REG / CLF)

        df = InputHandler(data).get_dataframe()
        print("[PIPELINE] Input processed")

        current_model_name = (
            self.model_name
            if self.model_name
            else self.model_info.get("model_name", "")
        )

        df = FeatureEngineer(
            df=df,
            mode=self.task,
            model_name=current_model_name
        ).transform()
        print("[PIPELINE] Feature engineering completed")

        pred = Predictor(self.model).predict(df)
        print("[PIPELINE] Prediction completed")

        return pred[0], self.model_info