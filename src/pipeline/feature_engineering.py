import pandas as pd
import joblib
import os
import gender_guesser.detector as gender


class FeatureEngineer:

    def __init__(self, df, mode, model_name=None):

        self.df = df.copy()
        self.mode = mode
        self.model_name = str(model_name).lower() if model_name else ""

        ENCODER_DIR = "./models/encoders"

        if mode == "regression":
            self.encoders = joblib.load(os.path.join(ENCODER_DIR, "flight_encoders.pkl"))
            self.expected_cols = joblib.load(os.path.join(ENCODER_DIR, "flight_columns.pkl"))
            self.scaler = joblib.load(os.path.join(ENCODER_DIR, "flight_scaler.pkl"))

        elif mode == "classification":
            self.encoders = joblib.load(os.path.join(ENCODER_DIR, "user_encoders.pkl"))
            self.expected_cols = joblib.load(os.path.join(ENCODER_DIR, "user_columns.pkl"))
            self.scaler = joblib.load(os.path.join(ENCODER_DIR, "user_scaler.pkl"))
            self.name_gender_map = joblib.load(os.path.join(ENCODER_DIR, "name_gender_map.pkl"))
            self.gender_detector = gender.Detector()

        else:
            self.encoders = {}
            self.expected_cols = []
            self.scaler = None

    def should_scale(self):

        model = (
            str(self.model_name)
            .lower()
            .strip()
            .replace(" ", "_")
        )

        print("\n==============================")
        print("MODEL RECEIVED:", model)
        print("==============================")

        regression_models = [
            "linearregression",
            "linear_regression",
            "linear_regression_base",
            "flightpricemodel"
        ]

        classification_models = [
            "logisticregression",
            "logistic_regression",
            "logistic_regression_base",
            "genderpredictionmodel"
        ]

        if self.mode == "regression":

            scaling = model in regression_models

            if scaling:
                print("SCALING APPLIED → REGRESSION")
            else:
                print("NO SCALING APPLIED → REGRESSION")

            return scaling

        if self.mode == "classification":

            scaling = model in classification_models

            if scaling:
                print("SCALING APPLIED → CLASSIFICATION")
            else:
                print("NO SCALING APPLIED → CLASSIFICATION")

            return scaling

        print("NO SCALING APPLIED")
        return False

  
    def transform(self):

        # CLEAN COLUMN NAMES
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )


        # REGRESSION PIPELINE

        if self.mode == "regression":

            date_cols = []

            for col in self.df.columns:
                if "date" in col:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

                    self.df[col + "_year"] = self.df[col].dt.year
                    self.df[col + "_month"] = self.df[col].dt.month
                    self.df[col + "_day"] = self.df[col].dt.day

                    date_cols.append(col)

            self.df.drop(columns=date_cols, inplace=True, errors='ignore')

            # Route feature
            if 'from' in self.df.columns and 'to' in self.df.columns:
                self.df['route'] = self.df['from'].astype(str) + "_" + self.df['to'].astype(str)
                self.df.drop(columns=['from', 'to'], inplace=True, errors='ignore')


        # CLASSIFICATION PIPELINE

        if self.mode == "classification":

            if 'name' in self.df.columns:
                self.df['first_name'] = self.df['name'].astype(str).str.split().str[0]

                def get_name_gender(name):
                    if pd.isna(name):
                        return "unknown"

                    name = str(name).split()[0]
                    g = self.gender_detector.get_gender(name)

                    if g == "male":
                        return "male"
                    elif g == "female":
                        return "female"
                    elif g == "mostly_male":
                        return "mostly_male"
                    elif g == "mostly_female":
                        return "mostly_female"
                    else:
                        return "unknown"

                self.df['name_gender'] = self.df['first_name'].apply(get_name_gender)
                self.df['name_gender'] = self.df['name_gender'].map(self.name_gender_map)

            if 'age' in self.df.columns:
                self.df['age_group'] = pd.cut(
                    self.df['age'],
                    bins=[0, 18, 30, 45, 60, 100],
                    labels=["Teen", "Young", "Adult", "Mid-Age", "Senior"]
                ).astype(str)


        # HANDLE MISSING

        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].fillna("unknown")
            else:
                self.df[col] = self.df[col].fillna(0)


        # ENCODING

        for col, encoder in self.encoders.items():

            if col in self.df.columns:

                self.df[col] = self.df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0]
                    if x in encoder.classes_
                    else -1
                )



        drop_cols = ['usercode', 'travelcode', 'name', 'code']
        drop_cols = [c for c in drop_cols if c in self.df.columns]
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')


        # COLUMN ALIGNMENT

        for col in self.expected_cols:
            if col not in self.df.columns:
                self.df[col] = 0

        self.df = self.df[self.expected_cols]


        # SCALING
        if self.scaler is not None and self.should_scale():
            self.df = pd.DataFrame(
                self.scaler.transform(self.df),
                columns=self.df.columns
            )

        return self.df