import pandas as pd
import numpy as np

class Predictor:
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        X = df.values
        return self.model.predict(X)

    def predict_proba(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        X = df.values
        return self.model.predict_proba(X)