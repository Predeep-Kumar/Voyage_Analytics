import pandas as pd

class InputHandler:
    
    def __init__(self, data: dict):
        self.data = data
    
    def get_dataframe(self):
        return pd.DataFrame([self.data])