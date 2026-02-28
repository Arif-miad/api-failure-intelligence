import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class CategoricalEncoder:
    def __init__(self, categorical_cols: list[str]):
        self.categorical_cols = categorical_cols
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df: pd.DataFrame):
        self.encoder.fit(df[self.categorical_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.categorical_cols] = self.encoder.transform(out[self.categorical_cols])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)