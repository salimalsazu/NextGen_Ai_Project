from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class PopularityRecommender:
    top_items: List[int]

    def predict(self, user_id: int, k: int = 10) -> List[int]:
        return self.top_items[:k]

    def batch_predict(self, df: pd.DataFrame) -> List[List[int]]:
        out = []
        for _, row in df.iterrows():
            k = int(row["k"]) if "k" in df.columns else 10
            out.append(self.predict(int(row["user_id"]), k))
        return out
