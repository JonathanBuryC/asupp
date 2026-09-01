
from datasets import Dataset, Features, Sequence, Value
import pandas as pd
    # ---------------------
    # DATA PREP
    # ---------------------
def load_and_prepare_data(train_path, val_path, test_path):
        """Load Excel datasets and return HF datasets (train/val/test)."""

        def filter_pos(df):
            return df[df["Annotation"] == 2]

        train_df, val_df, test_df = map(
            pd.read_excel, [train_path, val_path, test_path]
        )
        train_pos, val_pos, test_pos = map(filter_pos, [train_df, val_df, test_df])

        def build_pairs(df, epochs=1):
            q2a = df.groupby("Question")["Content"].apply(list).to_dict()
            pairs = []
            for _ in range(epochs):
                for q, answers in q2a.items():
                    for a in answers:
                        pairs.append({"texts": [q, a]})
            return pairs

        features = Features({"texts": Sequence(Value("string"))})

        train_dataset = Dataset.from_list(build_pairs(train_pos, 3), features=features)
        val_dataset = Dataset.from_list(build_pairs(val_pos, 1), features=features)
        test_dataset = Dataset.from_list(build_pairs(test_pos, 1), features=features)

        return train_dataset, val_dataset, test_dataset