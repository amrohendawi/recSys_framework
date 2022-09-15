import os
from datetime import datetime
import numpy as np
import pandas as pd
from classes.BaseDataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, dataset_name):
        super(Dataset, self).__init__(dataset_name)
        self.dataset_name = dataset_name

        self.inter_file = os.path.join("datasets", dataset_name, "transactions_train.csv")
        self.item_file = os.path.join("datasets", dataset_name, "articles.csv")
        self.user_file = os.path.join("datasets", dataset_name, "customers.csv")

        self.sep = ","

        # selected feature fields
        self.inter_fields = {
            0: "t_dat:float",
            1: "customer_id:token",
            2: "article_id:token",
        }

        self.item_fields = {
            0: "article_id:token",
        }

        self.user_fields = {
            0: "customer_id:token",
        }

    def load_inter_data(self):
        df = pd.read_csv(self.inter_file,
                         dtype={"t_dat": "object", "customer_id": "object", "article_id": "object", "price": float,
                                "sales_channel_id": int}
                         )
        # approx. 1 month + 2 weeks
        df = df[-len(df) * 3 // 48:].reset_index(drop=True)
        # Further downsampling to avoid OOM
        uus = df["customer_id"].unique()
        sampled_users = np.random.choice(uus, len(uus) // 6)
        df = df.query('customer_id in @sampled_users')
        df['t_dat'] = df['t_dat'].apply(lambda x: datetime.timestamp(datetime.strptime(x, "%Y-%m-%d")))
        return df

    def load_item_data(self):
        return pd.read_csv(self.item_file, delimiter=self.sep, engine="python")

    def load_user_data(self):
        return pd.read_csv(self.user_file, delimiter=self.sep, engine="python")
