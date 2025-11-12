import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List

@dataclass
class CanonicalTrafficData:
    edge_df: pd.DataFrame
    topology_df: pd.DataFrame
    speed_scaler: StandardScaler
    train_run_ids: List[str] # run_id bây giờ là string
    val_run_ids: List[str]
    test_run_ids: List[str]

    @classmethod
    def from_parquet(cls, path: str) -> 'CanonicalTrafficData':
        df = pd.read_parquet(path)
        
        # Đổi tên cột cho nhất quán với logic cũ của chúng ta
        df.rename(columns={'node_a_id': 'source_node', 'node_b_id': 'destination_node'}, inplace=True)
        df['edge_id'] = df['source_node'] + '->' + df['destination_node']

        topology_df = df[['edge_id', 'source_node', 'destination_node']].drop_duplicates().reset_index(drop=True)
        
        # Chia dữ liệu theo run_id (bây giờ là string)
        all_run_ids = sorted(df['run_id'].unique())
        n_total = len(all_run_ids)
        n_train = int(n_total * 0.7) # 70/15/15 split
        n_val = int(n_total * 0.15)

        train_ids = all_run_ids[:n_train]
        val_ids = all_run_ids[n_train : n_train + n_val]
        test_ids = all_run_ids[n_train + n_val:]
        
        train_df = df[df['run_id'].isin(train_ids)]
        
        speed_scaler = StandardScaler()
        speed_scaler.fit(train_df[['speed_kmh']].values)
        
        return cls(
            edge_df=df,
            topology_df=topology_df,
            speed_scaler=speed_scaler,
            train_run_ids=train_ids,
            val_run_ids=val_ids,
            test_run_ids=test_ids,
        )