import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from .canonical_data import CanonicalTrafficData

class GraphWaveNetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class GraphWaveNetAdapter:
    def _aggregate_edges_to_nodes(self, edge_df: pd.DataFrame) -> pd.DataFrame:
        node_df = edge_df.groupby(['timestamp', 'destination_node'])[['speed_kmh']].mean().reset_index()
        node_df.rename(columns={'destination_node': 'node_id'}, inplace=True)
        return node_df

    def __call__(self, canonical: CanonicalTrafficData, split: str, seq_len: int, pred_len: int, batch_size: int) -> DataLoader:
        if split == 'train':
            run_ids = canonical.train_run_ids
        elif split == 'val':
            run_ids = canonical.val_run_ids
        else:
            run_ids = canonical.test_run_ids
            
        # 1. Lọc DataFrame theo đúng các run_id của split
        split_df = canonical.edge_df[canonical.edge_df['run_id'].isin(run_ids)].copy()
        
        # Sắp xếp lại để đảm bảo tính liên tục của thời gian
        split_df.sort_values('timestamp', inplace=True)

        # 2. Tổng hợp sang node-centric
        node_df = self._aggregate_edges_to_nodes(split_df)

        # 3. Pivot để tạo data cube
        all_node_ids = sorted(list(set(canonical.topology_df['source_node']) | set(canonical.topology_df['destination_node'])))
        pivot_df = node_df.pivot(index='timestamp', columns='node_id', values='speed_kmh')
        
        # Quan trọng: Reindex theo một dải thời gian đầy đủ để xử lý các timestamp bị thiếu
        full_time_range = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='15min') # Giả sử 15 phút
        pivot_df = pivot_df.reindex(full_time_range)
        pivot_df = pivot_df.reindex(columns=all_node_ids)

        # Xử lý NaN
        pivot_df.interpolate(method='time', limit_direction='both', inplace=True)
        if pivot_df.isnull().values.any():
            # Kiểm tra nếu scaler đã được fit và có thuộc tính mean_
            if hasattr(canonical.speed_scaler, 'mean_') and canonical.speed_scaler.mean_ is not None:
                mean_val = canonical.speed_scaler.mean_[0]
                pivot_df.fillna(mean_val, inplace=True)
            else:
                # Fallback: nếu scaler chưa được fit, điền bằng 0 và cảnh báo
                print("Warning: Scaler has not been fitted. Filling remaining NaNs with 0.")
                pivot_df.fillna(0, inplace=True)
        
        data_cube = pivot_df.values.astype(np.float32)
        
        # 4. Tạo sequences trên toàn bộ data_cube của split
        X_list, y_list = [], []
        if len(data_cube) >= seq_len + pred_len:
            for i in range(len(data_cube) - seq_len - pred_len + 1):
                X_list.append(data_cube[i : i + seq_len, :])
                y_list.append(data_cube[i + seq_len : i + seq_len + pred_len, :])
        
        if not X_list:
            X, y = np.empty((0, seq_len, len(all_node_ids))), np.empty((0, pred_len, len(all_node_ids)))
        else:
            X, y = np.array(X_list), np.array(y_list)

        if X.shape[0] == 0:
            empty_dataset = GraphWaveNetDataset(np.empty((0, seq_len, len(all_node_ids), 1)), np.empty((0, pred_len, len(all_node_ids))))
            return DataLoader(empty_dataset, batch_size=batch_size)

        # 5. Chuẩn hóa và Reshape
        X_scaled = canonical.speed_scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = canonical.speed_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        
        X_scaled = np.expand_dims(X_scaled, axis=-1)
        X_final = X_scaled # (batch, time, nodes, features)
        y_final = y_scaled # (batch, time, nodes)
        
        dataset = GraphWaveNetDataset(X_final, y_final)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))