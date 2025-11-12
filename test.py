import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
import json
from datetime import datetime
from pathlib import Path
import sys

# Thêm đường dẫn dự án để import
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models.graphwavenet import GraphWaveNet
from utils.canonical_data import CanonicalTrafficData
# Import hàm load_run_data từ scripts.combine_runs
from scripts.combine_runs import load_run_data

# --- Cấu hình ---
TRAIN_VAL_DATA_PATH = 'data/processed/train_val_extreme_augmented.parquet'
HOLDOUT_TEST_DIR = 'data/runs_holdout_test' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
SEQ_LEN = 24
PRED_LEN = 12
MODEL_PATH = 'best_graphwavenet_model.pth'

class TestDataset(Dataset):
    """Lớp Dataset đơn giản cho dữ liệu test."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def create_test_sequences(node_df: pd.DataFrame, all_node_ids: list, seq_len: int, pred_len: int):
    """Tạo cửa sổ trượt từ dữ liệu test."""
    # Sắp xếp theo thời gian để đảm bảo tính liên tục
    node_df.sort_values('timestamp', inplace=True)
    
    pivot_df = node_df.pivot(index='timestamp', columns='node_id', values='speed_kmh')
    
    # Reindex để đảm bảo có đủ các timestamp và node
    full_time_range = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='15min')
    pivot_df = pivot_df.reindex(full_time_range)
    pivot_df = pivot_df.reindex(columns=all_node_ids)

    # Nội suy để lấp đầy các khoảng trống
    pivot_df.interpolate(method='time', limit_direction='both', inplace=True)
    if pivot_df.isnull().values.any():
        pivot_df.fillna(pivot_df.mean().mean(), inplace=True)
        
    data_cube = pivot_df.values.astype(np.float32)

    X_list, y_list = [], []
    if len(data_cube) >= seq_len + pred_len:
        for i in range(len(data_cube) - seq_len - pred_len + 1):
            X_list.append(data_cube[i : i + seq_len, :])
            y_list.append(data_cube[i + seq_len : i + seq_len + pred_len, :])
            
    return np.array(X_list), np.array(y_list)

def _resolve_relative(path: Path) -> Path:
    """Resolves a path relative to the project root."""
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

def calculate_metrics(preds, labels):
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)
    threshold = 1.0
    valid_indices = labels > threshold
    if np.sum(valid_indices) > 0:
        mape = np.mean(np.abs((labels[valid_indices] - preds[valid_indices]) / labels[valid_indices])) * 100
    else:
        mape = float('nan')
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def main():
    print("--- Bắt đầu quá trình đánh giá CUỐI CÙNG trên tập Test 'còn zin' ---")

    # --- 1. Tải Scaler và Topology ---
    print("Đang tải Scaler và Topology từ dữ liệu huấn luyện...")
    train_val_canonical = CanonicalTrafficData.from_parquet(TRAIN_VAL_DATA_PATH)
    scaler = train_val_canonical.speed_scaler
    topology_df = train_val_canonical.topology_df
    all_node_ids = sorted(list(set(topology_df['source_node']) | set(topology_df['destination_node'])))
    num_nodes = len(all_node_ids)

    # --- 2. Xử lý dữ liệu Test "còn zin" ---
    print(f"Đang xử lý dữ liệu test từ: {HOLDOUT_TEST_DIR}")
    test_run_dirs = sorted(glob.glob(f"{HOLDOUT_TEST_DIR}/run_*"))
    all_test_records = []
    for run_dir in test_run_dirs:
        # Dùng Path() để đảm bảo kiểu dữ liệu đúng
        records = load_run_data(Path(run_dir))
        if records: all_test_records.extend(records)
    
    if not all_test_records:
        print("Không tìm thấy dữ liệu trong thư mục test. Kết thúc.")
        return

    test_df_raw = pd.DataFrame(all_test_records)
    test_df_raw.rename(columns={'node_a_id': 'source_node', 'node_b_id': 'destination_node'}, inplace=True)
    
    # Tổng hợp sang node-centric
    node_df_test = test_df_raw.groupby(['timestamp', 'destination_node'])[['speed_kmh']].mean().reset_index()
    node_df_test.rename(columns={'destination_node': 'node_id'}, inplace=True)

    # Tạo chuỗi thời gian cho dữ liệu test
    X_test, y_test = create_test_sequences(node_df_test, all_node_ids, SEQ_LEN, PRED_LEN)

    if X_test.shape[0] == 0:
        print("Tập test không đủ dài để tạo sample. Kết thúc.")
        return

    # Chuẩn hóa dữ liệu test BẰNG SCALER CỦA TẬP TRAIN
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)
    
    test_dataset = TestDataset(X_test_scaled, y_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 3. TẢI MODEL VÀ DỰ ĐOÁN (PHẦN BỊ THIẾU) ---
    print(f"Đang tải model từ: {MODEL_PATH}")
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    node_to_idx = {node_id: i for i, node_id in enumerate(all_node_ids)}
    for _, row in topology_df.iterrows():
        u, v = row['source_node'], row['destination_node']
        if u in node_to_idx and v in node_to_idx:
            adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
            adj_matrix[node_to_idx[v], node_to_idx[u]] = 1
    supports = [torch.from_numpy(adj_matrix).to(DEVICE)]
    
    model = GraphWaveNet(num_nodes=num_nodes, in_dim=1, out_dim=PRED_LEN, supports=supports).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Tải model thành công.")

    print("Đang thực hiện dự đoán trên tập test...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch_scaled = x_batch.to(DEVICE), y_batch.to(DEVICE)
            output_scaled = model(x_batch)
            
            y_batch_permuted_scaled = y_batch_scaled.permute(0, 2, 1)

            pred_unscaled = scaler.inverse_transform(output_scaled.detach().cpu().numpy().reshape(-1, 1)).reshape(output_scaled.shape)
            label_unscaled = scaler.inverse_transform(y_batch_permuted_scaled.detach().cpu().numpy().reshape(-1, 1)).reshape(y_batch_permuted_scaled.shape)
            
            all_preds.append(pred_unscaled)
            all_labels.append(label_unscaled)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print("Dự đoán hoàn tất.")

    # --- 4. In kết quả ---
    print("\n--- Kết quả đánh giá trên tập Test ---")
    
    # Tính toán metrics cho từng bước dự báo (horizon)
    for i in range(PRED_LEN):
        # Chúng ta đang dự báo avg_speed, nên chỉ lấy feature đầu tiên
        # (Lưu ý: y_batch đã được xử lý trong DataLoader để chỉ chứa 1 feature)
        preds_horizon = all_preds[:, :, i]
        labels_horizon = all_labels[:, :, i]
        
        metrics = calculate_metrics(preds_horizon, labels_horizon)
        print(f"Horizon {i+1} (Dự đoán cho { (i+1)*15 } phút sau):")
        print(f"  MAE: {metrics['MAE']:.4f} km/h, RMSE: {metrics['RMSE']:.4f} km/h, MAPE: {metrics['MAPE']:.2f}%")

    # Tính toán metrics tổng thể trên tất cả các horizon
    print("\nMetrics tổng thể trên tập Test:")
    overall_metrics = calculate_metrics(all_preds.flatten(), all_labels.flatten())
    print(f"  MAE: {overall_metrics['MAE']:.4f} km/h, RMSE: {overall_metrics['RMSE']:.4f} km/h, R2: {overall_metrics['R2']:.4f}, MAPE: {overall_metrics['MAPE']:.2f}%")


if __name__ == '__main__':
    main()