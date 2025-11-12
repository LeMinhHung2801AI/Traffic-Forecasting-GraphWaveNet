import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.graphwavenet import GraphWaveNet
from utils.canonical_data import CanonicalTrafficData
from utils.adapters import GraphWaveNetAdapter

# --- Cấu hình ---
CANONICAL_DATA_PATH = 'data/processed/train_val_extreme_augmented.parquet' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
SEQ_LEN = 24
PRED_LEN = 12

def plot_and_save_loss_curve(train_losses, val_losses, filename='loss_curve.png'):
    # ... (Hàm này đã đúng, giữ nguyên) ...
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (L1Loss)', fontsize=12)
    plt.legend()
    plt.grid(True)
    if val_losses: # Thêm kiểm tra để không bị lỗi nếu val_losses rỗng
        best_epoch = np.argmin(val_losses) + 1
        plt.axvline(x=float(best_epoch), color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.legend()
    plt.savefig(filename, dpi=300)
    print(f"Biểu đồ loss đã được lưu vào file: {filename}")

def main():
    # --- Cấu hình cho Early Stopping ---
    # Tăng max epochs
    EPOCHS = 100 
    # Số epochs kiên nhẫn chờ trước khi dừng nếu val_loss không cải thiện
    EARLY_STOPPING_PATIENCE = 10
    
    # --- 1. Tải dữ liệu ---
    print("Đang tải dữ liệu Canonical...")
    canonical_data = CanonicalTrafficData.from_parquet(CANONICAL_DATA_PATH)

    # --- 2. Tạo DataLoaders ---
    print("Đang tạo DataLoaders thông qua Adapter...")
    adapter = GraphWaveNetAdapter()
    train_loader = adapter(canonical_data, 'train', SEQ_LEN, PRED_LEN, BATCH_SIZE)
    val_loader = adapter(canonical_data, 'val', SEQ_LEN, PRED_LEN, BATCH_SIZE)

    # --- 3. Khởi tạo model ---
    # ... (Toàn bộ phần khởi tạo model, adj_matrix, optimizer, criterion giữ nguyên) ...
    num_nodes = len(set(canonical_data.topology_df['source_node']) | set(canonical_data.topology_df['destination_node']))
    node_to_idx = {node_id: i for i, node_id in enumerate(sorted(list(set(canonical_data.topology_df['source_node']) | set(canonical_data.topology_df['destination_node']))))}
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in canonical_data.topology_df.iterrows():
        u, v = row['source_node'], row['destination_node']
        if u in node_to_idx and v in node_to_idx:
            adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
            adj_matrix[node_to_idx[v], node_to_idx[u]] = 1
    supports = [torch.from_numpy(adj_matrix).to(DEVICE)]
    
    model = GraphWaveNet(
        num_nodes=num_nodes,
        in_dim=1,
        out_dim=PRED_LEN,
        supports=supports
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in GraphWaveNet: {num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    # --- Vòng lặp huấn luyện với Early Stopping ---
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Bộ đếm số epoch không cải thiện
    
    history_train_loss = []
    history_val_loss = []
    
    print(f"Bắt đầu huấn luyện cho tối đa {EPOCHS} epochs với Early Stopping (patience={EARLY_STOPPING_PATIENCE})...")
    for epoch in range(1, EPOCHS + 1):
        # ... (Vòng lặp training và validation bên trong giữ nguyên) ...
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            # ... (code training)
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(x_batch)
            y_batch_permuted = y_batch.permute(0, 2, 1)
            loss = criterion(output, y_batch_permuted)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                # ... (code validation)
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(x_batch)
                y_batch_permuted = y_batch.permute(0, 2, 1)
                loss = criterion(output, y_batch_permuted)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- LOGIC EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 # Reset bộ đếm
            torch.save(model.state_dict(), 'best_graphwavenet_model.pth')
            print(f"  -> Val Loss cải thiện. Đã lưu model tốt nhất.")
        else:
            epochs_no_improve += 1
            print(f"  -> Val Loss không cải thiện. Kiên nhẫn: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEARLY STOPPING! Val Loss không cải thiện trong {EARLY_STOPPING_PATIENCE} epochs.")
            print(f"Model tốt nhất đã được lưu ở epoch {epoch - EARLY_STOPPING_PATIENCE} với Val Loss = {best_val_loss:.4f}")
            break # Thoát khỏi vòng lặp huấn luyện
            
    plot_and_save_loss_curve(history_train_loss, history_val_loss)

if __name__ == '__main__':
    main()