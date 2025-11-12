from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pandas as pd

DEFAULT_REQUIRED_COLUMNS = [
    "run_id", "timestamp", "node_a_id", "node_b_id", "speed_kmh"
]

@dataclass
class DatasetValidationResult:
    path: Path
    exists: bool = False
    is_valid: bool = True # Bắt đầu với giả định là hợp lệ
    rows: int = 0
    columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    null_columns: List[str] = field(default_factory=list)
    duplicate_rows: int = 0
    errors: List[str] = field(default_factory=list)

def validate_processed_dataset(path: Path, required_columns: List[str]) -> DatasetValidationResult:
    """
    Thực hiện kiểm tra toàn diện trên một file dataset Parquet.
    """
    if not path.exists():
        return DatasetValidationResult(
            path=path, exists=False, is_valid=False, errors=["File not found."]
        )
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return DatasetValidationResult(
            path=path, exists=True, is_valid=False, errors=[f"Failed to read Parquet file: {e}"]
        )

    result = DatasetValidationResult(path=path, exists=True, rows=len(df), columns=list(df.columns))

    # --- 1. Kiểm tra các cột bị thiếu ---
    result.missing_columns = [col for col in required_columns if col not in df.columns]
    if result.missing_columns:
        result.is_valid = False
        result.errors.append(f"Missing required columns: {result.missing_columns}")

    # --- 2. Kiểm tra các cột có giá trị Null/NaN ---
    # Chỉ kiểm tra trên các cột quan trọng để tránh báo cáo nhiễu
    cols_to_check_nulls = [col for col in DEFAULT_REQUIRED_COLUMNS if col in df.columns]
    null_info = df[cols_to_check_nulls].isnull().sum()
    result.null_columns = null_info[null_info > 0].index.tolist()
    if result.null_columns:
        result.is_valid = False
        # Tạo thông báo lỗi chi tiết hơn
        for col in result.null_columns:
            count = null_info[col]
            pct = 100 * count / len(df)
            result.errors.append(f"Column '{col}' contains {count} null values ({pct:.2f}%).")

    # --- 3. Kiểm tra các dòng bị trùng lặp ---
    # Một dòng được coi là trùng lặp nếu nó có cùng run_id, node_a_id, và node_b_id
    # Điều này không nên xảy ra trong dữ liệu đã xử lý
    key_cols = ['run_id', 'node_a_id', 'node_b_id']
    if all(col in df.columns for col in key_cols):
        duplicates = df.duplicated(subset=key_cols, keep=False)
        result.duplicate_rows = duplicates.sum()
        if result.duplicate_rows > 0:
            result.is_valid = False
            result.errors.append(
                f"Found {result.duplicate_rows} duplicate rows based on (run_id, node_a_id, node_b_id) key."
            )
            
    # --- 4. Kiểm tra các ràng buộc dữ liệu cơ bản (ví dụ) ---
    if 'speed_kmh' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['speed_kmh']):
            result.is_valid = False
            result.errors.append("Column 'speed_kmh' is not a numeric type.")
        elif df['speed_kmh'].min() < 0:
            result.is_valid = False
            result.errors.append("Found negative values in 'speed_kmh' column.")

    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        result.is_valid = False
        result.errors.append("Column 'timestamp' is not a datetime type.")

    return result

def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> bool:
    """
    Validates that there is no data leakage between temporal splits.
    """
    print("\n=== Validating No Data Leakage Across Splits ===")
    
    # Chuyển đổi timestamp nếu cần
    # Dùng .copy() để tránh SettingWithCopyWarning
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    train_max_ts = train_df['timestamp'].max()
    val_min_ts = val_df['timestamp'].min()
    val_max_ts = val_df['timestamp'].max()
    test_min_ts = test_df['timestamp'].min()
    
    checks_passed = True
    
    # 1. Kiểm tra sự chồng chéo về thời gian
    if not pd.isna(train_max_ts) and not pd.isna(val_min_ts) and train_max_ts >= val_min_ts:
        print(f"✗ LEAKAGE DETECTED: Training data extends into validation period!")
        print(f"  Train Max Timestamp: {train_max_ts}")
        print(f"  Validation Min Timestamp: {val_min_ts}")
        checks_passed = False
    else:
        print("✓ No temporal overlap between train and validation sets.")

    if not pd.isna(val_max_ts) and not pd.isna(test_min_ts) and val_max_ts >= test_min_ts:
        print(f"✗ LEAKAGE DETECTED: Validation data extends into test period!")
        print(f"  Validation Max Timestamp: {val_max_ts}")
        print(f"  Test Min Timestamp: {test_min_ts}")
        checks_passed = False
    else:
        print("✓ No temporal overlap between validation and test sets.")
        
    # 2. Kiểm tra sự chồng chéo run_id gốc
    if 'run_id' in train_df.columns and 'run_id' in val_df.columns and 'run_id' in test_df.columns:
        orig_train_runs = set(train_df[~train_df['run_id'].str.contains('aug_', na=False)]['run_id'].unique())
        orig_val_runs = set(val_df[~val_df['run_id'].str.contains('aug_', na=False)]['run_id'].unique())
        orig_test_runs = set(test_df[~test_df['run_id'].str.contains('aug_', na=False)]['run_id'].unique())
        
        # Kiểm tra chéo
        if not orig_train_runs.isdisjoint(orig_val_runs):
            print(f"✗ LEAKAGE DETECTED: Shared original run_ids between train and val!")
            checks_passed = False
        else:
            print("✓ No shared original run_ids between train and validation sets.")
            
        if not orig_val_runs.isdisjoint(orig_test_runs):
            print(f"✗ LEAKAGE DETECTED: Shared original run_ids between val and test!")
            checks_passed = False
        else:
            print("✓ No shared original run_ids between validation and test sets.")

    if checks_passed:
        print("\n✓ All leakage checks passed!")
    else:
        print("\n✗ DATA LEAKAGE DETECTED! Please review the data splitting and augmentation process.")
        
    return checks_passed