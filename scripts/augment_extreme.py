"""
EXTREME Data Augmentation - Going Beyond
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
import json

# Thêm đường dẫn dự án để import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.validation.dataset_validation import (
    DEFAULT_REQUIRED_COLUMNS,
    validate_processed_dataset,
    validate_no_leakage,
)

def _resolve_path(path: Path) -> Path:
    """Resolves a path relative to the project root."""
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

np.random.seed(42)

class ExtremeAugmentor:
    """
    Extreme augmentation for maximum data generation
    """
    def __init__(self, df_original):
        self.df_orig = df_original.copy()
        
        # Đảm bảo tên cột nhất quán
        if 'source_node' in self.df_orig.columns:
            self.df_orig.rename(columns={'source_node': 'node_a_id', 'destination_node': 'node_b_id'}, inplace=True)
            
        self.df_orig['timestamp'] = pd.to_datetime(self.df_orig['timestamp'])
        self._learn_patterns()
    
    def _learn_patterns(self):
        """Learn patterns from data"""
        print("Learning advanced patterns...")
        
        # Hourly patterns
        self.df_orig['hour'] = self.df_orig['timestamp'].dt.hour
        self.df_orig['minute'] = self.df_orig['timestamp'].dt.minute
        
        # Edge-specific hourly profiles
        self.edge_hourly_profiles = {}
        for (node_a, node_b), group in self.df_orig.groupby(['node_a_id', 'node_b_id']):
            hourly_profile = group.groupby('hour')['speed_kmh'].agg(['mean', 'std', 'count'])
            self.edge_hourly_profiles[(node_a, node_b)] = hourly_profile.to_dict()
        
        # Global patterns
        self.global_hourly = self.df_orig.groupby('hour')['speed_kmh'].agg(['mean', 'std'])
        
        # Weather ranges
        self.weather_ranges = {
            'temperature_c': (self.df_orig['temperature_c'].min(), self.df_orig['temperature_c'].max()),
            'wind_speed_kmh': (self.df_orig['wind_speed_kmh'].min(), self.df_orig['wind_speed_kmh'].max()),
            'precipitation_mm': (self.df_orig['precipitation_mm'].min(), self.df_orig['precipitation_mm'].max())
        }
        
        print(f"  Learned {len(self.edge_hourly_profiles)} edge-specific profiles")
    
    def augment_hourly_interpolation(self, num_interpolations=2):
        print(f"\nMethod 1: Hourly Interpolation (x{num_interpolations} between each pair)")
        runs = sorted(self.df_orig['run_id'].unique())
        run_groups = [self.df_orig[self.df_orig['run_id'] == r] for r in runs]
        augmented_dfs = []
        for i in range(len(run_groups) - 1):
            run1, run2 = run_groups[i], run_groups[i + 1]
            t1, t2 = run1.iloc[0]['timestamp'], run2.iloc[0]['timestamp']
            time_diff = (t2 - t1).total_seconds()
            for interp_idx in range(1, num_interpolations + 1):
                alpha = interp_idx / (num_interpolations + 1)
                t_interp = t1 + timedelta(seconds=time_diff * alpha)
                df_interp = []
                for (node_a, node_b), edge1 in run1.groupby(['node_a_id', 'node_b_id']):
                    edge2_match = run2[(run2['node_a_id'] == node_a) & (run2['node_b_id'] == node_b)]
                    if len(edge2_match) == 0: continue
                    edge2, edge1_row = edge2_match.iloc[0], edge1.iloc[0]
                    speed_interp = edge1_row['speed_kmh'] * (1 - alpha) + edge2['speed_kmh'] * alpha + np.random.normal(0, 0.5)
                    temp_interp = edge1_row.get('temperature_c', 25) * (1 - alpha) + edge2.get('temperature_c', 25) * alpha
                    wind_interp = edge1_row.get('wind_speed_kmh', 5) * (1 - alpha) + edge2.get('wind_speed_kmh', 5) * alpha
                    precip_interp = edge1_row.get('precipitation_mm', 0) * (1 - alpha) + edge2.get('precipitation_mm', 0) * alpha
                    interp_row = edge1_row.copy()
                    interp_row.update({'timestamp': t_interp, 'speed_kmh': speed_interp, 'temperature_c': temp_interp, 'wind_speed_kmh': wind_interp, 'precipitation_mm': precip_interp, 'run_id': f"aug_interp_{i}_{interp_idx}_{t_interp.strftime('%Y%m%d_%H%M%S')}"})
                    df_interp.append(interp_row)
                if df_interp: augmented_dfs.append(pd.DataFrame(df_interp))
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} interpolated runs\n  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_synthetic_weather(self, num_weather_scenarios=10):
        """
        Method 2: Create variations with different weather conditions
        """
        print(f"\nMethod 2: Synthetic Weather Variations (x{num_weather_scenarios})")
        
        augmented_dfs = []
        
        # Weather scenarios
        weather_scenarios = [
            {'name': 'hot_dry', 'temp_mult': 1.15, 'wind_mult': 0.8, 'precip_mult': 0.0},
            {'name': 'cool_windy', 'temp_mult': 0.85, 'wind_mult': 1.5, 'precip_mult': 0.0},
            {'name': 'light_rain', 'temp_mult': 0.95, 'wind_mult': 1.1, 'precip_mult': 2.0},
            {'name': 'heavy_rain', 'temp_mult': 0.90, 'wind_mult': 1.3, 'precip_mult': 5.0},
            {'name': 'extreme_heat', 'temp_mult': 1.20, 'wind_mult': 0.7, 'precip_mult': 0.0},
            {'name': 'moderate', 'temp_mult': 1.0, 'wind_mult': 1.0, 'precip_mult': 0.5},
            {'name': 'cold_front', 'temp_mult': 0.80, 'wind_mult': 1.6, 'precip_mult': 1.5},
            {'name': 'humid_calm', 'temp_mult': 1.05, 'wind_mult': 0.6, 'precip_mult': 0.2},
            {'name': 'storm', 'temp_mult': 0.88, 'wind_mult': 1.8, 'precip_mult': 8.0},
            {'name': 'clear_perfect', 'temp_mult': 0.98, 'wind_mult': 0.9, 'precip_mult': 0.0}
        ]
        
        for scenario in weather_scenarios[:num_weather_scenarios]:
            df_weather = self.df_orig.copy()
            
            # Modify weather
            temp_base = df_weather['temperature_c'].mean()
            df_weather['temperature_c'] = temp_base + (df_weather['temperature_c'] - temp_base) * scenario['temp_mult']
            df_weather['temperature_c'] = df_weather['temperature_c'].clip(*self.weather_ranges['temperature_c'])
            
            df_weather['wind_speed_kmh'] *= scenario['wind_mult']
            df_weather['wind_speed_kmh'] = df_weather['wind_speed_kmh'].clip(0, self.weather_ranges['wind_speed_kmh'][1])
            
            df_weather['precipitation_mm'] *= scenario['precip_mult']
            df_weather['precipitation_mm'] = df_weather['precipitation_mm'].clip(0, 10.0)
            
            # Adjust speed based on weather impact
            # Rain reduces speed
            rain_factor = 1 - (df_weather['precipitation_mm'] / 10.0) * 0.15  # Max 15% reduction
            df_weather['speed_kmh'] *= rain_factor
            
            # High temp reduces speed slightly
            temp_deviation = (df_weather['temperature_c'] - 27.0) / 10.0  # 27°C baseline
            temp_factor = 1 - temp_deviation * 0.05  # Max 5% impact per 10°C
            df_weather['speed_kmh'] *= temp_factor
            
            # Strong wind reduces speed
            wind_factor = 1 - (df_weather['wind_speed_kmh'] - 5) / 50  # 5 km/h baseline
            wind_factor = wind_factor.clip(0.85, 1.0)
            df_weather['speed_kmh'] *= wind_factor
            
            # Clip to valid range
            df_weather['speed_kmh'] = df_weather['speed_kmh'].clip(3.0, 55.0)
            
            # New run IDs
            df_weather['run_id'] = f"aug_weather_{scenario['name']}_" + df_weather['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_weather)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} weather-augmented runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_multi_scenarios(self, num_scenarios=10):
        """
        Method 3: Extended pattern variations (10 instead of 5)
        """
        print(f"\nMethod 3: Multi-Scenario Variations (x{num_scenarios})")
        
        scenarios = [
            'rush_hour_heavy', 'rush_hour_light', 'weekend_traffic',
            'holiday_pattern', 'night_shift', 'early_morning',
            'midday_peak', 'accident_scenario', 'construction_zone',
            'special_event'
        ]
        
        augmented_dfs = []
        
        for idx, scenario_name in enumerate(scenarios[:num_scenarios]):
            df_var = self.df_orig.copy()
            df_var['hour'] = df_var['timestamp'].dt.hour
            
            if scenario_name == 'rush_hour_heavy':
                mask = df_var['hour'].isin([7, 8, 9, 17, 18, 19])
                df_var.loc[mask, 'speed_kmh'] *= 0.65
            elif scenario_name == 'rush_hour_light':
                mask = df_var['hour'].isin([7, 8, 9, 17, 18, 19])
                df_var.loc[mask, 'speed_kmh'] *= 0.85
            elif scenario_name == 'weekend_traffic':
                df_var['speed_kmh'] *= 1.15
            elif scenario_name == 'holiday_pattern':
                df_var['speed_kmh'] *= 1.25
            elif scenario_name == 'night_shift':
                mask = df_var['hour'].isin([0, 1, 2, 3, 4, 5])
                df_var.loc[mask, 'speed_kmh'] *= 1.20
            elif scenario_name == 'early_morning':
                mask = df_var['hour'].isin([5, 6, 7])
                df_var.loc[mask, 'speed_kmh'] *= 0.90
            elif scenario_name == 'midday_peak':
                mask = df_var['hour'].isin([11, 12, 13, 14])
                df_var.loc[mask, 'speed_kmh'] *= 0.80
            elif scenario_name == 'accident_scenario':
                # Random edges affected
                num_affected = len(df_var) // 20
                affected_idx = np.random.choice(len(df_var), num_affected, replace=False)
                df_var.loc[df_var.index[affected_idx], 'speed_kmh'] *= 0.50
            elif scenario_name == 'construction_zone':
                # Some edges permanently slower
                num_affected = len(df_var) // 15
                affected_idx = np.random.choice(len(df_var), num_affected, replace=False)
                df_var.loc[df_var.index[affected_idx], 'speed_kmh'] *= 0.70
            else:  # special_event
                mask = df_var['hour'].isin([18, 19, 20, 21])
                df_var.loc[mask, 'speed_kmh'] *= 0.60
            
            df_var['speed_kmh'] = df_var['speed_kmh'].clip(3.0, 55.0)
            df_var['run_id'] = f"aug_scenario_{scenario_name}_" + df_var['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_var)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} scenario runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_time_perturbation(self, num_perturbations=3, max_shift_minutes=30):
        """
        Method 4: Small time shifts to create variations
        """
        print(f"\nMethod 4: Time Perturbation (x{num_perturbations}, ±{max_shift_minutes}min)")
        
        augmented_dfs = []
        
        for pert_idx in range(num_perturbations):
            df_pert = self.df_orig.copy()
            
            # Random time shift
            shift_minutes = np.random.uniform(-max_shift_minutes, max_shift_minutes)
            df_pert['timestamp'] = df_pert['timestamp'] + timedelta(minutes=shift_minutes)
            
            # Adjust speed based on new hour
            df_pert['hour'] = df_pert['timestamp'].dt.hour
            
            # Apply hourly pattern adjustment
            for hour in df_pert['hour'].unique():
                if hour in self.global_hourly.index:
                    mask = df_pert['hour'] == hour
                    hour_mean = self.global_hourly.loc[hour, 'mean']
                    global_mean = self.global_hourly['mean'].mean()
                    
                    adjustment = hour_mean / global_mean
                    df_pert.loc[mask, 'speed_kmh'] *= adjustment
            
            df_pert['speed_kmh'] = df_pert['speed_kmh'].clip(3.0, 55.0)
            df_pert['run_id'] = f"aug_timeshift_{pert_idx}_{shift_minutes:.0f}min_" + df_pert['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_pert)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} time-perturbed runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate EXTREME augmented training datasets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/all_runs_augmented.parquet"),
        help="Input augmented parquet to enhance (default: data/processed/all_runs_augmented.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/all_runs_extreme_augmented.parquet"),
        help="Destination parquet for EXTREME augmented data (default: data/processed/all_runs_extreme_augmented.parquet)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the final augmented parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args() # Sử dụng hàm parse_args bạn đã có
    
    # --- TẢI CONFIG ---
    config_path = PROJECT_ROOT / 'config' / 'augmentation_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    ext_config = config['extreme']
    np.random.seed(config['seed'])
    
    print("=" * 70)
    print("EXTREME DATA AUGMENTATION (SAFE FROM DATA LEAKAGE)")
    print("=" * 70)
    
    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    
    # Đọc dữ liệu đã tăng cường "advanced" (chỉ chứa train/val)
    df_train_val_augmented = pd.read_parquet(input_path)
    print(f"\nBase augmented data loaded from '{input_path}': {len(df_train_val_augmented)} records, {df_train_val_augmented['run_id'].nunique()} runs")

    # --- BƯỚC 1: TÁCH LẠI TẬP TRAIN ĐỂ TĂNG CƯỜNG "EXTREME" ---
    # Lấy ra tất cả các run_id gốc (không có 'aug_')
    orig_run_ids = sorted([rid for rid in df_train_val_augmented[~df_train_val_augmented['run_id'].str.contains('aug_', na=False)]['run_id'].unique()])

    # Dựa trên các run_id gốc này, chia lại train/val
    n_total_orig = len(orig_run_ids)
    # Chia 56 run thành ~85% train (48 runs) và ~15% val (8 runs)
    n_train_orig = int(n_total_orig * (0.7 / 0.85)) # Giữ tỷ lệ 70/15 của bộ dữ liệu gốc

    train_ids_orig = orig_run_ids[:n_train_orig]
    val_ids_orig = orig_run_ids[n_train_orig:]
    
    # Lấy toàn bộ dữ liệu train đã có (gốc + augmented) từ bước trước
    all_train_ids_base = [rid for rid in df_train_val_augmented['run_id'].unique() if any(orig_id in rid for orig_id in train_ids_orig)]
    df_train_base = df_train_val_augmented[df_train_val_augmented['run_id'].isin(all_train_ids_base)].copy()
    
    # Lấy ra tập val gốc, không bị thay đổi
    df_val = df_train_val_augmented[df_train_val_augmented['run_id'].isin(val_ids_orig)].copy()

    # --- TĂNG CƯỜNG EXTREME CHỈ TRÊN PHẦN TRAIN BASE ---
    print(f"Using {len(df_train_base)} records as base for extreme augmentation.")
    
    # --- BƯỚC 2: TĂNG CƯỜNG EXTREME CHỈ TRÊN TẬP TRAIN GỐC ---
    print("\nStarting EXTREME augmentation ONLY on the training set...")
    augmentor = ExtremeAugmentor(df_train_base)
    extreme_parts = []
    
    # --- KIỂM TRA VÀ CHẠY NỘI SUY (ĐÃ BỊ TẮT THEO MẶC ĐỊNH) ---
    if ext_config.get('run_interpolation', False):
        df_interp = augmentor.augment_hourly_interpolation(num_interpolations=ext_config['num_interpolations'])
        if not df_interp.empty: extreme_parts.append(df_interp)
    else:
        print("\nMethod 1: Hourly Interpolation is DISABLED by config.")

    # (Gọi các hàm augment khác, sử dụng tham số từ ext_config)
    df_weather = augmentor.augment_synthetic_weather(num_weather_scenarios=ext_config['num_weather_scenarios'])
    if not df_weather.empty: extreme_parts.append(df_weather)

    df_scenarios = augmentor.augment_multi_scenarios(num_scenarios=ext_config['num_multi_scenarios'])
    if not df_scenarios.empty: extreme_parts.append(df_scenarios)

    df_timeshift = augmentor.augment_time_perturbation(num_perturbations=ext_config['num_time_perturbations'], max_shift_minutes=ext_config['max_shift_minutes'])
    if not df_timeshift.empty: extreme_parts.append(df_timeshift)
    
    # --- BƯỚC 3: GỘP VÀ LƯU ---
    if extreme_parts:
        df_extreme_new = pd.concat(extreme_parts, ignore_index=True)
        
        # Dữ liệu train cuối cùng = train base + extreme mới
        df_train_final = pd.concat([df_train_base, df_extreme_new], ignore_index=True)
        
        print("\nEXTREME AUGMENTATION SUMMARY...")
        print(f"Total final records: {len(df_train_final)}")
        
        df_train_final.to_parquet(output_path, index=False)
        print(f"\n✓ Saved final EXTREME augmented dataset to {output_path}")

        # --- GỌI HÀM KIỂM TRA LEAKAGE ---
        df_test_empty = pd.DataFrame(columns=df_val.columns)
        if not validate_no_leakage(df_train_final, df_val, df_test_empty):
            sys.exit("Stopping due to detected data leakage.")

        if not args.no_validate:
            validation_result = validate_processed_dataset(output_path, DEFAULT_REQUIRED_COLUMNS)
            if not validation_result.is_valid:
                print("Validation FAILED:", validation_result.errors)
                sys.exit(1)
            print("✓ Validation successful!")
    else:
        print("\n⚠ No new extreme augmented data was generated.")


if __name__ == '__main__':
    main()