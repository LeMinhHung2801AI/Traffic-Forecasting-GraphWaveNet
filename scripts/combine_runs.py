"""Combine collection runs into a validated parquet dataset."""

import argparse
import glob
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] # parents[1] vì file đang ở trong scripts/
sys.path.append(str(PROJECT_ROOT))

from utils.validation.dataset_validation import (
    DEFAULT_REQUIRED_COLUMNS,
    DatasetValidationResult,
    validate_processed_dataset,
)

warnings.filterwarnings("ignore")

def _resolve_relative(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _print_validation(result: DatasetValidationResult) -> None:
    print("\n== Dataset Validation Report ==")
    print(f"Path: {result.path}")
    print(f"Exists: {result.exists}")
    if not result.exists:
        for error in result.errors:
            print(f"  - {error}")
        return
    print(f"Rows: {result.rows}")
    print(f"Missing Columns: {result.missing_columns or '-'}")
    print(f"Columns With Nulls: {result.null_columns or '-'}")
    print(f"Duplicate Rows: {result.duplicate_rows}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine collection runs into a parquet dataset.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/runs"),
        help="Directory containing run_* folders (default: data/runs)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/all_runs_combined.parquet"),
        help="Destination parquet file (default: data/processed/all_runs_combined.parquet)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run dataset validation after writing the parquet",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="Additional required columns for validation (space separated)",
    )
    return parser


def parse_run_timestamp(run_dir_name: str) -> Optional[datetime]:
    """Extract timestamp from run directory name"""
    # run_20251030_032440 -> 2025-10-30 03:24:40
    try:
        timestamp_str = run_dir_name.replace('run_', '')
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except Exception:
        return None


def load_run_data(run_dir: Path) -> Optional[List[dict]]:
    """Load all data from a single run directory"""
    run_path = Path(run_dir)
    run_id = run_path.name
    timestamp = parse_run_timestamp(run_id)
    
    if timestamp is None:
        return None
    
    # Load files
    files = {
        'traffic': run_path / 'traffic_edges.json',
        'weather': run_path / 'weather_snapshot.json',
        'nodes': run_path / 'nodes.json'
    }
    
    # Check if all required files exist
    if not all(f.exists() for f in files.values()):
        return None
    
    # Load traffic data
    with open(files['traffic']) as f:
        traffic_data = json.load(f)
    
    # Load weather data
    with open(files['weather']) as f:
        weather_data = json.load(f)
    
    # Load nodes for coordinates
    with open(files['nodes']) as f:
        nodes_data = json.load(f)
        nodes_dict = {node['node_id']: node for node in nodes_data}
    
    # Process traffic records
    records = []
    for traffic in traffic_data:
        node_a = nodes_dict.get(traffic['node_a_id'], {})
        node_b = nodes_dict.get(traffic['node_b_id'], {})
        
        # Convert duration_sec to duration_min
        duration_sec = traffic.get('duration_sec', 0)
        duration_min = duration_sec / 60.0 if duration_sec else 0
        
        record = {
            'run_id': run_id,
            'timestamp': timestamp,
            'node_a_id': traffic['node_a_id'],
            'node_b_id': traffic['node_b_id'],
            'speed_kmh': traffic['speed_kmh'],
            'distance_km': traffic.get('distance_km', 0),
            'duration_min': duration_min,
            'lat_a': node_a.get('lat', 0),
            'lon_a': node_a.get('lon', 0),
            'lat_b': node_b.get('lat', 0),
            'lon_b': node_b.get('lon', 0),
        }
        
        # Add weather data (assuming same for all edges in a run)
        if weather_data:
            weather = weather_data[0] if isinstance(weather_data, list) else weather_data
            record.update({
                'temperature_c': weather.get('temperature_c', None),
                'wind_speed_kmh': weather.get('wind_speed_kmh', None),
                'precipitation_mm': weather.get('precipitation_mm', None),
                'humidity_percent': weather.get('humidity_percent', None),
                'weather_description': weather.get('description', None)
            })
        
        records.append(record)
    
    return records


def combine_all_runs(
    runs_dir: Path,
    output_file: Path,
    *,
    validate: bool = False,
    extra_required: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Combine all run data into a single parquet file."""

    runs_path = _resolve_relative(runs_dir)
    output_path = _resolve_relative(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMBINING RUN DATA")
    print("="*70)
    print()
    print(f"Looking in: {runs_path}")
    print()
    
    # Find all run directories
    run_dirs = sorted(glob.glob(str(runs_path / 'run_*')))
    print(f"Found {len(run_dirs)} run directories")
    print()
    
    # Load all data
    all_records = []
    successful_runs = 0
    
    for i, run_dir in enumerate(run_dirs, 1):
        records = load_run_data(Path(run_dir))
        
        if records:
            all_records.extend(records)
            successful_runs += 1
            
        if i % 10 == 0 or i == len(run_dirs):
            print(f"Processed: {i}/{len(run_dirs)} runs ({successful_runs} successful)")
    
    print()
    print(f"Total records: {len(all_records):,}")
    
    if not all_records:
        print("Error: No data loaded!")
        raise SystemExit(1)
    
    # Convert to DataFrame
    print("\nConverting to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique runs: {df['run_id'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Unique edges: {len(df[['node_a_id', 'node_b_id']].drop_duplicates())}")
    print(f"  Speed range: {df['speed_kmh'].min():.2f} - {df['speed_kmh'].max():.2f} km/h")
    
    if 'temperature_c' in df.columns:
        print(f"  Temperature range: {df['temperature_c'].min():.2f} - {df['temperature_c'].max():.2f} C")
    
    # Check missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("  None")
    
    # Save to parquet
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {file_size_mb:.2f} MB")

    if validate:
        required = list(DEFAULT_REQUIRED_COLUMNS)
        if extra_required:
            required.extend(extra_required)
        validation = validate_processed_dataset(output_path, required)
        _print_validation(validation)
        if not validation.is_valid:
            raise SystemExit("Dataset validation failed; see report above.")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return df


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    combine_all_runs(
        runs_dir=args.runs_dir,
        output_file=args.output_file,
        validate=args.validate,
        extra_required=args.require,
    )


if __name__ == "__main__":
    main()
