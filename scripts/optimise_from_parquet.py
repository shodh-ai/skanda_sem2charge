#!/usr/bin/env python3
"""
Step 2: Create optimized binary chunks from pruned parquet
- Proper normalization for DNN (Min-Max or Z-score)
- Train/Val/Test split
- Handles negative values appropriately
- Multi-worker processing support

Usage:
    python create_optimized_dataset.py
"""

import numpy as np
import pandas as pd
import yaml
import json
import zlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple
from litdata import optimize
from sklearn.model_selection import train_test_split


# ============================================================================
# CONFIGURATION
# ============================================================================


def load_config():
    """Load config files"""
    config_dir = Path.cwd() / "configs"

    with open(config_dir / "optimise_config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open(config_dir / "paths.yml", "r") as f:
        paths = yaml.safe_load(f)

    return config, paths


# ============================================================================
# DATA SPLITTING
# ============================================================================


def split_data(df, config):
    """
    Split data into train/val/test sets

    Args:
        df: DataFrame with pruned data
        config: Configuration dict with split ratios

    Returns:
        train_df, val_df, test_df
    """
    # Get split ratios from config
    train_ratio = config["data"].get("train_split", 0.7)
    val_ratio = config["data"].get("val_split", 0.15)
    test_ratio = config["data"].get("test_split", 0.15)

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total}"

    print(f"\n📊 Splitting data:")
    print(f"   Train: {train_ratio*100:.1f}%")
    print(f"   Val:   {val_ratio*100:.1f}%")
    print(f"   Test:  {test_ratio*100:.1f}%")

    # Random seed for reproducibility
    random_state = config["data"].get("random_seed", 42)

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=random_state, shuffle=True
    )

    # Second split: val vs test
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_adjusted),
        random_state=random_state,
        shuffle=True,
    )

    print(f"\n✓ Data split complete:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")

    return train_df, val_df, test_df


# ============================================================================
# NORMALIZATION
# ============================================================================


def compute_normalization_stats(
    train_df: pd.DataFrame, norm_method: str = "zscore"
) -> Dict:
    """
    Compute normalization stats from training data only

    Args:
        train_df: Training DataFrame
        norm_method: 'zscore' or 'minmax'

    Returns:
        Dictionary with normalization parameters
    """
    print(f"\n📊 Computing normalization statistics ({norm_method})...")
    print(f"   Training samples: {len(train_df)}")

    def get_stats(feature_list, method="zscore"):
        """Get normalization stats for list of features"""
        arrays = np.array(feature_list.tolist())

        if method == "zscore":
            # Z-score normalization: (x - mean) / std
            mean = np.nanmean(arrays, axis=0).tolist()
            std = np.nanstd(arrays, axis=0).tolist()

            # Replace zero std with 1.0 (constant features)
            std = [s if s > 1e-8 else 1.0 for s in std]

            return {"method": "zscore", "mean": mean, "std": std}

        elif method == "minmax":
            # Min-Max normalization: (x - min) / (max - min)
            # Maps to [0, 1] range
            min_vals = np.nanmin(arrays, axis=0).tolist()
            max_vals = np.nanmax(arrays, axis=0).tolist()

            # Handle constant features
            ranges = [
                mx - mn if abs(mx - mn) > 1e-8 else 1.0
                for mn, mx in zip(min_vals, max_vals)
            ]

            return {
                "method": "minmax",
                "min": min_vals,
                "max": max_vals,
                "range": ranges,
            }

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    stats = {
        "normalization_method": norm_method,
        "input_params": get_stats(train_df["input_params"], norm_method),
        "microstructure_outputs": get_stats(
            train_df["microstructure_outputs"], norm_method
        ),
        "performance_outputs": get_stats(train_df["performance_outputs"], norm_method),
    }

    print(f"✅ Statistics computed\n")

    return stats


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================


def decompress_image(compressed_bytes: bytes, shape: List[int]) -> np.ndarray:
    """Decompress image from bytes"""
    decompressed = zlib.decompress(compressed_bytes)
    image = np.frombuffer(decompressed, dtype=np.uint8).reshape(shape)
    return image.astype(np.float32) / 255.0  # Normalize images to [0, 1]


def normalize_features(features: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Normalize features using computed statistics

    Args:
        features: Input features array
        stats: Statistics dict with normalization parameters

    Returns:
        Normalized features
    """
    method = stats.get("method", "zscore")

    if method == "zscore":
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        return (features - mean) / std

    elif method == "minmax":
        min_vals = np.array(stats["min"], dtype=np.float32)
        ranges = np.array(stats["range"], dtype=np.float32)
        return (features - min_vals) / ranges

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def process_sample(sample_tuple: Tuple[Dict, Dict]) -> Dict:
    """
    Process one sample (row + norm_stats packaged together)

    Args:
        sample_tuple: (row_dict, norm_stats)

    Returns:
        Processed sample dict
    """
    row_dict, norm_stats = sample_tuple

    # Decompress image
    image = decompress_image(row_dict["image_compressed"], row_dict["image_shape"])

    # Get features as arrays
    input_params = np.array(row_dict["input_params"], dtype=np.float32)
    micro_outputs = np.array(row_dict["microstructure_outputs"], dtype=np.float32)
    perf_outputs = np.array(row_dict["performance_outputs"], dtype=np.float32)

    # Normalize
    input_params = normalize_features(input_params, norm_stats["input_params"])
    micro_outputs = normalize_features(
        micro_outputs, norm_stats["microstructure_outputs"]
    )
    perf_outputs = normalize_features(perf_outputs, norm_stats["performance_outputs"])

    return {
        "image": image,
        "input_params": input_params,
        "microstructure_outputs": micro_outputs,
        "performance_outputs": perf_outputs,
        "sample_id": row_dict["sample_id"],
        "param_id": row_dict["param_id"],
    }


# ============================================================================
# OPTIMIZATION
# ============================================================================


def optimize_split(
    split_name: str, df_split: pd.DataFrame, config: Dict, paths: Dict, norm_stats: Dict
):
    """Optimize one split using multi-worker processing"""

    print(f"\n{'='*80}")
    print(f"🔄 Optimizing {split_name.upper()} split")
    print(f"{'='*80}")

    # Convert to list of dicts
    rows = df_split.to_dict("records")
    print(f"   Samples: {len(rows):,}")

    # Output directory
    output_base = Path(paths["data"]["output"]["optimized_dir"])

    if split_name == "train":
        output_dir = output_base / paths["data"]["output"]["train_dir"]
    elif split_name == "val":
        output_dir = output_base / paths["data"]["output"]["val_dir"]
    else:
        output_dir = output_base / paths["data"]["output"]["test_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Platform info
    num_workers = config["data"]["num_workers"]
    print(f"   Platform: {platform.system()}")
    print(f"   Output: {output_dir}")
    print(f"   Workers: {num_workers}")
    print(f"   Chunk size: {config['data']['chunk_size']}")
    print(f"{'='*80}\n")

    # Package each row with norm_stats
    inputs = [(row, norm_stats) for row in rows]

    # Optimize with multi-worker support
    optimize(
        fn=process_sample,
        inputs=inputs,
        output_dir=str(output_dir),
        chunk_size=config["data"]["chunk_size"],
        num_workers=num_workers,
        mode="overwrite",
    )

    print(f"\n✅ {split_name.upper()} optimization complete!\n")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("⚡ Create Optimized Dataset with Train/Val/Test Split")
    print("=" * 80)
    print(f"Platform: {platform.system()} {platform.machine()}\n")

    # Load config
    config, paths = load_config()
    print(f"✓ Configuration loaded")
    print(f"   Workers: {config['data']['num_workers']}")
    print(f"   Chunk size: {config['data']['chunk_size']}")

    # Normalization method
    norm_method = config["data"].get("normalization_method", "zscore")
    print(f"   Normalization: {norm_method}")

    # Load pruned data
    pruned_file = Path("data") / "intermediate_pruned.parquet"
    print(f"\n📂 Loading pruned data from: {pruned_file}")

    if not pruned_file.exists():
        raise FileNotFoundError(f"Pruned data not found: {pruned_file}")

    df_full = pd.read_parquet(pruned_file)
    print(f"✓ Loaded {len(df_full):,} samples")
    print(f"  Columns: {list(df_full.columns)}")

    # Split data into train/val/test
    train_df, val_df, test_df = split_data(df_full, config)

    # Compute normalization stats from training data only
    norm_stats = compute_normalization_stats(train_df, norm_method)

    # Validate normalization stats
    print("\n📋 Normalization Statistics Summary:")
    print(f"   Method: {norm_stats['normalization_method']}")
    print(
        f"   Input features: {len(norm_stats['input_params'].get('mean', norm_stats['input_params'].get('min', [])))} "
    )
    print(
        f"   Microstructure outputs: {len(norm_stats['microstructure_outputs'].get('mean', norm_stats['microstructure_outputs'].get('min', [])))}"
    )
    print(
        f"   Performance outputs: {len(norm_stats['performance_outputs'].get('mean', norm_stats['performance_outputs'].get('min', [])))}"
    )

    # Save normalization stats
    output_base = Path(paths["data"]["output"]["optimized_dir"])
    stats_path = output_base / paths["data"]["output"]["stats_file"]

    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n💾 Normalization stats saved: {stats_path}\n")

    # Optimize each split
    optimize_split("train", train_df, config, paths, norm_stats)
    optimize_split("val", val_df, config, paths, norm_stats)
    optimize_split("test", test_df, config, paths, norm_stats)

    # Final summary
    print("=" * 80)
    print("🎉 All optimizations complete!")
    print("=" * 80)
    print(f"📁 Output structure:")
    print(f"   {output_base}/")
    print(f"   ├── train/          ({len(train_df):,} samples)")
    print(f"   ├── val/            ({len(val_df):,} samples)")
    print(f"   ├── test/           ({len(test_df):,} samples)")
    print(f"   └── {paths['data']['output']['stats_file']}")
    print("=" * 80)

    # Data quality summary
    print(f"\n📊 Data Quality Summary:")
    print(f"   Total samples: {len(df_full):,}")
    print(f"   Train: {len(train_df):,} ({len(train_df)/len(df_full)*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} ({len(val_df)/len(df_full)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} ({len(test_df)/len(df_full)*100:.1f}%)")
    print(f"   Normalization: {norm_method}")
    print("=" * 80)


if __name__ == "__main__":
    main()
