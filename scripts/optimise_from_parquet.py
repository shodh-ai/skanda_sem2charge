#!/usr/bin/env python3
"""
Step 3: Create optimized binary chunks from pruned parquet
- Proper normalization for DNN (Min-Max or Z-score)
- Train/Val/Test split
- Multi-worker processing support
- Full config integration

Usage:
    python scripts/optimise_from_parquet.py
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
# DATA VALIDATION
# ============================================================================


def validate_pruned_data(df: pd.DataFrame, config: Dict):
    """Validate that pruned parquet has expected structure"""

    print(f"\n🔍 Validating pruned data structure...")

    required_cols = [
        "sample_id",
        "param_id",
        "image_compressed",
        "image_shape",
        "input_params",
        "microstructure_outputs",
        "performance_outputs",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Expected feature counts from config
    expected_input = len(config["data"]["input_features"])
    expected_micro = len(config["data"]["output_features"]["microstructure"])
    expected_perf = len(config["data"]["output_features"]["performance"])

    # Check a sample row
    sample_row = df.iloc[0]

    actual_input = len(sample_row["input_params"])
    actual_micro = len(sample_row["microstructure_outputs"])
    actual_perf = len(sample_row["performance_outputs"])

    print(f"   ✓ Required columns present: {len(required_cols)}")
    print(f"   ✓ Input features: {actual_input} (expected: {expected_input})")
    print(f"   ✓ Microstructure outputs: {actual_micro} (expected: {expected_micro})")
    print(f"   ✓ Performance outputs: {actual_perf} (expected: {expected_perf})")

    # Validate counts match
    if actual_input != expected_input:
        raise ValueError(
            f"Input feature count mismatch: {actual_input} != {expected_input}"
        )
    if actual_micro != expected_micro:
        raise ValueError(
            f"Microstructure output count mismatch: {actual_micro} != {expected_micro}"
        )
    if actual_perf != expected_perf:
        raise ValueError(
            f"Performance output count mismatch: {actual_perf} != {expected_perf}"
        )

    print(f"   ✓ All validations passed!\n")


# ============================================================================
# DATA SPLITTING
# ============================================================================


def split_data(
    df: pd.DataFrame, config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets using config ratios

    Args:
        df: DataFrame with pruned data
        config: Configuration dict with split ratios

    Returns:
        train_df, val_df, test_df
    """
    # Get split ratios from config (FIXED)
    splits = config["data"]["splits"]
    train_ratio = splits["train"]
    val_ratio = splits["val"]
    test_ratio = splits["test"]

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total}"

    print(f"\n📊 Splitting data:")
    print(f"   Train: {train_ratio*100:.1f}%")
    print(f"   Val:   {val_ratio*100:.1f}%")
    print(f"   Test:  {test_ratio*100:.1f}%")

    # Random seed for reproducibility
    random_state = config["data"]["random_seed"]
    print(f"   Random seed: {random_state}")

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


def compute_normalization_stats(train_df: pd.DataFrame, config: Dict) -> Dict:
    """
    Compute normalization stats from training data only

    Args:
        train_df: Training DataFrame
        config: Config dict with normalization method

    Returns:
        Dictionary with normalization parameters
    """
    norm_method = config["data"]["normalization_method"]

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
        # Store feature names for reference
        "input_features": config["data"]["input_features"],
        "microstructure_features": config["data"]["output_features"]["microstructure"],
        "performance_features": config["data"]["output_features"]["performance"],
    }

    print(f"✅ Statistics computed")
    print(f"   Input features: {len(stats['input_features'])}")
    print(f"   Microstructure features: {len(stats['microstructure_features'])}")
    print(f"   Performance features: {len(stats['performance_features'])}")

    return stats


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================


def decompress_image(compressed_bytes: bytes, shape: List[int]) -> np.ndarray:
    """
    Decompress image from bytes

    Note: Images are already binary (0/1) from create_intermediate_dataset.py
    """
    decompressed = zlib.decompress(compressed_bytes)
    image = np.frombuffer(decompressed, dtype=np.uint8).reshape(shape)
    # Images are already 0/1, just convert to float32
    return image.astype(np.float32)


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

    # Get optimization settings from config
    num_workers = config["data"]["num_workers"]
    chunk_size = config["data"]["chunk_size"]

    print(f"   Platform: {platform.system()}")
    print(f"   Output: {output_dir}")
    print(f"   Workers: {num_workers}")
    print(f"   Chunk size: {chunk_size}")
    print(f"{'='*80}\n")

    # Package each row with norm_stats
    inputs = [(row, norm_stats) for row in rows]

    # Optimize with multi-worker support
    optimize(
        fn=process_sample,
        inputs=inputs,
        output_dir=str(output_dir),
        chunk_size=chunk_size,
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

    # Print config summary
    print(f"\n📋 Configuration Summary:")
    print(f"   Total samples: {config['data']['total_samples']}")
    print(f"   Params per sample: {config['data']['params_per_sample']}")
    print(
        f"   Expected total rows: {config['data']['total_samples'] * config['data']['params_per_sample']:,}"
    )
    print(f"   Random seed: {config['data']['random_seed']}")
    print(f"   Normalization: {config['data']['normalization_method']}")
    print(f"   Chunk size: {config['data']['chunk_size']}")
    print(f"   Workers: {config['data']['num_workers']}")
    print(
        f"   Split ratios: Train={config['data']['splits']['train']}, Val={config['data']['splits']['val']}, Test={config['data']['splits']['test']}"
    )

    # Load pruned data
    pruned_folder = Path("data") / "intermediate_pruned"

    # Fallback: Check if the folder has the .parquet extension (from previous runs)
    if (
        not pruned_folder.exists()
        and (Path("data") / "intermediate_pruned.parquet").exists()
    ):
        pruned_folder = Path("data") / "intermediate_pruned.parquet"

    print(f"\n📂 Loading pruned data from: {pruned_folder}")

    if not pruned_folder.exists():
        raise FileNotFoundError(f"Pruned data directory not found: {pruned_folder}")

    # Pandas automatically detects it is a directory and reads all part_*.parquet files
    df_full = pd.read_parquet(pruned_folder)
    print(f"✓ Loaded {len(df_full):,} samples")

    # Validate data structure
    validate_pruned_data(df_full, config)

    # Split data into train/val/test
    train_df, val_df, test_df = split_data(df_full, config)

    # Compute normalization stats from training data only
    norm_stats = compute_normalization_stats(train_df, config)

    # Save normalization stats
    output_base = Path(paths["data"]["output"]["optimized_dir"])
    output_base.mkdir(parents=True, exist_ok=True)
    stats_path = output_base / paths["data"]["output"]["stats_file"]

    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n💾 Normalization stats saved: {stats_path}")

    # Optimize each split
    optimize_split("train", train_df, config, paths, norm_stats)
    optimize_split("val", val_df, config, paths, norm_stats)
    optimize_split("test", test_df, config, paths, norm_stats)

    # Final summary
    print("\n" + "=" * 80)
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
    print(f"   Normalization: {config['data']['normalization_method']}")
    print(f"   Input features: {len(config['data']['input_features'])}")
    print(
        f"   Microstructure features: {len(config['data']['output_features']['microstructure'])}"
    )
    print(
        f"   Performance features: {len(config['data']['output_features']['performance'])}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
