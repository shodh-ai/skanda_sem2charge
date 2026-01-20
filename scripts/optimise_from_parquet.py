#!/usr/bin/env python3
"""
Step 2: Create optimized binary chunks from intermediate parquet
Supports multi-worker processing on both macOS and Linux
- macOS: Works with spawn mode (slower but functional)
- Linux: Full speed with fork mode and 256+ workers
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
from tqdm import tqdm

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
# NORMALIZATION
# ============================================================================


def compute_normalization_stats(intermediate_dir: Path) -> Dict:
    """Compute normalization stats from training parquet files"""

    print("\n📊 Computing normalization statistics...")

    # Find all training parquet files (from MPI ranks)
    train_files = sorted(intermediate_dir.glob("train_rank*.parquet"))

    if len(train_files) == 0:
        # Fallback to single file (non-MPI mode)
        single_file = intermediate_dir / "train.parquet"
        if single_file.exists():
            train_files = [single_file]
        else:
            raise FileNotFoundError("No training parquet files found")

    print(f"   Loading {len(train_files)} file(s)...")

    # Load and concatenate all training files
    dfs = [pd.read_parquet(f) for f in train_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"   Total training samples: {len(df)}")

    def get_stats(feature_list):
        """Get mean/std/min/max for list of features"""
        arrays = np.array(feature_list.tolist())

        # Handle NaNs
        mean = np.nanmean(arrays, axis=0).tolist()
        std = np.nanstd(arrays, axis=0).tolist()
        min_vals = np.nanmin(arrays, axis=0).tolist()
        max_vals = np.nanmax(arrays, axis=0).tolist()

        return {"mean": mean, "std": std, "min": min_vals, "max": max_vals}

    stats = {
        "input_params": get_stats(df["input_params"]),
        "microstructure_outputs": get_stats(df["microstructure_outputs"]),
        "performance_outputs": get_stats(df["performance_outputs"]),
    }

    print(f"✅ Statistics computed\n")

    return stats


# ============================================================================
# PROCESSING FUNCTIONS (MODULE LEVEL - PICKLABLE)
# ============================================================================


def decompress_image(compressed_bytes: bytes, shape: List[int]) -> np.ndarray:
    """Decompress image from bytes"""
    decompressed = zlib.decompress(compressed_bytes)
    image = np.frombuffer(decompressed, dtype=np.uint8).reshape(shape)
    return image.astype(np.float32)


def normalize_features(features: np.ndarray, stats: Dict) -> np.ndarray:
    """Normalize using z-score"""
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std


def process_sample(sample_tuple: Tuple[Dict, Dict]) -> Dict:
    """
    Process one sample (row + norm_stats packaged together)

    This function receives both the data and normalization stats as a tuple,
    making it fully self-contained and picklable on both macOS and Linux.

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


def optimize_split(split_name: str, config: Dict, paths: Dict, norm_stats: Dict):
    """Optimize one split using multi-worker processing (cross-platform)"""

    intermediate_dir = Path(paths["data"]["output"]["optimized_dir"]) / "intermediate"

    # Find all parquet files for this split (from MPI ranks)
    parquet_files = sorted(intermediate_dir.glob(f"{split_name}_rank*.parquet"))

    if len(parquet_files) == 0:
        # Fallback to single file (non-MPI mode)
        single_file = intermediate_dir / f"{split_name}.parquet"
        if single_file.exists():
            parquet_files = [single_file]
        else:
            print(f"⚠️  Skipping {split_name}: No parquet files found")
            return

    # Load all parquet files and concatenate
    print(f"\n📂 Loading {split_name} parquet files...")
    print(f"   Found {len(parquet_files)} file(s)")

    dfs = []
    for pf in parquet_files:
        dfs.append(pd.read_parquet(pf))

    df = pd.concat(dfs, ignore_index=True)

    # Convert to list of dicts
    rows = df.to_dict("records")

    print(f"✓ Loaded {len(rows)} total rows\n")

    # Output directory
    output_base = Path(paths["data"]["output"]["optimized_dir"])

    if split_name == "train":
        output_dir = output_base / paths["data"]["output"]["train_dir"]
    elif split_name == "val":
        output_dir = output_base / paths["data"]["output"]["val_dir"]
    else:
        output_dir = output_base / paths["data"]["output"]["test_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect platform
    is_macos = platform.system() == "Darwin"
    num_workers = config["data"]["num_workers"]

    if is_macos and num_workers > 1:
        print(f"⚠️  macOS detected with {num_workers} workers")
        print(
            f"   Note: macOS spawn mode has overhead. For testing, consider num_workers=1"
        )
        print(f"   Production Linux will be much faster with fork mode.\n")

    print(f"{'='*80}")
    print(f"🔄 Optimizing {split_name.upper()} split")
    print(f"   Platform: {platform.system()}")
    print(f"   Rows: {len(rows)}")
    print(f"   Output: {output_dir}")
    print(f"   Workers: {num_workers}")
    print(f"   Chunk size: {config['data']['chunk_size']}")
    print(f"{'='*80}\n")

    # Package each row with norm_stats (makes it fully self-contained)
    # This works on both macOS (spawn) and Linux (fork)
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
    print("⚡ Step 2: Optimize Dataset from Parquet (Multi-Worker)")
    print("=" * 80)
    print(f"Platform: {platform.system()} {platform.machine()}")

    # Load config
    config, paths = load_config()
    print(f"✓ Configuration loaded")
    print(f"   Workers: {config['data']['num_workers']}")
    print(f"   Chunk size: {config['data']['chunk_size']}\n")

    # Compute normalization stats from training set
    intermediate_dir = Path(paths["data"]["output"]["optimized_dir"]) / "intermediate"

    norm_stats = compute_normalization_stats(intermediate_dir)

    # Validate normalization stats
    print("📋 Normalization Statistics Summary:")
    print(f"   Input features: {len(norm_stats['input_params']['mean'])}")
    print(
        f"   Microstructure outputs: {len(norm_stats['microstructure_outputs']['mean'])}"
    )
    print(f"   Performance outputs: {len(norm_stats['performance_outputs']['mean'])}")

    # Check for issues
    input_stds = np.array(norm_stats["input_params"]["std"])
    if np.any(input_stds == 0):
        print(
            f"   ⚠️  Warning: {np.sum(input_stds == 0)} input features have zero std (constant values)"
        )

    print()

    # Save normalization stats
    output_base = Path(paths["data"]["output"]["optimized_dir"])
    stats_path = output_base / paths["data"]["output"]["stats_file"]

    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"💾 Normalization stats saved: {stats_path}\n")

    # Optimize each split
    for split_name in ["train", "val", "test"]:
        optimize_split(split_name, config, paths, norm_stats)

    # Final summary
    print("=" * 80)
    print("🎉 All optimizations complete!")
    print("=" * 80)
    print(f"📁 Output structure:")
    print(f"   {output_base}/")
    print(f"   ├── train/          (training data chunks)")
    print(f"   ├── val/            (validation data chunks)")
    print(f"   ├── test/           (test data chunks)")
    print(f"   └── {paths['data']['output']['stats_file']}")
    print("=" * 80)

    # Platform-specific notes
    if platform.system() == "Darwin":
        print("\n📝 macOS Testing Notes:")
        print("   - For faster testing, use num_workers: 1 in config.yml")
        print("   - Production Linux will be 10-50x faster with fork mode")
        print("=" * 80)
    else:
        print("\n🚀 Linux Production Mode")
        print("   - Fork mode enabled for maximum performance")
        print("   - Scale to 256+ workers for large datasets")
        print("=" * 80)


if __name__ == "__main__":
    main()
