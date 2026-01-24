#!/usr/bin/env python3
"""
Step 1: Create intermediate compressed parquet dataset (MPI-only)
Converts raw TIFF + CSV + Parquet → Compressed intermediate parquet

Usage: mpirun -n <num_processes> python scripts/create_intermediate_dataset.py
"""

import numpy as np
import pandas as pd
import tifffile
import yaml
import zlib
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore")

# ============================================================================
# MPI SETUP
# ============================================================================

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
except ImportError:
    print("❌ ERROR: mpi4py not found!")
    print("Install with: pip install mpi4py")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================


def load_config() -> Tuple[Dict, Dict]:
    """Load config files"""
    config_dir = Path.cwd() / "configs"

    with open(config_dir / "optimise_config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open(config_dir / "paths.yml", "r") as f:
        paths = yaml.safe_load(f)

    return config, paths


# ============================================================================
# DATA PROCESSING
# ============================================================================


def load_image_binary(image_path: Path, pore_value: int) -> np.ndarray:
    """Load TIFF and convert to binary (0/1 uint8)"""
    img = tifffile.imread(image_path)
    return np.where(img == pore_value, 0, 1).astype(np.uint8)


def compress_image(image: np.ndarray) -> bytes:
    """Compress binary image using zlib"""
    return zlib.compress(image.tobytes(), level=9)


def safe_float(value):
    """Convert value to float, handling None and NaN"""
    if value is None:
        return np.nan
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def get_sample_data(
    sample_id: int,
    param_id: int,
    tau_df: pd.DataFrame,
    parquet_dir: Path,
    parquet_pattern: str,
    input_features: List[str],
) -> Tuple[List, List, List]:
    """
    Extract all features for a sample+param combination

    Returns:
        (input_params, microstructure_outputs, performance_outputs)

    Raises:
        Exception if data cannot be extracted
    """
    # Get microstructure from tau_results
    tau_rows = tau_df[tau_df["id"] == sample_id]
    if len(tau_rows) == 0:
        raise ValueError(f"Sample {sample_id} not in tau_results")
    tau_row = tau_rows.iloc[0]

    # Get battery params and performance from parquet
    parquet_path = parquet_dir / parquet_pattern.format(sample_id)
    if not parquet_path.exists():
        raise ValueError(f"Parquet file not found: {parquet_path}")

    parquet_df = pd.read_parquet(parquet_path)

    # Get specific param_id
    param_rows = parquet_df[parquet_df["param_id"] == param_id]
    if len(param_rows) == 0:
        raise ValueError(f"Param {param_id} not found in sample {sample_id}")
    parquet_row = param_rows.iloc[0]

    # Extract input params (allow NaN)
    input_params = []
    for feat in input_features:
        val = parquet_row.get(feat, None)
        input_params.append(safe_float(val))

    # Extract microstructure outputs (allow NaN)
    microstructure = [
        safe_float(tau_row.get("D_eff", None)),
        safe_float(tau_row.get("porosity_measured", None)),
        safe_float(tau_row.get("tau_factor", None)),
        safe_float(parquet_row.get("bruggeman_derived", None)),
    ]

    # Extract performance outputs
    capacity_trend = parquet_row.get("capacity_trend_ah", None)

    if capacity_trend is not None and len(capacity_trend) > 0:
        capacity_trend = np.array(capacity_trend)
        initial_cap = safe_float(capacity_trend[0])
        final_cap = safe_float(capacity_trend[-1])

        # Avoid division by zero
        if initial_cap > 0 and not np.isnan(initial_cap):
            retention = float((final_cap / initial_cap) * 100)
        else:
            retention = np.nan

        total_cycles = int(len(capacity_trend))
    else:
        initial_cap = final_cap = retention = np.nan
        total_cycles = 0

    performance = [
        safe_float(parquet_row.get("eol_cycle_measured", None)),
        initial_cap,
        final_cap,
        retention,
        float(total_cycles),
    ]

    return input_params, microstructure, performance


# ============================================================================
# DATASET CREATION
# ============================================================================


def discover_samples(config: Dict, paths: Dict) -> List[int]:
    """Find all available samples"""
    base_dir = Path(paths["data"]["base_dir"])
    images_dir = base_dir / paths["data"]["input"]["images_dir"]
    image_pattern = paths["data"]["input"]["image_pattern"]

    available = []
    for sample_id in range(config["data"]["total_samples"]):
        if (images_dir / image_pattern.format(sample_id)).exists():
            available.append(sample_id)

    return available


def distribute_samples(sample_ids: List[int], rank: int, world_size: int) -> List[int]:
    """Distribute samples across MPI ranks (round-robin)"""
    return [s for i, s in enumerate(sample_ids) if i % world_size == rank]


def create_intermediate_parquet(
    sample_ids: List[int],
    config: Dict,
    paths: Dict,
    tau_df: pd.DataFrame,
    rank: int,
    world_size: int,
):
    """Create compressed intermediate parquet (MPI-parallel)"""

    base_dir = Path(paths["data"]["base_dir"])
    images_dir = base_dir / paths["data"]["input"]["images_dir"]
    image_pattern = paths["data"]["input"]["image_pattern"]
    parquet_dir = base_dir / paths["data"]["input"]["parquet_dir"]
    parquet_pattern = paths["data"]["input"]["parquet_pattern"]

    output_dir = Path(paths["data"]["output"]["intermediate_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each rank creates its own file
    output_file = output_dir / f"data_rank{rank:04d}.parquet"

    num_params = config["data"]["params_per_sample"]
    input_features = config["data"]["input_features"]
    pore_value = config["data"]["image"]["pore_value"]

    # Distribute samples
    my_samples = distribute_samples(sample_ids, rank, world_size)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"📦 Creating Intermediate Dataset")
        print(f"   Total samples: {len(sample_ids)}")
        print(f"   Params per sample: {num_params}")
        print(f"   Expected total: {len(sample_ids) * num_params}")
        print(f"   MPI ranks: {world_size}")
        print(f"   Samples per rank: ~{len(sample_ids) // world_size}")
        print(f"   Output: {output_dir}")
        print(f"{'='*80}\n")

    comm.Barrier()

    # Process samples
    rows = []
    error_log = {
        "image_load_failed": 0,
        "image_compress_failed": 0,
        "data_extraction_failed": 0,
    }

    for sample_id in tqdm(
        my_samples, desc=f"Rank {rank}", position=rank, disable=(rank != 0)
    ):
        image_path = images_dir / image_pattern.format(sample_id)

        if not image_path.exists():
            error_log["image_load_failed"] += num_params
            continue

        # Load and compress image once per sample
        try:
            image = load_image_binary(image_path, pore_value)
            image_compressed = compress_image(image)
            image_shape = list(image.shape)
        except Exception as e:
            error_log["image_compress_failed"] += num_params
            continue

        # Process all parameter variations
        for param_id in range(num_params):
            try:
                input_params, microstructure, performance = get_sample_data(
                    sample_id,
                    param_id,
                    tau_df,
                    parquet_dir,
                    parquet_pattern,
                    input_features,
                )

                rows.append(
                    {
                        "sample_id": sample_id,
                        "param_id": param_id,
                        "image_compressed": image_compressed,
                        "image_shape": image_shape,
                        "input_params": input_params,
                        "microstructure_outputs": microstructure,
                        "performance_outputs": performance,
                    }
                )

            except Exception as e:
                error_log["data_extraction_failed"] += 1

    # Save parquet
    df = pd.DataFrame(rows)

    if len(df) > 0 or rank == 0:
        total_errors = sum(error_log.values())
        print(f"\n[Rank {rank}] 💾 Saving {len(df)} rows (errors: {total_errors})")

    df.to_parquet(output_file, compression="snappy", index=False)

    comm.Barrier()

    # Gather stats
    all_row_counts = comm.gather(len(df), root=0)
    all_errors = comm.gather(error_log, root=0)
    all_file_sizes = comm.gather(output_file.stat().st_size / (1024 * 1024), root=0)

    if rank == 0:
        total_rows = sum(all_row_counts)
        total_size_mb = sum(all_file_sizes)
        expected = len(sample_ids) * num_params

        # Aggregate errors
        total_error_counts = {}
        for err_dict in all_errors:
            for key, val in err_dict.items():
                total_error_counts[key] = total_error_counts.get(key, 0) + val

        print(f"\n{'='*80}")
        print(f"✅ Intermediate Dataset Created!")
        print(f"   Expected rows: {expected}")
        print(f"   Created rows: {total_rows}")
        print(f"   Missing rows: {expected - total_rows}")
        print(f"   Success rate: {total_rows/expected*100:.2f}%")
        print(f"   Total size: {total_size_mb:.1f} MB")
        print(f"   Files: {world_size} parquet files")

        if sum(total_error_counts.values()) > 0:
            print(f"\n   ⚠️  Error summary:")
            for error_type, count in total_error_counts.items():
                if count > 0:
                    print(f"      {error_type}: {count}")

        print(f"   Location: {output_dir}")
        print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================


def main():
    if rank == 0:
        print("=" * 80)
        print("📦 Step 1: Create Intermediate Compressed Dataset (MPI)")
        print("=" * 80)
        print(f"🚀 Running with {world_size} MPI processes")
        print("=" * 80)

    # Load config
    config, paths = load_config()

    if rank == 0:
        print(f"\n✓ Configuration loaded")

    # Discover samples
    samples = discover_samples(config, paths)

    if rank == 0:
        print(f"✓ Discovered {len(samples)} samples")
        print(
            f"✓ Expected data points: {len(samples) * config['data']['params_per_sample']}\n"
        )

    # Load CSV
    if rank == 0:
        print("📂 Loading CSV metadata...")

    base_dir = Path(paths["data"]["base_dir"])
    tau_results = pd.read_csv(base_dir / paths["data"]["input"]["tau_results_csv"])

    if rank == 0:
        print(f"✓ Loaded {len(tau_results)} tau results\n")

    # Create intermediate dataset
    create_intermediate_parquet(samples, config, paths, tau_results, rank, world_size)

    if rank == 0:
        print("=" * 80)
        print("🎉 Intermediate dataset creation complete!")
        print("=" * 80)
        print("\nNext step: python scripts/optimize_from_parquet.py")
        print("=" * 80)


if __name__ == "__main__":
    main()
