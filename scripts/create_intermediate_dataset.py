#!/usr/bin/env python3
"""
Step 1: Create intermediate compressed parquet dataset (MPI-only)
Converts raw TIFF + CSV + Parquet → Compressed intermediate parquet with computed features

Output: One parquet file per sample (sample_0001.parquet, sample_0002.parquet, ...)

Usage: mpirun -n 8 python scripts/create_intermediate_dataset.py
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
# CONSTANTS
# ============================================================================

MAX_CYCLES = 5000  # Maximum cycle life ceiling
CURRENT_THRESHOLD = -0.01  # Amperes - threshold to detect discharge

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
# UTILITY FUNCTIONS
# ============================================================================


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


def load_image_binary(image_path: Path, pore_value: int) -> np.ndarray:
    """Load TIFF and convert to binary (0/1 uint8)"""
    img = tifffile.imread(image_path)
    return np.where(img == pore_value, 0, 1).astype(np.uint8)


def compress_image(image: np.ndarray) -> bytes:
    """Compress binary image using zlib"""
    return zlib.compress(image.tobytes(), level=9)


# ============================================================================
# FEATURE COMPUTATION FUNCTIONS
# ============================================================================


def compute_projected_cycle_life(parquet_row: pd.Series, tau_row: pd.Series) -> float:
    """
    Compute projected cycle life to 80% capacity using linear extrapolation

    Method:
    1. Get nominal capacity from first cycle
    2. Fit linear regression on LAST 50% of capacity trend
    3. Extrapolate to 80% capacity threshold
    4. Apply ceil() and clamp to [0, MAX_CYCLES]

    Returns:
        float: Projected cycle count (or NaN if computation fails)
    """
    try:
        capacity_trend = parquet_row.get("capacity_trend_ah", None)

        if capacity_trend is None or len(capacity_trend) < 2:
            return np.nan

        capacity_trend = np.array(capacity_trend)

        # Get nominal capacity (first cycle)
        C_nom = capacity_trend[0]
        if np.isnan(C_nom) or C_nom <= 0:
            return np.nan

        # 80% capacity threshold
        threshold = 0.8 * C_nom

        # Check if already reached 80%
        if np.any(capacity_trend <= threshold):
            # Find first cycle that dropped below threshold
            idx = np.where(capacity_trend <= threshold)[0][0]
            return float(idx + 1)  # +1 because cycles are 1-indexed

        # Use last 50% of data for linear regression
        n = len(capacity_trend)
        start_idx = n // 2

        y = capacity_trend[start_idx:]
        x = np.arange(start_idx, n)

        if len(x) < 2:
            return np.nan

        # Linear fit: y = mx + c
        coeffs = np.polyfit(x, y, deg=1)
        m, c = coeffs[0], coeffs[1]

        # If slope is non-negative (no degradation), return max
        if m >= 0:
            return float(MAX_CYCLES)

        # Solve for x when y = threshold: x = (threshold - c) / m
        predicted_cycle = (threshold - c) / m

        # Apply ceiling and clamp
        predicted_cycle = np.ceil(predicted_cycle)
        predicted_cycle = np.clip(predicted_cycle, 0, MAX_CYCLES)

        return float(predicted_cycle)

    except Exception as e:
        return np.nan


def compute_capacity_fade_rate(parquet_row: pd.Series, tau_row: pd.Series) -> float:
    """
    Compute capacity fade rate (Ah per cycle)

    Method:
    1. Fit linear regression on ALL capacity points
    2. Return abs(slope) as positive fade rate
    3. If slope ≥ 0 (increasing capacity), return NaN

    Returns:
        float: Capacity fade rate in Ah/cycle (positive) or NaN
    """
    try:
        capacity_trend = parquet_row.get("capacity_trend_ah", None)

        if capacity_trend is None or len(capacity_trend) < 2:
            return np.nan

        capacity_trend = np.array(capacity_trend)

        # Linear fit on all points
        x = np.arange(len(capacity_trend))
        y = capacity_trend

        coeffs = np.polyfit(x, y, deg=1)
        slope = coeffs[0]

        # If slope is non-negative (capacity increasing), return NaN
        if slope >= 0:
            return np.nan

        # Return absolute value (positive fade rate)
        return float(abs(slope))

    except Exception as e:
        return np.nan


def compute_internal_resistance(parquet_row: pd.Series, tau_row: pd.Series) -> float:
    """
    Compute initial DC internal resistance (DCIR) from cycle_first

    Method:
    1. V_rest = Voltage[0] (OCV at t=0)
    2. Find first index k where Current < -0.01 A
    3. V_discharge = Voltage[k]
    4. ΔV = V_rest - V_discharge
    5. R_dcir = ΔV / |I_discharge|

    Returns:
        float: Internal resistance in Ohms (or NaN)
    """
    try:
        cycle_first = parquet_row.get("cycle_first", {})

        if not cycle_first or not isinstance(cycle_first, dict):
            return np.nan

        # Extract voltage and current arrays
        voltage = cycle_first.get("Voltage_V", None)
        current = cycle_first.get("Current_A", None)

        if voltage is None or current is None:
            return np.nan

        voltage = np.array(voltage)
        current = np.array(current)

        if len(voltage) < 2 or len(current) < 2:
            return np.nan

        # V_rest = initial OCV
        V_rest = voltage[0]

        # Find first discharge point (Current < CURRENT_THRESHOLD)
        discharge_indices = np.where(current < CURRENT_THRESHOLD)[0]

        if len(discharge_indices) == 0:
            return np.nan

        k = discharge_indices[0]
        V_discharge = voltage[k]
        I_discharge = current[k]

        # Calculate resistance
        delta_V = V_rest - V_discharge
        R_dcir = delta_V / abs(I_discharge)

        # Sanity check (typical range: 0.001 - 0.5 Ω)
        if R_dcir < 0 or R_dcir > 1.0:
            return np.nan

        return float(R_dcir)

    except Exception as e:
        return np.nan


def compute_nominal_capacity(parquet_row: pd.Series, tau_row: pd.Series) -> float:
    """
    Compute nominal capacity (first cycle capacity)

    Returns:
        float: Nominal capacity in Ah
    """
    try:
        capacity_trend = parquet_row.get("capacity_trend_ah", None)

        if capacity_trend is None or len(capacity_trend) == 0:
            return np.nan

        return safe_float(capacity_trend[0])

    except Exception as e:
        return np.nan


def compute_energy_density(parquet_row: pd.Series, tau_row: pd.Series) -> float:
    """
    Compute energy density indicator (average discharge voltage)

    Method:
    1. Filter discharge phase (Current < 0)
    2. Compute simple mean(Voltage)

    Returns:
        float: Average discharge voltage in V (or NaN)
    """
    try:
        cycle_first = parquet_row.get("cycle_first", {})

        if not cycle_first or not isinstance(cycle_first, dict):
            return np.nan

        # Extract voltage and current arrays
        voltage = cycle_first.get("Voltage_V", None)
        current = cycle_first.get("Current_A", None)

        if voltage is None or current is None:
            return np.nan

        voltage = np.array(voltage)
        current = np.array(current)

        if len(voltage) == 0 or len(current) == 0:
            return np.nan

        # Filter discharge phase (current < 0)
        discharge_mask = current < 0

        if not np.any(discharge_mask):
            return np.nan

        discharge_voltage = voltage[discharge_mask]

        # Simple mean (could upgrade to time-weighted later)
        avg_voltage = np.mean(discharge_voltage)

        # Sanity check (typical Li-ion range: 2.5 - 4.5V)
        if avg_voltage < 2.0 or avg_voltage > 5.0:
            return np.nan

        return float(avg_voltage)

    except Exception as e:
        return np.nan


# ============================================================================
# FEATURE COMPUTATION REGISTRY
# ============================================================================

FEATURE_COMPUTERS = {
    "projected_cycle_life": compute_projected_cycle_life,
    "capacity_fade_rate": compute_capacity_fade_rate,
    "internal_resistance": compute_internal_resistance,
    "nominal_capacity": compute_nominal_capacity,
    "energy_density": compute_energy_density,
}


# ============================================================================
# DATA EXTRACTION
# ============================================================================


def extract_output_feature(
    feature_name: str, parquet_row: pd.Series, tau_row: pd.Series
) -> float:
    """
    Extract or compute a single output feature

    Priority:
    1. If feature has compute function in FEATURE_COMPUTERS → compute it
    2. Else try tau_row (from tau_results.csv)
    3. Else try parquet_row (from PyBaMM parquet)
    4. Else raise Exception

    Returns:
        float: Feature value (may be NaN if computation failed)

    Raises:
        ValueError: If feature cannot be found or computed
    """
    # Check if feature has compute function
    if feature_name in FEATURE_COMPUTERS:
        compute_func = FEATURE_COMPUTERS[feature_name]
        return compute_func(parquet_row, tau_row)

    # Try tau_results CSV first
    if feature_name in tau_row.index:
        return safe_float(tau_row[feature_name])

    # Try parquet row
    if feature_name in parquet_row.index:
        return safe_float(parquet_row[feature_name])

    # Feature not found
    raise ValueError(
        f"Feature '{feature_name}' not found in compute functions, "
        f"tau_results CSV, or parquet data"
    )


def process_single_sample(
    sample_id: int,
    tau_df: pd.DataFrame,
    parquet_dir: Path,
    parquet_pattern: str,
    images_dir: Path,
    image_pattern: str,
    config: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process a single sample and return DataFrame with all param variations

    Returns:
        (df, error_stats)
    """
    error_stats = {
        "image_load_failed": False,
        "image_compress_failed": False,
        "data_extraction_failed": 0,
    }

    num_params = config["data"]["params_per_sample"]
    input_features = config["data"]["input_features"]
    output_microstructure = config["data"]["output_features"]["microstructure"]
    output_performance = config["data"]["output_features"]["performance"]
    pore_value = config["data"]["image"]["pore_value"]

    # Get tau_results row for this sample
    tau_rows = tau_df[tau_df["id"] == sample_id]
    if len(tau_rows) == 0:
        error_stats["data_extraction_failed"] = num_params
        return pd.DataFrame(), error_stats
    tau_row = tau_rows.iloc[0]

    # Load image
    image_path = images_dir / image_pattern.format(sample_id)
    if not image_path.exists():
        error_stats["image_load_failed"] = True
        return pd.DataFrame(), error_stats

    try:
        image = load_image_binary(image_path, pore_value)
        image_compressed = compress_image(image)
        image_shape = list(image.shape)
    except Exception as e:
        error_stats["image_compress_failed"] = True
        return pd.DataFrame(), error_stats

    # Load parquet
    parquet_path = parquet_dir / parquet_pattern.format(sample_id)
    if not parquet_path.exists():
        error_stats["data_extraction_failed"] = num_params
        return pd.DataFrame(), error_stats

    parquet_df = pd.read_parquet(parquet_path)

    # Process all parameter variations
    rows = []
    for param_id in range(num_params):
        try:
            # Get param row
            param_rows = parquet_df[parquet_df["param_id"] == param_id]
            if len(param_rows) == 0:
                error_stats["data_extraction_failed"] += 1
                continue
            parquet_row = param_rows.iloc[0]

            # Extract input params
            input_params = []
            for feat in input_features:
                if feat not in parquet_row.index:
                    raise ValueError(f"Input feature '{feat}' not found")
                input_params.append(safe_float(parquet_row[feat]))

            # Extract/compute microstructure
            microstructure = []
            for feat in output_microstructure:
                value = extract_output_feature(feat, parquet_row, tau_row)
                microstructure.append(value)

            # Extract/compute performance
            performance = []
            for feat in output_performance:
                value = extract_output_feature(feat, parquet_row, tau_row)
                performance.append(value)

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
            error_stats["data_extraction_failed"] += 1

    return pd.DataFrame(rows), error_stats


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
# SAMPLE DISCOVERY
# ============================================================================


def discover_samples(tau_df: pd.DataFrame, paths: Dict) -> List[int]:
    """Find all available samples"""
    base_dir = Path(paths["data"]["base_dir"])
    images_dir = base_dir / paths["data"]["input"]["images_dir"]
    image_pattern = paths["data"]["input"]["image_pattern"]

    if rank == 0:
        print(f"\n🔍 Discovering samples from tau_results.csv...")

    all_sample_ids = tau_df["id"].unique().tolist()

    if rank == 0:
        print(f"   Found {len(all_sample_ids)} unique sample IDs in CSV")
        print(f"   Sample ID range: {min(all_sample_ids)} - {max(all_sample_ids)}")

    # Verify TIFF files exist
    available = []
    missing = []

    for sample_id in all_sample_ids:
        img_path = images_dir / image_pattern.format(sample_id)
        if img_path.exists():
            available.append(sample_id)
        else:
            missing.append(sample_id)

    if rank == 0:
        print(f"   ✓ {len(available)} samples have TIFF files")
        if len(missing) > 0:
            print(f"   ⚠️  {len(missing)} samples missing TIFF files")

    return sorted(available)


def distribute_samples(sample_ids: List[int], rank: int, world_size: int) -> List[int]:
    """Distribute samples across MPI ranks (round-robin)"""
    return [s for i, s in enumerate(sample_ids) if i % world_size == rank]


# ============================================================================
# DATASET CREATION
# ============================================================================


def create_intermediate_parquet(
    sample_ids: List[int],
    config: Dict,
    paths: Dict,
    tau_df: pd.DataFrame,
    rank: int,
    world_size: int,
):
    """Create intermediate parquet files (ONE FILE PER SAMPLE)"""

    if len(sample_ids) == 0:
        if rank == 0:
            print("\n❌ No samples to process. Exiting.")
        return

    base_dir = Path(paths["data"]["base_dir"])
    images_dir = base_dir / paths["data"]["input"]["images_dir"]
    image_pattern = paths["data"]["input"]["image_pattern"]
    parquet_dir = base_dir / paths["data"]["input"]["parquet_dir"]
    parquet_pattern = paths["data"]["input"]["parquet_pattern"]

    output_dir = Path(paths["data"]["output"]["intermediate_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    num_params = config["data"]["params_per_sample"]
    input_features = config["data"]["input_features"]
    output_microstructure = config["data"]["output_features"]["microstructure"]
    output_performance = config["data"]["output_features"]["performance"]

    # Distribute samples
    my_samples = distribute_samples(sample_ids, rank, world_size)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"📦 Creating Intermediate Dataset (ONE FILE PER SAMPLE)")
        print(f"{'='*80}")
        print(f"   Total samples: {len(sample_ids)}")
        print(f"   Params per sample: {num_params}")
        print(f"   Expected files: {len(sample_ids)} parquet files")
        print(f"   Expected rows per file: {num_params}")
        print(f"   MPI ranks: {world_size}")
        print(f"   Samples per rank: ~{len(sample_ids) // world_size}")
        print(f"\n   Input features ({len(input_features)})")
        print(f"   Output - Microstructure ({len(output_microstructure)})")
        print(
            f"   Output - Performance ({len(output_performance)}) - {len([f for f in output_performance if f in FEATURE_COMPUTERS])} computed"
        )
        print(f"\n   Output: {output_dir}")
        print(f"{'='*80}\n")

    comm.Barrier()

    # Process samples assigned to this rank
    total_errors = {
        "image_load_failed": 0,
        "image_compress_failed": 0,
        "data_extraction_failed": 0,
    }

    files_created = 0
    total_rows = 0

    for sample_id in tqdm(
        my_samples, desc=f"Rank {rank}", position=rank, disable=(rank != 0)
    ):
        # Process sample
        df_sample, error_stats = process_single_sample(
            sample_id,
            tau_df,
            parquet_dir,
            parquet_pattern,
            images_dir,
            image_pattern,
            config,
        )

        # Aggregate errors
        if error_stats["image_load_failed"]:
            total_errors["image_load_failed"] += 1
        if error_stats["image_compress_failed"]:
            total_errors["image_compress_failed"] += 1
        total_errors["data_extraction_failed"] += error_stats["data_extraction_failed"]

        # Save if successful
        if len(df_sample) > 0:
            output_file = output_dir / f"sample_{sample_id:05d}.parquet"
            df_sample.to_parquet(output_file, compression="snappy", index=False)
            files_created += 1
            total_rows += len(df_sample)

    comm.Barrier()

    # Gather stats
    all_files_created = comm.gather(files_created, root=0)
    all_rows_created = comm.gather(total_rows, root=0)
    all_errors = comm.gather(total_errors, root=0)

    if rank == 0:
        total_files = sum(all_files_created)
        total_rows_all = sum(all_rows_created)
        expected_files = len(sample_ids)
        expected_rows = len(sample_ids) * num_params

        # Aggregate errors
        agg_errors = {}
        for err_dict in all_errors:
            for key, val in err_dict.items():
                agg_errors[key] = agg_errors.get(key, 0) + val

        # Calculate total size
        parquet_files = list(output_dir.glob("sample_*.parquet"))
        total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)

        print(f"\n{'='*80}")
        print(f"✅ Intermediate Dataset Created!")
        print(f"{'='*80}")
        print(f"   Expected files: {expected_files}")
        print(f"   Created files: {total_files}")
        print(f"   Missing files: {expected_files - total_files}")
        print(f"\n   Expected rows: {expected_rows:,}")
        print(f"   Created rows: {total_rows_all:,}")
        print(f"   Missing rows: {expected_rows - total_rows_all:,}")

        if expected_files > 0:
            print(f"\n   File success rate: {total_files/expected_files*100:.2f}%")
        if expected_rows > 0:
            print(f"   Row success rate: {total_rows_all/expected_rows*100:.2f}%")

        print(f"\n   Total size: {total_size_mb:.1f} MB")
        print(f"   Avg per file: {total_size_mb/max(total_files, 1):.2f} MB")

        if sum(agg_errors.values()) > 0:
            print(f"\n   ⚠️  Error summary:")
            for error_type, count in agg_errors.items():
                if count > 0:
                    print(f"      {error_type}: {count:,}")

        print(f"\n   Location: {output_dir}")
        print(f"   Pattern: sample_XXXXX.parquet")
        print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================


def main():
    if rank == 0:
        print("=" * 80)
        print("📦 Step 1: Create Intermediate Dataset (ONE FILE PER SAMPLE)")
        print("=" * 80)
        print(f"🚀 Running with {world_size} MPI processes")
        print(
            f"📊 Constants: MAX_CYCLES={MAX_CYCLES}, CURRENT_THRESHOLD={CURRENT_THRESHOLD}A"
        )
        print("=" * 80)

    # Load config
    config, paths = load_config()

    if rank == 0:
        print(f"\n✓ Configuration loaded")
        print("📂 Loading CSV metadata...")

    base_dir = Path(paths["data"]["base_dir"])

    # Load tau results
    tau_results_path = base_dir / paths["data"]["input"]["tau_results_csv"]
    if not tau_results_path.exists():
        if rank == 0:
            print(f"❌ ERROR: Tau results CSV not found: {tau_results_path}")
        sys.exit(1)
    tau_results = pd.read_csv(tau_results_path)

    if rank == 0:
        print(f"✓ Loaded {len(tau_results)} tau results from: {tau_results_path.name}")

    # Discover samples
    samples = discover_samples(tau_results, paths)

    if rank == 0:
        if len(samples) > 0:
            print(f"\n✓ Final sample count: {len(samples)}")
            print(f"✓ Sample IDs: {samples[:10]}{'...' if len(samples) > 10 else ''}")
        else:
            print(f"\n❌ No valid samples found!")
            sys.exit(1)

    # Create intermediate dataset
    create_intermediate_parquet(samples, config, paths, tau_results, rank, world_size)

    if rank == 0:
        print("=" * 80)
        print("🎉 Intermediate dataset creation complete!")
        print("=" * 80)


if __name__ == "__main__":
    main()
