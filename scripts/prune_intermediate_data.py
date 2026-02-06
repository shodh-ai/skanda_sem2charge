#!/usr/bin/env python3
"""
Step 2: Prune intermediate dataset by removing:
1. Rows with NaN values
2. Rows violating physical constraints
3. Statistical outliers (IQR method)

Input: data/intermediate/sample_*.parquet files
Output: data/intermediate_pruned.parquet

Usage: mpirun -n 8 python scripts/prune_intermediate_data.py
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# ============================================================================
# MPI SETUP
# ============================================================================

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    MPI_AVAILABLE = True
except ImportError:
    comm = None
    rank = 0
    world_size = 1
    MPI_AVAILABLE = False
    print("⚠️  Warning: mpi4py not available. Running in serial mode.")


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

INPUT_FEATURES = [
    "input_SEI kinetic rate constant [m.s-1]",
    "input_Electrolyte diffusivity [m2.s-1]",
    "input_Initial concentration in electrolyte [mol.m-3]",
    "input_Separator porosity",
    "input_Positive particle radius [m]",
    "input_Negative particle radius [m]",
    "input_Positive electrode thickness [m]",
    "input_Negative electrode thickness [m]",
    "input_Outer SEI solvent diffusivity [m2.s-1]",
    "input_Dead lithium decay constant [s-1]",
    "input_Lithium plating kinetic rate constant [m.s-1]",
    "input_Negative electrode LAM constant proportional term [s-1]",
    "input_Negative electrode cracking rate",
    "input_Outer SEI partial molar volume [m3.mol-1]",
    "input_SEI growth activation energy [J.mol-1]",
]

MICROSTRUCTURE_FEATURES = [
    "D_eff",
    "porosity_measured",
    "tau_factor",
    "bruggeman_derived",
]

PERFORMANCE_FEATURES = [
    "projected_cycle_life",
    "capacity_fade_rate",
    "internal_resistance",
    "nominal_capacity",
    "energy_density",
]

ALL_PARAMETER_FEATURES = INPUT_FEATURES + MICROSTRUCTURE_FEATURES + PERFORMANCE_FEATURES


# ============================================================================
# PHYSICAL CONSTRAINT LIMITS
# ============================================================================

PHYSICAL_LIMITS = {
    # Input features
    "input_SEI kinetic rate constant [m.s-1]": (0, np.inf),
    "input_Electrolyte diffusivity [m2.s-1]": (0, np.inf),
    "input_Initial concentration in electrolyte [mol.m-3]": (0, np.inf),
    "input_Separator porosity": (0, 1),
    "input_Positive particle radius [m]": (0, np.inf),
    "input_Negative particle radius [m]": (0, np.inf),
    "input_Positive electrode thickness [m]": (0, np.inf),
    "input_Negative electrode thickness [m]": (0, np.inf),
    "input_Outer SEI solvent diffusivity [m2.s-1]": (0, np.inf),
    "input_Dead lithium decay constant [s-1]": (0, np.inf),
    "input_Lithium plating kinetic rate constant [m.s-1]": (0, np.inf),
    "input_Negative electrode LAM constant proportional term [s-1]": (0, np.inf),
    "input_Negative electrode cracking rate": (0, np.inf),
    "input_Outer SEI partial molar volume [m3.mol-1]": (0, np.inf),
    "input_SEI growth activation energy [J.mol-1]": (0, np.inf),
    # Microstructure features
    "D_eff": (0, np.inf),
    "porosity_measured": (0, 1),
    "tau_factor": (1, 8),
    "bruggeman_derived": (0, np.inf),
    # Performance features
    "projected_cycle_life": (0, 5000),
    "capacity_fade_rate": (0, np.inf),
    "internal_resistance": (0.001, 1.0),
    "nominal_capacity": (0, np.inf),
    "energy_density": (2.0, 5.0),
}


# ============================================================================
# CONFIGURATION
# ============================================================================


def load_config() -> Tuple[Dict, Dict]:
    """Load configuration files"""
    config_dir = Path.cwd() / "configs"

    with open(config_dir / "optimise_config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open(config_dir / "paths.yml", "r") as f:
        paths = yaml.safe_load(f)

    return config, paths


# ============================================================================
# DATA LOADING
# ============================================================================


def load_intermediate_files(
    intermediate_dir: Path, rank: int, world_size: int
) -> pd.DataFrame:
    """Load intermediate parquet files in parallel"""

    parquet_files = sorted(intermediate_dir.glob("sample_*.parquet"))

    if len(parquet_files) == 0:
        raise FileNotFoundError(
            f"No sample_*.parquet files found in {intermediate_dir}"
        )

    # Distribute files among ranks
    my_files = [f for i, f in enumerate(parquet_files) if i % world_size == rank]

    if rank == 0:
        print(f"\n📂 Loading intermediate data...")
        print(f"   Found {len(parquet_files)} parquet files")
        print(f"   MPI processes: {world_size}")
        print(f"   Files per rank: ~{len(parquet_files) // world_size}")

    # Load files assigned to this rank
    dfs = []
    for pf in tqdm(
        my_files, desc=f"Rank {rank} loading", position=rank, disable=(rank != 0)
    ):
        df = pd.read_parquet(pf)
        dfs.append(df)

    if dfs:
        df_local = pd.concat(dfs, ignore_index=True)
    else:
        df_local = pd.DataFrame()

    if rank == 0:
        print(f"\n   Rank {rank} loaded {len(df_local):,} rows")

    return df_local


def expand_dataframe(df_local: pd.DataFrame, rank: int) -> pd.DataFrame:
    """Expand nested arrays into separate columns"""

    if rank == 0:
        print(f"\n🔧 Expanding nested data structures...")

    if len(df_local) == 0:
        # Return empty DataFrame with correct columns
        all_cols = (
            ["sample_id", "param_id"]
            + INPUT_FEATURES
            + MICROSTRUCTURE_FEATURES
            + PERFORMANCE_FEATURES
        )
        return pd.DataFrame(columns=all_cols)

    # Extract arrays
    input_arrays = np.array(df_local["input_params"].tolist())
    micro_arrays = np.array(df_local["microstructure_outputs"].tolist())
    perf_arrays = np.array(df_local["performance_outputs"].tolist())

    # Create DataFrames
    df_inputs = pd.DataFrame(input_arrays, columns=INPUT_FEATURES)
    df_micro = pd.DataFrame(micro_arrays, columns=MICROSTRUCTURE_FEATURES)
    df_perf = pd.DataFrame(perf_arrays, columns=PERFORMANCE_FEATURES)

    # Combine (exclude image data to save memory)
    df_expanded = pd.concat(
        [
            df_local[["sample_id", "param_id"]].reset_index(drop=True),
            df_inputs,
            df_micro,
            df_perf,
        ],
        axis=1,
    )

    if rank == 0:
        print(f"   Expanded shape: {df_expanded.shape}")
        print(
            f"   Features: {len(INPUT_FEATURES)} inputs + {len(MICROSTRUCTURE_FEATURES)} micro + {len(PERFORMANCE_FEATURES)} perf = {len(ALL_PARAMETER_FEATURES)} total"
        )

    return df_expanded


# ============================================================================
# PRUNING STEP 1: NaN REMOVAL
# ============================================================================


def remove_nans(df: pd.DataFrame, rank: int) -> Tuple[pd.DataFrame, Dict]:
    """Remove rows with any NaN values"""

    initial_count = len(df)

    if initial_count == 0:
        return df, {"initial": 0, "final": 0, "removed": 0}

    # Check for NaNs in all parameter columns
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"STEP 1: NaN REMOVAL")
        print(f"{'='*80}")

        nan_counts = df[ALL_PARAMETER_FEATURES].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

        if len(nan_cols) > 0:
            print(f"\n   Columns with NaN values:")
            for col, count in nan_cols.items():
                pct = count / len(df) * 100
                print(f"     {col:60s}: {count:6,} ({pct:5.2f}%)")

    # Remove NaNs
    df_clean = df.dropna(subset=ALL_PARAMETER_FEATURES)

    final_count = len(df_clean)
    removed_count = initial_count - final_count

    stats = {
        "initial": initial_count,
        "final": final_count,
        "removed": removed_count,
    }

    if rank == 0:
        print(f"\n   Initial samples: {initial_count:,}")
        print(f"   After NaN removal: {final_count:,}")
        print(f"   Removed: {removed_count:,} ({removed_count/initial_count*100:.2f}%)")

    return df_clean, stats


# ============================================================================
# PRUNING STEP 2: PHYSICAL CONSTRAINT VALIDATION
# ============================================================================


def validate_physical_constraints(
    df: pd.DataFrame, rank: int
) -> Tuple[pd.DataFrame, Dict]:
    """Remove rows that violate physical constraints"""

    initial_count = len(df)

    if initial_count == 0:
        return df, {"initial": 0, "final": 0, "removed": 0, "violations": {}}

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"STEP 2: PHYSICAL CONSTRAINT VALIDATION")
        print(f"{'='*80}")

    violations = {}
    valid_mask = pd.Series(True, index=df.index)

    for feature, (lower, upper) in PHYSICAL_LIMITS.items():
        if feature not in df.columns:
            continue

        # Check lower bound (exclusive if > 0)
        if lower == 0:
            lower_violations = df[feature] <= lower
        else:
            lower_violations = df[feature] < lower

        # Check upper bound
        if np.isinf(upper):
            upper_violations = pd.Series(False, index=df.index)
        else:
            upper_violations = df[feature] > upper

        feature_violations = lower_violations | upper_violations
        violation_count = feature_violations.sum()

        if violation_count > 0:
            violations[feature] = violation_count
            valid_mask &= ~feature_violations

    df_valid = df[valid_mask].copy()

    final_count = len(df_valid)
    removed_count = initial_count - final_count

    stats = {
        "initial": initial_count,
        "final": final_count,
        "removed": removed_count,
        "violations": violations,
    }

    if rank == 0:
        print(f"\n   Initial samples: {initial_count:,}")
        print(f"   After constraint validation: {final_count:,}")
        print(f"   Removed: {removed_count:,} ({removed_count/initial_count*100:.2f}%)")

        if violations:
            print(f"\n   Top constraint violations:")
            sorted_violations = sorted(
                violations.items(), key=lambda x: x[1], reverse=True
            )
            for feat, count in sorted_violations[:10]:
                pct = count / initial_count * 100
                limits = PHYSICAL_LIMITS[feat]
                print(f"     {feat:60s}: {count:6,} ({pct:5.2f}%) | Limits: {limits}")

    return df_valid, stats


# ============================================================================
# PRUNING STEP 3: OUTLIER REMOVAL
# ============================================================================


def detect_outliers_iqr(
    df: pd.DataFrame, features: List[str], multiplier: float = 1.5
) -> Tuple[pd.Series, Dict, Dict]:
    """Detect outliers using IQR method"""

    outlier_mask = pd.Series(False, index=df.index)
    outlier_counts = {}
    outlier_bounds = {}

    for feat in features:
        if feat not in df.columns:
            continue

        data = df[feat].dropna()

        if len(data) == 0:
            continue

        # Calculate IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        # Skip if IQR is too small (likely constant)
        if IQR < 1e-10:
            continue

        # Define outlier bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outlier_bounds[feat] = {
            "lower": lower_bound,
            "upper": upper_bound,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
        }

        # Mark outliers
        feat_outliers = (df[feat] < lower_bound) | (df[feat] > upper_bound)
        outlier_counts[feat] = feat_outliers.sum()

        # Update overall mask (OR operation)
        outlier_mask = outlier_mask | feat_outliers

    return outlier_mask, outlier_counts, outlier_bounds


def remove_outliers(
    df: pd.DataFrame, rank: int, iqr_multiplier: float = 1.5
) -> Tuple[pd.DataFrame, Dict]:
    """Remove statistical outliers using IQR method"""

    initial_count = len(df)

    if initial_count == 0:
        return df, {"initial": 0, "final": 0, "removed": 0}

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"STEP 3: OUTLIER REMOVAL (IQR Method)")
        print(f"{'='*80}")
        print(f"   IQR multiplier: {iqr_multiplier}")

    # Detect outliers
    outlier_mask, outlier_counts, outlier_bounds = detect_outliers_iqr(
        df, ALL_PARAMETER_FEATURES, iqr_multiplier
    )

    # Remove outliers
    df_clean = df[~outlier_mask].copy()

    final_count = len(df_clean)
    removed_count = initial_count - final_count

    stats = {
        "initial": initial_count,
        "final": final_count,
        "removed": removed_count,
        "outlier_counts": outlier_counts,
        "outlier_bounds": outlier_bounds,
    }

    if rank == 0:
        print(f"\n   Initial samples: {initial_count:,}")
        print(f"   After outlier removal: {final_count:,}")
        print(f"   Removed: {removed_count:,} ({removed_count/initial_count*100:.2f}%)")

        if outlier_counts:
            sorted_outliers = sorted(
                outlier_counts.items(), key=lambda x: x[1], reverse=True
            )
            print(f"\n   Top features with outliers:")
            for feat, count in sorted_outliers[:10]:
                if count > 0:
                    pct = count / initial_count * 100
                    bounds = outlier_bounds[feat]
                    print(f"     {feat:60s}: {count:6,} ({pct:5.2f}%)")
                    print(
                        f"       Bounds: [{bounds['lower']:.6e}, {bounds['upper']:.6e}]"
                    )

    return df_clean, stats


# ============================================================================
# STATISTICS GATHERING
# ============================================================================


def gather_and_print_summary(
    nan_stats: Dict,
    constraint_stats: Dict,
    outlier_stats: Dict,
    comm,
    rank: int,
    world_size: int,
):
    """Gather statistics from all ranks and print summary"""

    if comm is not None:
        all_nan_stats = comm.gather(nan_stats, root=0)
        all_constraint_stats = comm.gather(constraint_stats, root=0)
        all_outlier_stats = comm.gather(outlier_stats, root=0)
    else:
        all_nan_stats = [nan_stats]
        all_constraint_stats = [constraint_stats]
        all_outlier_stats = [outlier_stats]

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE PRUNING SUMMARY (ALL RANKS)")
        print(f"{'='*80}")

        # Aggregate NaN stats
        nan_initial = sum(s.get("initial", 0) for s in all_nan_stats)
        nan_final = sum(s.get("final", 0) for s in all_nan_stats)
        nan_removed = sum(s.get("removed", 0) for s in all_nan_stats)

        print(f"\nSTEP 1: NaN Removal")
        print(f"   Initial: {nan_initial:,}")
        print(f"   Final: {nan_final:,}")
        print(
            f"   Removed: {nan_removed:,} ({nan_removed/max(nan_initial,1)*100:.2f}%)"
        )

        # Aggregate constraint stats
        const_initial = sum(s.get("initial", 0) for s in all_constraint_stats)
        const_final = sum(s.get("final", 0) for s in all_constraint_stats)
        const_removed = sum(s.get("removed", 0) for s in all_constraint_stats)

        print(f"\nSTEP 2: Physical Constraint Validation")
        print(f"   Initial: {const_initial:,}")
        print(f"   Final: {const_final:,}")
        print(
            f"   Removed: {const_removed:,} ({const_removed/max(const_initial,1)*100:.2f}%)"
        )

        # Aggregate outlier stats
        outlier_initial = sum(s.get("initial", 0) for s in all_outlier_stats)
        outlier_final = sum(s.get("final", 0) for s in all_outlier_stats)
        outlier_removed = sum(s.get("removed", 0) for s in all_outlier_stats)

        print(f"\nSTEP 3: Outlier Removal")
        print(f"   Initial: {outlier_initial:,}")
        print(f"   Final: {outlier_final:,}")
        print(
            f"   Removed: {outlier_removed:,} ({outlier_removed/max(outlier_initial,1)*100:.2f}%)"
        )

        # Overall summary
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"   Starting samples: {nan_initial:,}")
        print(f"   Final clean samples: {outlier_final:,}")
        print(f"   Total removed: {nan_initial - outlier_final:,}")
        if nan_initial > 0:
            print(f"   Overall retention: {outlier_final/nan_initial*100:.2f}%")
        print(f"{'='*80}")


# ============================================================================
# DATA SAVING
# ============================================================================


def gather_and_save_pruned_data(
    df_pruned_local: pd.DataFrame,
    df_original_local: pd.DataFrame,
    output_file: Path,
    comm,
    rank: int,
):
    """
    Save pruned data as a partitioned Parquet dataset.
    Avoids MPI gather crashes by letting every rank write its own file.
    """

    # 1. Filter local data to match pruned indices
    df_export_local = df_original_local[
        df_original_local.index.isin(df_pruned_local.index)
    ].copy()

    # 2. Prepare Output Directory
    # We treat 'intermediate_pruned.parquet' as a FOLDER, not a file.
    if rank == 0:
        # If a single file exists with this name, delete it to create a folder
        if output_file.exists() and output_file.is_file():
            try:
                output_file.unlink()
                print(f"   Removed existing single file to create dataset directory.")
            except OSError:
                pass

        output_file.mkdir(parents=True, exist_ok=True)

    # Wait for Rank 0 to create the directory
    if comm is not None:
        comm.Barrier()

    # 3. Write Local Partition
    if len(df_export_local) > 0:
        # Create a unique filename for this rank: part_00001.parquet
        part_path = output_file / f"part_{rank:05d}.parquet"
        df_export_local.reset_index(drop=True).to_parquet(
            part_path, compression="snappy", index=False
        )

    # 4. Final Sync
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        # Count files to ensure success
        num_files = len(list(output_file.glob("part_*.parquet")))
        print(f"\n✅ PRUNED DATA SAVED SUCCESSFULLY (Partitioned)!")
        print(f"{'─'*80}")
        print(f"   Location: {output_file}")
        print(f"   Partitions created: {num_files}")
        print(f"   (Pandas will read this folder automatically as a single dataset)")
        print(f"{'─'*80}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    if rank == 0:
        print("=" * 80)
        print("PRUNING INTERMEDIATE DATASET")
        print("=" * 80)
        print(f"MPI Processes: {world_size}")
        print(f"MPI Available: {MPI_AVAILABLE}")
        print("=" * 80)

    # Load config
    config, paths = load_config()

    intermediate_dir = Path(paths["data"]["output"]["intermediate_dir"])
    output_file = Path("data") / "intermediate_pruned"

    if rank == 0:
        print(f"\n   Input: {intermediate_dir}")
        print(f"   Output: {output_file}")

    # Load data
    df_local = load_intermediate_files(intermediate_dir, rank, world_size)

    # Expand nested structures
    df_expanded = expand_dataframe(df_local, rank)

    # PRUNING PIPELINE
    df_after_nan, nan_stats = remove_nans(df_expanded, rank)
    df_after_constraints, constraint_stats = validate_physical_constraints(
        df_after_nan, rank
    )
    df_final, outlier_stats = remove_outliers(
        df_after_constraints, rank, iqr_multiplier=1.5
    )

    # Gather and print summary
    gather_and_print_summary(
        nan_stats, constraint_stats, outlier_stats, comm, rank, world_size
    )

    # Save pruned data
    if rank == 0:
        print(f"\n📦 Gathering pruned data from all ranks...")

    gather_and_save_pruned_data(df_final, df_local, output_file, comm, rank)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"🎉 PRUNING COMPLETE!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
