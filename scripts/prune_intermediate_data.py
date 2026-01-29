#!/usr/bin/env python3
"""
MPI-enabled script to prune intermediate data by:
1. Removing rows where tau_factor > 8 (physical constraint)
2. Removing rows with any NaN values (data quality)
3. Removing statistical outliers using IQR method (data cleanup)
4. Saving to: data/intermediate_pruned.parquet

Usage:
    Single process: python scripts/prune_intermediate_data.py
    MPI parallel:   mpirun -n 4 python scripts/prune_intermediate_data.py
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running in serial mode.")
    print("Install with: pip install mpi4py")


# ============================================================================
# FEATURE DEFINITIONS (MUST MATCH create_intermediate_dataset.py - 17 total)
# ============================================================================

INPUT_FEATURES = [
    # Original 10 features (with exact names from parquet)
    "input_SEI kinetic rate constant [m.s-1]",
    "input_Electrolyte diffusivity [m2.s-1]",
    "input_Initial concentration in electrolyte [mol.m-3]",
    "input_Separator porosity",
    "input_Positive particle radius [m]",
    "input_Negative particle radius [m]",
    "input_Positive electrode thickness [m]",
    "input_Negative electrode thickness [m]",
    # New 7 degradation features (with exact names from parquet)
    "input_Outer SEI solvent diffusivity [m2.s-1]",
    "input_Dead lithium decay constant [s-1]",
    "input_Lithium plating kinetic rate constant [m.s-1]",
    "input_Negative electrode LAM constant proportional term [s-1]",
    "input_Negative electrode cracking rate",
    "input_Outer SEI partial molar volume [m3.mol-1]",
    "input_SEI growth activation energy [J.mol-1]",
]

# Microstructure features from tau_results CSV + bruggeman from parquet
MICROSTRUCTURE_FEATURES = [
    "D_eff",
    "porosity_measured",
    "tau_factor",
    "bruggeman_derived",
]

# Performance features from parquet
PERFORMANCE_FEATURES = [
    "nominal_capacity_Ah",
    "eol_cycle_measured",
    "initial_capacity_Ah",
    "final_capacity_Ah",
    "capacity_retention_percent",
    "total_cycles",
    "final_RUL",
]


# ============================================================================
# PRUNING CONFIGURATION
# ============================================================================

# Physical constraint threshold
TAU_FACTOR_MAX = 8.0

# Outlier removal configuration (IQR method)
OUTLIER_CONFIG = {
    "enabled": True,
    "method": "IQR",  # IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    "iqr_multiplier": 1.5,  # Standard IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)
    # Features to check for outliers (exclude constant features)
    "features_to_check": [
        # Input features (check all that vary - exclude constants if any)
        "input_SEI kinetic rate constant [m.s-1]",
        "input_Electrolyte diffusivity [m2.s-1]",
        # "input_Initial concentration in electrolyte [mol.m-3]",  # Often constant
        "input_Separator porosity",
        # Skip the Bruggeman coefficients if they're constant or causing issues
        "input_Positive particle radius [m]",
        "input_Negative particle radius [m]",
        "input_Positive electrode thickness [m]",
        "input_Negative electrode thickness [m]",
        # New degradation features
        "input_Outer SEI solvent diffusivity [m2.s-1]",
        "input_Dead lithium decay constant [s-1]",
        "input_Lithium plating kinetic rate constant [m.s-1]",
        "input_Negative electrode LAM constant proportional term [s-1]",
        "input_Negative electrode cracking rate",
        "input_Outer SEI partial molar volume [m3.mol-1]",
        "input_SEI growth activation energy [J.mol-1]",
        # Microstructure features (all)
        "D_eff",
        "porosity_measured",
        "tau_factor",
        "bruggeman_derived",
        # Performance features (key ones)
        "initial_capacity_Ah",
        "final_capacity_Ah",
        "capacity_retention_percent",
        "total_cycles",
    ],
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_config():
    """Load configuration from YAML files"""
    with open("./configs/paths.yml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("./configs/optimise_config.yml", "r") as f:
        opt_config = yaml.safe_load(f)

    return paths_config, opt_config


def get_mpi_info():
    """Get MPI communicator, rank, and size"""
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    return comm, rank, size


# ============================================================================
# DATA LOADING
# ============================================================================


def load_intermediate_data_parallel(intermediate_dir, rank, size):
    """Load intermediate parquet files in parallel"""
    intermediate_path = Path(intermediate_dir)
    parquet_files = sorted(intermediate_path.glob("data_rank*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {intermediate_path}")

    # Distribute files among ranks
    my_files = [f for i, f in enumerate(parquet_files) if i % size == rank]

    if rank == 0:
        print(f"Found {len(parquet_files)} parquet files in {intermediate_path}")
        print(f"Running with {size} MPI processes")
        print(f"Each process will handle ~{len(parquet_files)//size + 1} files")

    # Load files assigned to this rank
    dfs = []
    for pf in my_files:
        df = pd.read_parquet(pf)
        dfs.append(df)
        if rank == 0 or len(my_files) <= 3:
            print(f"  Rank {rank}: {pf.name} - {len(df)} rows")

    if dfs:
        df_local = pd.concat(dfs, ignore_index=True)
        if rank == 0:
            print(f"\nRank {rank} loaded {len(df_local)} rows")
    else:
        df_local = pd.DataFrame()

    return df_local


def expand_dataframe(df_local, rank):
    """Expand nested arrays into separate columns"""
    if rank == 0:
        print("\nExpanding nested data structures...")

    if len(df_local) == 0:
        all_cols = (
            ["sample_id", "param_id"]
            + INPUT_FEATURES
            + MICROSTRUCTURE_FEATURES
            + PERFORMANCE_FEATURES
        )
        return pd.DataFrame(columns=all_cols)

    # Extract arrays
    input_arrays = np.array(df_local["input_params"].tolist())

    # Check if we have the right number of features
    if rank == 0:
        print(f"  Input array shape: {input_arrays.shape}")
        print(f"  Expected features: {len(INPUT_FEATURES)}")
        if input_arrays.shape[1] != len(INPUT_FEATURES):
            print(
                f"  ⚠️  WARNING: Mismatch! Array has {input_arrays.shape[1]} features but we have {len(INPUT_FEATURES)} names"
            )

    df_inputs = pd.DataFrame(input_arrays, columns=INPUT_FEATURES)

    micro_arrays = np.array(df_local["microstructure_outputs"].tolist())
    df_micro = pd.DataFrame(micro_arrays, columns=MICROSTRUCTURE_FEATURES)

    perf_arrays = np.array(df_local["performance_outputs"].tolist())
    df_perf = pd.DataFrame(perf_arrays, columns=PERFORMANCE_FEATURES)

    # Combine
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
        print(f"Rank {rank} expanded dataset shape: {df_expanded.shape}")
        print(f"  Input features: {len(INPUT_FEATURES)}")
        print(f"  Microstructure features: {len(MICROSTRUCTURE_FEATURES)}")
        print(f"  Performance features: {len(PERFORMANCE_FEATURES)}")
        print(
            f"  Total features: {len(INPUT_FEATURES) + len(MICROSTRUCTURE_FEATURES) + len(PERFORMANCE_FEATURES)}"
        )

    return df_expanded


# ============================================================================
# PRUNING STEP 1: TAU FACTOR FILTERING
# ============================================================================


def filter_tau_factor(df_expanded, rank, tau_max=TAU_FACTOR_MAX):
    """
    Step 1: Remove rows where tau_factor > tau_max (physical constraint)

    Args:
        df_expanded: Expanded DataFrame
        rank: MPI rank
        tau_max: Maximum allowed tau_factor value

    Returns:
        Filtered DataFrame and statistics
    """
    initial_count = len(df_expanded)

    if initial_count == 0:
        return df_expanded, {"initial": 0, "final": 0, "removed": 0}

    # Filter by tau_factor
    df_filtered = df_expanded[df_expanded["tau_factor"] <= tau_max].copy()
    final_count = len(df_filtered)
    removed_count = initial_count - final_count

    stats = {"initial": initial_count, "final": final_count, "removed": removed_count}

    if rank == 0:
        print(f"\nRank {rank} - Tau Factor Filtering:")
        print(f"  Initial samples:     {initial_count:,}")
        print(f"  After filtering:     {final_count:,}")
        print(
            f"  Removed:             {removed_count:,} ({removed_count/initial_count*100:.2f}%)"
        )

    return df_filtered, stats


# ============================================================================
# PRUNING STEP 2: NaN REMOVAL
# ============================================================================


def remove_nans(df, rank):
    """
    Step 2: Remove rows with any NaN values (data quality)

    Args:
        df: DataFrame to clean
        rank: MPI rank

    Returns:
        Cleaned DataFrame and statistics
    """
    initial_count = len(df)

    if initial_count == 0:
        return df, {"initial": 0, "final": 0, "removed": 0}

    # Check for NaNs in all parameter columns
    all_param_cols = INPUT_FEATURES + MICROSTRUCTURE_FEATURES + PERFORMANCE_FEATURES

    # Count NaNs per column for reporting
    if rank == 0:
        nan_counts = df[all_param_cols].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if len(nan_cols) > 0:
            print(f"\n  Columns with NaN values:")
            for col, count in nan_cols.items():
                pct = count / len(df) * 100
                print(f"    {col:70s}: {count:6,} NaNs ({pct:5.1f}%)")

    df_clean = df.dropna(subset=all_param_cols)
    final_count = len(df_clean)
    removed_count = initial_count - final_count

    stats = {"initial": initial_count, "final": final_count, "removed": removed_count}

    if rank == 0:
        print(f"\nRank {rank} - NaN Removal Summary:")
        print(f"  Initial samples:     {initial_count:,}")
        print(f"  After NaN removal:   {final_count:,}")
        print(
            f"  Removed:             {removed_count:,} ({removed_count/initial_count*100:.2f}%)"
        )

    return df_clean, stats


# ============================================================================
# PRUNING STEP 3: OUTLIER REMOVAL
# ============================================================================


def detect_outliers_iqr(df, features, multiplier=1.5):
    """
    Detect outliers using IQR method

    Args:
        df: DataFrame with expanded features
        features: List of feature names to check
        multiplier: IQR multiplier (1.5 for standard outliers, 3.0 for extreme)

    Returns:
        Boolean mask where True indicates outlier, and outlier counts per feature
    """
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

        # Define outlier bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Store bounds for reporting
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

        # Update overall mask (OR operation - any feature is outlier)
        outlier_mask = outlier_mask | feat_outliers

    return outlier_mask, outlier_counts, outlier_bounds


def remove_outliers(df, rank, config=OUTLIER_CONFIG):
    """
    Step 3: Remove statistical outliers (data cleanup)

    Args:
        df: DataFrame to clean
        rank: MPI rank
        config: Outlier removal configuration

    Returns:
        Cleaned DataFrame and detailed statistics
    """
    if not config["enabled"] or len(df) == 0:
        return df, {}

    initial_count = len(df)

    # Detect outliers using IQR method
    outlier_mask, outlier_counts, outlier_bounds = detect_outliers_iqr(
        df, config["features_to_check"], config["iqr_multiplier"]
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
        print(f"\nRank {rank} - Outlier Removal:")
        print(f"  Initial samples:     {initial_count:,}")
        print(f"  After outlier removal: {final_count:,}")
        print(
            f"  Removed:             {removed_count:,} ({removed_count/initial_count*100:.2f}%)"
        )

        # Show top features with outliers
        if outlier_counts:
            sorted_outliers = sorted(
                outlier_counts.items(), key=lambda x: x[1], reverse=True
            )
            print(f"\n  Top 10 features with outliers:")
            for feat, count in sorted_outliers[:10]:
                if count > 0:
                    pct = count / initial_count * 100
                    bounds = outlier_bounds[feat]
                    print(
                        f"    {feat:60s}: {count:6,} ({pct:5.2f}%) | Bounds: [{bounds['lower']:.6e}, {bounds['upper']:.6e}]"
                    )

    return df_clean, stats


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================


def gather_and_print_stats(
    tau_stats, nan_stats, outlier_stats, comm, rank, size, tau_max
):
    """Gather statistics from all ranks and print comprehensive summary"""
    if comm is not None:
        all_tau_stats = comm.gather(tau_stats, root=0)
        all_nan_stats = comm.gather(nan_stats, root=0)
        all_outlier_stats = comm.gather(outlier_stats, root=0)
    else:
        all_tau_stats = [tau_stats]
        all_nan_stats = [nan_stats]
        all_outlier_stats = [outlier_stats]

    if rank == 0:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PRUNING SUMMARY (ALL RANKS AGGREGATED)")
        print("=" * 80)

        # Calculate totals for each step

        # Step 1: Tau Factor Filtering
        tau_initial = sum(s.get("initial", 0) for s in all_tau_stats)
        tau_final = sum(s.get("final", 0) for s in all_tau_stats)
        tau_removed = sum(s.get("removed", 0) for s in all_tau_stats)

        print(f"\nSTEP 1: TAU FACTOR FILTERING (Physical Constraint)")
        print(f"{'─'*80}")
        print(f"  Constraint: tau_factor <= {tau_max}")
        print(f"  Initial samples:        {tau_initial:,}")
        print(f"  Samples passing filter: {tau_final:,}")
        print(f"  Samples removed:        {tau_removed:,}")
        if tau_initial > 0:
            print(f"  Retention rate:         {tau_final/tau_initial*100:.2f}%")

        # Step 2: NaN Removal
        nan_initial = sum(s.get("initial", 0) for s in all_nan_stats)
        nan_final = sum(s.get("final", 0) for s in all_nan_stats)
        nan_removed = sum(s.get("removed", 0) for s in all_nan_stats)

        print(f"\nSTEP 2: NaN REMOVAL (Data Quality)")
        print(f"{'─'*80}")
        print(f"  Initial samples:        {nan_initial:,}")
        print(f"  Samples without NaNs:   {nan_final:,}")
        print(f"  Samples removed:        {nan_removed:,}")
        if nan_initial > 0:
            print(f"  Retention rate:         {nan_final/nan_initial*100:.2f}%")

        # Step 3: Outlier Removal
        if OUTLIER_CONFIG["enabled"] and all_outlier_stats[0]:
            outlier_initial = sum(s.get("initial", 0) for s in all_outlier_stats)
            outlier_final = sum(s.get("final", 0) for s in all_outlier_stats)
            outlier_removed = sum(s.get("removed", 0) for s in all_outlier_stats)

            print(f"\nSTEP 3: OUTLIER REMOVAL (Statistical Cleanup)")
            print(f"{'─'*80}")
            print(f"  Method: IQR with multiplier {OUTLIER_CONFIG['iqr_multiplier']}")
            print(f"  Initial samples:        {outlier_initial:,}")
            print(f"  Samples without outliers: {outlier_final:,}")
            print(f"  Samples removed:        {outlier_removed:,}")
            if outlier_initial > 0:
                print(
                    f"  Retention rate:         {outlier_final/outlier_initial*100:.2f}%"
                )

            # Aggregate outlier counts across all ranks
            total_outlier_counts = {}
            for stats in all_outlier_stats:
                if "outlier_counts" in stats:
                    for feat, count in stats["outlier_counts"].items():
                        total_outlier_counts[feat] = (
                            total_outlier_counts.get(feat, 0) + count
                        )

            if total_outlier_counts:
                sorted_outliers = sorted(
                    total_outlier_counts.items(), key=lambda x: x[1], reverse=True
                )
                print(f"\n  Top features contributing to outliers:")
                for feat, count in sorted_outliers[:10]:
                    if count > 0:
                        pct = (
                            count / outlier_initial * 100 if outlier_initial > 0 else 0
                        )
                        print(f"    {feat:60s}: {count:6,} ({pct:5.2f}%)")

            final_total = outlier_final
        else:
            final_total = nan_final

        # Overall Summary
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"  Initial samples:        {tau_initial:,}")
        print(f"  Final clean samples:    {final_total:,}")
        print(f"  Total samples removed:  {tau_initial - final_total:,}")
        if tau_initial > 0:
            print(f"  Overall retention:      {final_total/tau_initial*100:.2f}%")
        print(f"\n  Breakdown of removed samples:")
        if tau_initial > 0:
            print(
                f"    Tau factor > {tau_max}:    {tau_removed:,} ({tau_removed/tau_initial*100:.2f}%)"
            )
            print(
                f"    NaN values:           {nan_removed:,} ({nan_removed/tau_initial*100:.2f}%)"
            )
            if OUTLIER_CONFIG["enabled"] and all_outlier_stats[0]:
                print(
                    f"    Statistical outliers: {outlier_removed:,} ({outlier_removed/tau_initial*100:.2f}%)"
                )
        print("=" * 80)


# ============================================================================
# DATA GATHERING AND SAVING
# ============================================================================


def gather_pruned_data(df_pruned_local, df_local, comm, rank):
    """Gather pruned data from all ranks to rank 0"""
    if comm is not None:
        # Get original rows that match pruned indices
        df_export_local = df_local[df_local.index.isin(df_pruned_local.index)].copy()
        all_dfs = comm.gather(df_export_local, root=0)

        if rank == 0:
            all_dfs_filtered = [df for df in all_dfs if len(df) > 0]
            if len(all_dfs_filtered) > 0:
                df_pruned_full = pd.concat(all_dfs_filtered, ignore_index=True)
            else:
                df_pruned_full = pd.DataFrame()
            return df_pruned_full
        else:
            return None
    else:
        df_export_local = df_local[df_local.index.isin(df_pruned_local.index)].copy()
        return df_export_local


def save_pruned_data(df_pruned_full, output_file, rank):
    """Save pruned data to a single parquet file (only rank 0)"""
    if rank != 0:
        return

    if df_pruned_full is None or len(df_pruned_full) == 0:
        print("\n⚠️  WARNING: No data to save! All rows were filtered out.")
        print("    Please check:")
        print("    1. Are there missing features in your parquet files?")
        print("    2. Do you have too many NaN values?")
        print("    3. Consider relaxing outlier removal settings")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_pruned_full = df_pruned_full.reset_index(drop=True)
    df_pruned_full.to_parquet(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✅ PRUNED DATA SAVED SUCCESSFULLY!")
    print(f"{'─'*80}")
    print(f"  File: {output_path}")
    print(f"  Rows: {len(df_pruned_full):,}")
    print(f"  Columns: {len(df_pruned_full.columns)}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"{'─'*80}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function"""
    comm, rank, size = get_mpi_info()

    if rank == 0:
        print("=" * 80)
        print("MPI-ENABLED DATA PRUNING SCRIPT")
        print("Sequential Flow: Tau Filter → NaN Removal → Outlier Removal")
        print("=" * 80)
        print(f"MPI Processes: {size}")
        print(f"MPI Available: {MPI_AVAILABLE}")
        print(f"\nFeature Configuration:")
        print(f"  Input features: {len(INPUT_FEATURES)}")
        print(f"  Microstructure features: {len(MICROSTRUCTURE_FEATURES)}")
        print(f"  Performance features: {len(PERFORMANCE_FEATURES)}")
        print(
            f"  Total features: {len(INPUT_FEATURES) + len(MICROSTRUCTURE_FEATURES) + len(PERFORMANCE_FEATURES)}"
        )
        print(f"\nPruning Configuration:")
        print(f"  Step 1 - Tau factor max: {TAU_FACTOR_MAX}")
        print(f"  Step 2 - NaN removal: Enabled")
        print(f"  Step 3 - Outlier removal: {OUTLIER_CONFIG['enabled']}")
        if OUTLIER_CONFIG["enabled"]:
            print(f"           Method: {OUTLIER_CONFIG['method']}")
            print(f"           IQR multiplier: {OUTLIER_CONFIG['iqr_multiplier']}")
            print(
                f"           Features checked: {len(OUTLIER_CONFIG['features_to_check'])}"
            )

    # Load configuration
    if rank == 0:
        print("\nLoading configuration...")

    paths_config, opt_config = load_config()
    intermediate_dir = paths_config["data"]["output"]["intermediate_dir"]
    output_file = Path("data") / "intermediate_pruned.parquet"

    if rank == 0:
        print(f"  Input directory: {intermediate_dir}")
        print(f"  Output file: {output_file}")
        print()

    # Load data in parallel
    df_local = load_intermediate_data_parallel(intermediate_dir, rank, size)

    # Expand nested structures
    df_expanded = expand_dataframe(df_local, rank)

    # ========================================================================
    # PRUNING PIPELINE: Tau → NaN → Outliers
    # ========================================================================

    # STEP 1: Tau Factor Filtering (Physical Constraint)
    if rank == 0:
        print("\n" + "=" * 80)
        print("STEP 1: TAU FACTOR FILTERING (Physical Constraint)")
        print("=" * 80)

    df_after_tau, tau_stats = filter_tau_factor(df_expanded, rank, TAU_FACTOR_MAX)

    # STEP 2: NaN Removal (Data Quality)
    if rank == 0:
        print("\n" + "=" * 80)
        print("STEP 2: NaN REMOVAL (Data Quality)")
        print("=" * 80)

    df_after_nan, nan_stats = remove_nans(df_after_tau, rank)

    # STEP 3: Outlier Removal (Statistical Cleanup)
    outlier_stats = {}
    if OUTLIER_CONFIG["enabled"]:
        if rank == 0:
            print("\n" + "=" * 80)
            print("STEP 3: OUTLIER REMOVAL (Statistical Cleanup)")
            print("=" * 80)

        df_final, outlier_stats = remove_outliers(df_after_nan, rank, OUTLIER_CONFIG)
    else:
        df_final = df_after_nan
        if rank == 0:
            print("\nStep 3: Outlier removal disabled")

    # ========================================================================
    # GATHER STATISTICS AND SAVE
    # ========================================================================

    # Gather and print comprehensive statistics
    gather_and_print_stats(
        tau_stats, nan_stats, outlier_stats, comm, rank, size, TAU_FACTOR_MAX
    )

    # Gather pruned data to rank 0
    if rank == 0:
        print("\nGathering pruned data from all ranks...")
    df_pruned_full = gather_pruned_data(df_final, df_local, comm, rank)

    # Save results
    save_pruned_data(df_pruned_full, output_file, rank)

    if rank == 0:
        if df_pruned_full is not None and len(df_pruned_full) > 0:
            print("\n" + "=" * 80)
            print("🎉 PRUNING COMPLETE!")
            print("=" * 80)
            print(f"\nData Quality Pipeline Executed:")
            print(f"  1. ✓ Physical constraint (tau_factor <= {TAU_FACTOR_MAX})")
            print(f"  2. ✓ Data quality (NaN removal)")
            print(f"  3. ✓ Statistical cleanup (outlier removal)")
            print(f"\nNext Steps:")
            print(f"  → Run: python scripts/optimise_from_parquet.py")
            print(f"  → This will create train/val/test splits with normalization")
            print("=" * 80)


if __name__ == "__main__":
    main()
