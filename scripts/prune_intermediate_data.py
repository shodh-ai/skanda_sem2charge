#!/usr/bin/env python3
"""
MPI-enabled script to prune intermediate data by:
1. Removing rows where tau_factor > 8
2. Removing rows with any NaN values in input, microstructure, or performance parameters
3. Saving to a single parquet file: data/intermediate_pruned.parquet

Usage:
    Single process: python prune_intermediate_data.py
    MPI parallel:   mpirun -n 4 python prune_intermediate_data.py
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


# Feature definitions
INPUT_FEATURES = [
    "SEI_kinetic_rate",
    "Electrolyte_diffusivity",
    "Initial_conc_electrolyte",
    "Separator_porosity",
    "Separator_Bruggeman_electrolyte",
    "Separator_Bruggeman",
    "Positive_particle_radius",
    "Negative_particle_radius",
    "Positive_electrode_thickness",
    "Negative_electrode_thickness",
]

MICROSTRUCTURE_FEATURES = [
    "D_eff",
    "porosity_measured",
    "tau_factor",
    "bruggeman_derived",
]

PERFORMANCE_FEATURES = [
    "nominal_capacity_Ah",
    "eol_cycle_measured",
    "initial_capacity_Ah",
    "final_capacity_Ah",
    "capacity_retention_percent",
    "total_cycles",
    "final_RUL",
]

# Pruning thresholds
TAU_FACTOR_MAX = 8.0


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


def load_intermediate_data_parallel(intermediate_dir, rank, size):
    """Load intermediate parquet files in parallel (each rank loads subset)"""
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
        if rank == 0 or len(my_files) <= 3:  # Print details for rank 0 or if few files
            print(f"  Rank {rank}: {pf.name} - {len(df)} rows")

    if dfs:
        df_local = pd.concat(dfs, ignore_index=True)
        if rank == 0:
            print(f"\nRank {rank} loaded {len(df_local)} rows")
    else:
        df_local = pd.DataFrame()  # Empty dataframe if no files assigned

    return df_local


def expand_dataframe(df_local, rank):
    """Expand nested arrays into separate columns"""
    if rank == 0:
        print("\nExpanding nested data structures...")

    if len(df_local) == 0:
        # Return empty expanded dataframe with correct columns
        all_cols = (
            ["sample_id", "param_id"]
            + INPUT_FEATURES
            + MICROSTRUCTURE_FEATURES
            + PERFORMANCE_FEATURES
        )
        return pd.DataFrame(columns=all_cols)

    # Extract input parameters
    input_arrays = np.array(df_local["input_params"].tolist())
    df_inputs = pd.DataFrame(input_arrays, columns=INPUT_FEATURES)

    # Extract microstructure outputs
    micro_arrays = np.array(df_local["microstructure_outputs"].tolist())
    df_micro = pd.DataFrame(micro_arrays, columns=MICROSTRUCTURE_FEATURES)

    # Extract performance outputs
    perf_arrays = np.array(df_local["performance_outputs"].tolist())
    df_perf = pd.DataFrame(perf_arrays, columns=PERFORMANCE_FEATURES)

    # Combine everything
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

    return df_expanded


def prune_data_local(df_expanded, rank, tau_max=TAU_FACTOR_MAX):
    """Prune data locally on each rank"""
    initial_samples = len(df_expanded)

    if len(df_expanded) == 0:
        return df_expanded, {"initial": 0, "after_tau": 0, "final": 0}

    # Step 1: Remove rows where tau_factor > tau_max
    df_pruned = df_expanded[df_expanded["tau_factor"] <= tau_max].copy()
    after_tau = len(df_pruned)

    # Step 2: Remove rows with any NaN values
    all_param_cols = INPUT_FEATURES + MICROSTRUCTURE_FEATURES + PERFORMANCE_FEATURES
    df_pruned = df_pruned.dropna(subset=all_param_cols)
    final_samples = len(df_pruned)

    stats = {"initial": initial_samples, "after_tau": after_tau, "final": final_samples}

    if rank == 0 or initial_samples > 0:
        print(
            f"Rank {rank}: {initial_samples} -> {after_tau} -> {final_samples} samples"
        )

    return df_pruned, stats


def gather_and_print_stats(stats, comm, rank, size, tau_max):
    """Gather statistics from all ranks and print summary"""
    if comm is not None:
        # Gather stats from all ranks
        all_stats = comm.gather(stats, root=0)
    else:
        all_stats = [stats]

    if rank == 0:
        # Aggregate statistics
        total_initial = sum(s["initial"] for s in all_stats)
        total_after_tau = sum(s["after_tau"] for s in all_stats)
        total_final = sum(s["final"] for s in all_stats)

        print("\n" + "=" * 60)
        print("PRUNING SUMMARY (AGGREGATED FROM ALL RANKS)")
        print("=" * 60)
        print(f"Conditions:")
        print(f"  - tau_factor <= {tau_max}")
        print(f"  - No NaN values in any parameter")
        print("=" * 60)
        print(f"\nInitial samples:     {total_initial:,}")
        print(f"\nAfter tau clipping:  {total_after_tau:,}")
        print(f"  Removed:           {total_initial - total_after_tau:,}")
        print(f"  Retention:         {total_after_tau/total_initial*100:.2f}%")
        print(f"\nAfter NaN removal:   {total_final:,}")
        print(f"  Removed:           {total_after_tau - total_final:,}")
        print(f"  Retention:         {total_final/total_after_tau*100:.2f}%")
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Initial samples:     {total_initial:,}")
        print(f"Final samples:       {total_final:,}")
        print(f"Total removed:       {total_initial - total_final:,}")
        print(f"Overall retention:   {total_final/total_initial*100:.2f}%")
        print("=" * 60)


def gather_pruned_data(df_pruned_local, df_local, comm, rank):
    """Gather pruned data from all ranks to rank 0"""
    if comm is not None:
        # Get original rows that match pruned indices
        df_export_local = df_local[df_local.index.isin(df_pruned_local.index)].copy()

        # Gather all dataframes to rank 0
        all_dfs = comm.gather(df_export_local, root=0)

        if rank == 0:
            # Concatenate all gathered dataframes
            df_pruned_full = pd.concat(
                [df for df in all_dfs if len(df) > 0], ignore_index=True
            )
            return df_pruned_full
        else:
            return None
    else:
        # Serial mode
        df_export_local = df_local[df_local.index.isin(df_pruned_local.index)].copy()
        return df_export_local


def save_pruned_data(df_pruned_full, output_file, rank):
    """Save pruned data to a single parquet file (only rank 0)"""
    if rank != 0:
        return

    if df_pruned_full is None or len(df_pruned_full) == 0:
        print("Warning: No data to save!")
        return

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset index and save
    df_pruned_full = df_pruned_full.reset_index(drop=True)
    df_pruned_full.to_parquet(output_path, index=False)

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✓ Pruned data saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Rows: {len(df_pruned_full):,}")
    print(f"  Size: {file_size_mb:.2f} MB")


def main():
    """Main execution function"""
    # Get MPI info
    comm, rank, size = get_mpi_info()

    if rank == 0:
        print("=" * 60)
        print("MPI-ENABLED DATA PRUNING SCRIPT")
        print("=" * 60)
        print(f"MPI Processes: {size}")
        print(f"MPI Available: {MPI_AVAILABLE}")

    # Load configuration (all ranks)
    if rank == 0:
        print("\nLoading configuration...")

    paths_config, opt_config = load_config()

    # Get paths from config
    intermediate_dir = paths_config["data"]["output"]["intermediate_dir"]
    output_file = Path("data") / "intermediate_pruned.parquet"

    if rank == 0:
        print(f"  Intermediate directory: {intermediate_dir}")
        print(f"  Output file: {output_file}")
        print()

    # Load data in parallel (each rank loads subset of files)
    df_local = load_intermediate_data_parallel(intermediate_dir, rank, size)

    # Expand nested structures (each rank processes its data)
    df_expanded_local = expand_dataframe(df_local, rank)

    # Prune data locally
    if rank == 0:
        print("\nPruning data on all ranks...")
    df_pruned_local, stats = prune_data_local(
        df_expanded_local, rank, tau_max=TAU_FACTOR_MAX
    )

    # Gather and print statistics
    gather_and_print_stats(stats, comm, rank, size, TAU_FACTOR_MAX)

    # Gather pruned data to rank 0
    if rank == 0:
        print("\nGathering pruned data from all ranks...")
    df_pruned_full = gather_pruned_data(df_pruned_local, df_local, comm, rank)

    # Save results (only rank 0)
    save_pruned_data(df_pruned_full, output_file, rank)

    if rank == 0:
        print("\n" + "=" * 60)
        print("PRUNING COMPLETE!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Use this file for creating optimized dataset:")
        print(f"     python create_optimized_dataset.py")
        print(f"  2. The file is ready to be loaded into memory for processing")
        print("=" * 60)


if __name__ == "__main__":
    main()
