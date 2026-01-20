import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pyarrow.parquet as pq
from IPython.display import display, HTML

# --- Configuration ---
# Update these paths relative to where you launch the notebook
BASE_DIR = "../data/raw"
PARAM_SWEEP_DIR = os.path.join(BASE_DIR, "output_parameter_sweep")
PYBAMM_DIR = os.path.join(BASE_DIR, "final_pybamm_output")

# File Names
MAIN_RESULTS_CSV = "final_results.csv"
TAUFACTOR_CSV = "taufactor_results.csv"

print(f"Searching for data in: {os.path.abspath(BASE_DIR)}")
