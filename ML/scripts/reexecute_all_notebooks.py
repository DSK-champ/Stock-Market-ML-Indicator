"""Execute notebooks 01, 02, 03, 04 in fresh kernels, verify all figures render, and compare sizes."""

import sys
import time
from pathlib import Path

# Setup paths
ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from scripts.execute_notebook import execute_notebook

NOTEBOOKS = [
    ML_DIR / "notebooks" / "01_Data_Collection.ipynb",
    ML_DIR / "notebooks" / "02_Define_Targets.ipynb",
    ML_DIR / "notebooks" / "03_EDA.ipynb",
    ML_DIR / "notebooks" / "04_Feature_Engineering.ipynb",
]

FIGURES_DIR = ML_DIR / "results" / "figures"

def run_reexecution():
    # 1. Snapshot figure sizes before execution
    before_sizes = {f.name: f.stat().st_size for f in FIGURES_DIR.glob("*.png")}
    print(f"Recorded {len(before_sizes)} figure sizes prior to execution.\n")

    # 2. Execute each notebook
    nb_stats = {}
    for nb_path in NOTEBOOKS:
        print(f"\n=======================================================")
        print(f"EXECUTING: {nb_path.name}")
        print(f"=======================================================")
        t0 = time.time()
        stats = execute_notebook(nb_path)
        stats["runtime_sec"] = round(time.time() - t0, 1)
        nb_stats[nb_path.name] = stats

    # 3. Snapshot figure sizes after execution
    after_sizes = {f.name: f.stat().st_size for f in FIGURES_DIR.glob("*.png")}
    
    print("\n\n=======================================================")
    print("NOTEBOOK EXECUTION SUMMARY TABLE")
    print("=======================================================")
    print(f"{'Notebook':35s} | {'Total':5s} | {'Outputs':7s} | {'Size (MB)':9s} | {'Time (s)':8s}")
    print("-" * 75)
    for name, st in nb_stats.items():
        print(f"{name:35s} | {st['total_cells']:5d} | {st['cells_with_output']:7d} | {st['file_size_mb']:9.2f} | {st['runtime_sec']:8.1f}")

    print("\n\n=======================================================")
    print("KEY FIGURE SIZE COMPARISON (Before vs After Re-execution)")
    print("=======================================================")
    key_figures = [
        "01_full_period_price_linear.png",
        "01_data_quality_visuals.png",
        "02_class_balance.png",
        "02_fwd_ret_distributions.png",
        "03_train_price_moving_averages.png",
        "03_train_return_distributions.png",
        "03_monthly_regimes.png",
        "03_label_counts_all.png",
        "03_label_percent_all.png",
        "03_label_by_horizon.png",
        "03_big_move_balance.png",
        "03_big_move_monthly.png",
        "04_dist_momentum.png",
        "04_dist_volatility.png",
        "04_corr_clustered.png",
        "04_auc_big_move_bar.png",
        "04_cohens_d_big_move_heatmap.png",
        "04_direction_metrics_heatmap.png",
        "04_direction_quarterly_stability.png",
        "04_context_profiles.png",
        "04_acf_deseasoning.png",
    ]

    print(f"{'Figure Name':38s} | {'Before (bytes)':14s} | {'After (bytes)':14s} | {'Diff':8s}")
    print("-" * 80)
    for fn in key_figures:
        b = before_sizes.get(fn, 0)
        a = after_sizes.get(fn, 0)
        diff = a - b
        diff_str = f"{diff:+d}" if diff != 0 else "0 (identical)"
        print(f"{fn:38s} | {b:14,d} | {a:14,d} | {diff_str:8s}")

if __name__ == "__main__":
    run_reexecution()

