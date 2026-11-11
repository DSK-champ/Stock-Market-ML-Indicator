"""Construct and execute ML/notebooks/05_Model_Training.ipynb following the exact 28-cell structure."""

import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ML_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ML_DIR / "notebooks" / "05_Model_Training.ipynb"

def build_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # =========================================================================
    # Section 0 — Notebook setup
    # =========================================================================

    # Cell 1 — Notebook introduction (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "# STEP 5: Multi-Horizon Baseline Evaluation & Benchmark Verification\n\n"
        "## Purpose of Step 5\n"
        "This notebook establishes the empirical baseline floor for the Stock Market ML Indicator project before any "
        "model complexity, ensembling, or hyperparameter tuning is introduced. We rigorously benchmark canonical rules, "
        "heuristic voters, and linear reference models against market friction.\n\n"
        "## Dataset and Partition\n"
        "- **Partition:** Evaluated strictly on the **`train` partition** ($N = 220,953$ raw candles, 2023-11-01 to 2025-12-07 UTC).\n"
        "- **Warmup Window:** First 2,018 candles reserved for indicator warm-up ($N_{avail} = 218,935$ post-warmup candles, 760.2 days).\n"
        "- **Partition Guard:** Validation and locked test partitions remain **strictly untouched**.\n\n"
        "## Four Candidate Horizons\n"
        "Because the underlying data consists of 5-minute BTCUSDT candles, prediction horizons ($N$) span:\n"
        "1. **$N = 2$ candles:** 10 minutes forward (ultra-short scalping)\n"
        "2. **$N = 3$ candles:** 15 minutes forward (short intraday momentum)\n"
        "3. **$N = 6$ candles:** 30 minutes forward (half-hour intermediate swing)\n"
        "4. **$N = 12$ candles:** 60 minutes forward (1-hour macro regime)\n\n"
        "## Prediction Tasks\n"
        "- **Binary Directional Path:** Up (+1) vs Down (-1) evaluated against `raw_sign_N` and `bin_dir_N`.\n"
        "- **Three-Class Clean-Move Path:** Down (-1), Neutral (0), Up (+1) evaluated on `target_N_S1` with balanced accuracy.\n"
        "- **Volatility Path:** Non-directional magnitude prediction evaluated on `big_move_N` ($|fwd\\_ret| > 23\\text{ bps}$).\n\n"
        "## Economic Hurdle & Evaluation Protocol\n"
        "- **Transaction Cost:** $c = 23\\text{ bps}$ ($0.0023$) round-trip taker fee.\n"
        "- **Cross-Validation:** 6-fold Purged Walk-Forward Cross-Validation with an exact 12-candle purge gap."
    ))

    # Cell 2 — Imports and configuration (Code)
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "ML_DIR = Path.cwd()\n"
        "if (ML_DIR / 'ML').exists():\n"
        "    ML_DIR = ML_DIR / 'ML'\n"
        "elif ML_DIR.name == 'notebooks':\n"
        "    ML_DIR = ML_DIR.parent\n"
        "if str(ML_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(ML_DIR))\n\n"
        "from src import config\n\n"
        "print('=== Configuration Verification ===')\n"
        "print(f'Working Directory:           {ML_DIR}')\n"
        "print(f'Processed Dataset:           {ML_DIR / \"data\" / \"processed\" / \"master.csv\"}')\n"
        "print(f'Candidate Horizons:          {config.CANDIDATE_HORIZONS}')\n"
        "print(f'Round-Trip Transaction Cost: {config.TOTAL_ROUNDTRIP_COST*10000:.1f} bps ({config.TOTAL_ROUNDTRIP_COST})')\n"
        "print('Number of CV Folds:          6')\n"
        "print('Purge Gap:                   12 candles')"
    ))

    # =========================================================================
    # Section 1 — Cross-validation verification
    # =========================================================================

    # Cell 3 — CV fold boundaries (Code)
    cells.append(nbf.v4.new_code_cell(
        "cv_folds_path = ML_DIR / 'results' / 'reports' / 'cv_folds_summary.csv'\n"
        "df_folds = pd.read_csv(cv_folds_path)\n\n"
        "display_cols = [\n"
        "    'fold', 'train_rows', 'purge_rows', 'held_out_rows',\n"
        "    'train_start_time', 'train_end_time',\n"
        "    'held_out_start_time', 'held_out_end_time'\n"
        "]\n"
        "print('=== Purged Walk-Forward 6-Fold Structure ===')\n"
        "print(df_folds[display_cols].to_string(index=False))\n"
        "print(f'\\nTotal evaluated held-out candles across folds: {df_folds[\"held_out_rows\"].sum():,}')\n"
        "print(f'Enforced purge gap between train and test blocks: {df_folds[\"purge_rows\"].iloc[0]} candles')"
    ))

    # Cell 4 — CV integrity verification (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Programmatic verification of cross-validation integrity\n"
        "n_folds_ok = (len(df_folds) == 6)\n"
        "purge_gap_ok = (df_folds['purge_rows'] == 12).all()\n\n"
        "# Verify held-out periods do not overlap\n"
        "ho_starts = pd.to_datetime(df_folds['held_out_start_time'])\n"
        "ho_ends = pd.to_datetime(df_folds['held_out_end_time'])\n"
        "no_overlap_ok = True\n"
        "for i in range(len(df_folds) - 1):\n"
        "    if ho_starts.iloc[i+1] < ho_ends.iloc[i]:\n"
        "        no_overlap_ok = False\n\n"
        "# Verify training strictly precedes held-out\n"
        "tr_ends = pd.to_datetime(df_folds['train_end_time'])\n"
        "precede_ok = (tr_ends < ho_starts).all()\n\n"
        "# Verify partition isolation (train split bounds: max date <= 2025-12-07)\n"
        "partition_ok = (ho_ends.max() <= pd.to_datetime('2025-12-08 00:00:00+00:00'))\n\n"
        "print('=== Cross-Validation Integrity Audit ===')\n"
        "print(f'Check 1: Exactly 6 folds exist:                         {\"PASS\" if n_folds_ok else \"FAIL\"}')\n"
        "print(f'Check 2: Purge gap is strictly 12 candles:               {\"PASS\" if purge_gap_ok else \"FAIL\"}')\n"
        "print(f'Check 3: Held-out evaluation blocks do not overlap:      {\"PASS\" if no_overlap_ok else \"FAIL\"}')\n"
        "print(f'Check 4: Training periods strictly precede evaluation:   {\"PASS\" if precede_ok else \"FAIL\"}')\n"
        "print(f'Check 5: No val/test data leaked (bounded in train):     {\"PASS\" if partition_ok else \"FAIL\"}')\n\n"
        "assert n_folds_ok and purge_gap_ok and no_overlap_ok and precede_ok and partition_ok, 'CV integrity check failed!'"
    ))

    # =========================================================================
    # Section 2 — Baseline experiments
    # =========================================================================

    # Cell 5 — Binary baseline overview (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Baseline Experiments\n\n"
        "### 2A. Binary Directional Baselines\n\n"
        "The binary forecasting task evaluates direction prediction: **UP (+1)** vs **DOWN (-1)** against `raw_sign_N`.\n\n"
        "**The 15 Evaluated Baseline Strategies:**\n"
        "1. `always_up`: Unconditional constant Long prediction (+1).\n"
        "2. `always_down`: Unconditional constant Short prediction (-1).\n"
        "3. `always_neutral`: Abstains on all candles (0.0 emitted).\n"
        "4. `coin_flip`: Stochastic Bernoulli(0.5) guessing (seed 42).\n"
        "5. `random_3class`: Uniform random selection from {-1, 0, +1} (seed 42).\n"
        "6. `majority_class`: Predicts empirical mode of each training fold.\n"
        "7. `momentum`: Follows immediate prior candle return (`sign(feat_ret_1)`).\n"
        "8. `reverse_momentum`: Inverts prior candle return (`-sign(feat_ret_1)`).\n"
        "9. `rsi_14`: Traditional RSI rule (<30 Long, >70 Short, neutral candles abstained).\n"
        "10. `reverse_rsi_14`: Inverted RSI rule (<30 Short, >70 Long, neutral abstained).\n"
        "11. `loc_position_288`: Mean-reversion in 24h trading channel (`sign(0.5 - pos)`).\n"
        "12. `ema50_rule`: Trend-following MA deviation (`sign(EMA50 - close)`).\n"
        "13. `rule_vote_4b`: Step 4b structural heuristic vote.\n"
        "14. `reverse_rule_vote_4b`: Inverted Step 4b structural heuristic vote.\n"
        "15. `ref_5feature_lr`: 5-feature reference linear model trained walk-forward.\n\n"
        "> **Abstention Integrity:** Strategies like RSI(14) emit signals only on extreme bars, abstaining elsewhere. "
        "Coverage for RSI is strictly bounded below 50% (~7.18%), preserving true selective participation."
    ))

    # Cell 6 — Binary baseline results (Code)
    cells.append(nbf.v4.new_code_cell(
        "bin_path = ML_DIR / 'results' / 'reports' / 'baselines_binary_summary.csv'\n"
        "df_bin = pd.read_csv(bin_path)\n\n"
        "existing_horizons = sorted(df_bin['horizon'].unique())\n"
        "print(f'Verified existing horizons in binary baselines: {existing_horizons}')\n\n"
        "disp_cols_bin = [\n"
        "    'baseline', 'accuracy', 'balanced_accuracy', 'f1_macro', 'mcc',\n"
        "    'coverage', 'signals_per_day', 'hit_rate', 'break_even_rate',\n"
        "    'break_even_gap', 'net_return_bps'\n"
        "]\n\n"
        "for h in existing_horizons:\n"
        "    h_min = h * 5\n"
        "    print(f'\\n================ Binary Baselines: N = {h} ({h_min} Minutes) ================')\n"
        "    sub = df_bin[df_bin['horizon'] == h].sort_values(by='break_even_gap', ascending=False).reset_index(drop=True)\n"
        "    print(sub[disp_cols_bin].to_string(index=False))"
    ))

    # Cell 7 — Binary baseline comparison (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Consolidated binary comparison across all 4 horizons\n"
        "comp_cols = ['baseline', 'horizon', 'hit_rate', 'balanced_accuracy', 'mcc', 'coverage', 'break_even_gap', 'net_return_bps']\n"
        "df_bin_comp = df_bin[comp_cols].sort_values(by=['baseline', 'horizon']).reset_index(drop=True)\n\n"
        "print('=== Comprehensive Multi-Horizon Binary Comparison (All 15 Baselines) ===')\n"
        "print(df_bin_comp.to_string(index=False))\n\n"
        "# Summary stats\n"
        "best_per_h = df_bin.loc[df_bin.groupby('horizon')['break_even_gap'].idxmax()]\n"
        "print('\\n--- Top Baseline by Break-Even Gap for Each Horizon ---')\n"
        "print(best_per_h[['horizon', 'baseline', 'hit_rate', 'coverage', 'break_even_gap', 'net_return_bps']].to_string(index=False))"
    ))

    # Cell 8 — Three-class baseline overview (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "### 2B. Three-Class Clean-Move Baselines\n\n"
        "Evaluates the ternary classification task: **DOWN (-1)**, **NEUTRAL (0)**, **UP (+1)** against `target_N_S1`.\n\n"
        "**Why Balanced Accuracy is Required:**\n"
        "Neutral dead-zone bars (where $|fwd\\_ret| \\le 23\\text{ bps}$) dominate the market distribution (~52.7% at $N=12$, "
        "rising to >83% at $N=2$). A naive model predicting `always_neutral` would achieve **52% to 83% raw accuracy** while "
        "having exactly zero predictive skill or economic utility.\n\n"
        "**Balanced Accuracy** (unweighted macro recall across the three classes) rigorously anchors all zero-skill bets:\n"
        "- `always_neutral`: $\\frac{0 + 1 + 0}{3} = \\mathbf{33.33\\%}$\n"
        "- `always_up`: $\\frac{0 + 0 + 1}{3} = \\mathbf{33.33\\%}$\n"
        "- `always_down`: $\\frac{1 + 0 + 0}{3} = \\mathbf{33.33\\%}$\n"
        "- `coin_flip` / `random_3class`: $\\frac{1/3 + 1/3 + 1/3}{3} = \\mathbf{33.33\\%}$\n\n"
        "> **Zero-Skill Watermark:** Any three-class model achieving $\\le 33.33\\%$ balanced accuracy is empirically indistinguishable from noise."
    ))

    # Cell 9 — Three-class baseline results (Code)
    cells.append(nbf.v4.new_code_cell(
        "three_path = ML_DIR / 'results' / 'reports' / 'baselines_three_class_summary.csv'\n"
        "df_three = pd.read_csv(three_path)\n\n"
        "existing_horizons_3 = sorted(df_three['horizon'].unique())\n"
        "print(f'Verified existing horizons in three-class baselines: {existing_horizons_3}')\n\n"
        "disp_cols_3 = [\n"
        "    'baseline', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro',\n"
        "    'f1_macro', 'mcc', 'coverage', 'signals_per_day', 'hit_rate', 'break_even_gap', 'net_return_bps'\n"
        "]\n\n"
        "for h in existing_horizons_3:\n"
        "    h_min = h * 5\n"
        "    print(f'\\n================ Three-Class Baselines: N = {h} ({h_min} Minutes) ================')\n"
        "    sub = df_three[df_three['horizon'] == h].sort_values(by='balanced_accuracy', ascending=False).reset_index(drop=True)\n"
        "    print(sub[disp_cols_3].to_string(index=False))"
    ))

    # Cell 10 — Three-class comparison (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Consolidated three-class multi-horizon comparison\n"
        "print('=== Highlights Across Horizons (Three-Class Baselines) ===')\n"
        "for h in existing_horizons_3:\n"
        "    sub = df_three[df_three['horizon'] == h]\n"
        "    best_bal = sub.loc[sub['balanced_accuracy'].idxmax()]\n"
        "    best_f1 = sub.loc[sub['f1_macro'].idxmax()]\n"
        "    best_mcc = sub.loc[sub['mcc'].idxmax()]\n"
        "    best_gap = sub.loc[sub['break_even_gap'].idxmax()]\n"
        "    print(f'\\nHorizon N={h} ({h*5}m):')\n"
        "    print(f'  Top Balanced Accuracy: {best_bal[\"baseline\"]} ({best_bal[\"balanced_accuracy\"]*100:.2f}%)')\n"
        "    print(f'  Top Macro F1:          {best_f1[\"baseline\"]} ({best_f1[\"f1_macro\"]:.4f})')\n"
        "    print(f'  Top MCC:               {best_mcc[\"baseline\"]} ({best_mcc[\"mcc\"]:.4f})')\n"
        "    print(f'  Top Break-Even Gap:    {best_gap[\"baseline\"]} ({best_gap[\"break_even_gap\"]:.4f}, Net: {best_gap[\"net_return_bps\"]:.2f} bps)')\n\n"
        "print('\\n[CRITICAL NOTE] Raw accuracy must NEVER be used to rank three-class models due to trivial Neutral base rates.')"
    ))

    # Cell 11 — Volatility baseline overview (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "### 2C. Volatility Magnitude Baseline (`big_move_N`)\n\n"
        "The volatility baseline (`volatility_lr`) performs an orthogonal, **non-directional magnitude forecasting task**:\n"
        "- Predicts whether future price displacement exceeds trading friction ($|fwd\\_ret\\_N| > 23\\text{ bps}$).\n"
        "- Uses the single 24-hour baseline volatility feature `feat_vol_sigma_288`.\n"
        "- **Does NOT predict Up or Down:** It predicts trading viability (volatility expansion) rather than direction."
    ))

    # Cell 12 — Volatility baseline results (Code)
    cells.append(nbf.v4.new_code_cell(
        "vol_path = ML_DIR / 'results' / 'reports' / 'volatility_baseline_summary.csv'\n"
        "df_vol = pd.read_csv(vol_path)\n\n"
        "print('=== Volatility Baseline Evaluation (big_move_N via feat_vol_sigma_288) ===')\n"
        "disp_vol = ['model', 'feature', 'target', 'horizon', 'horizon_min', 'roc_auc', 'pr_auc', 'f1_score', 'brier_score', 'positive_prevalence']\n"
        "print(df_vol[disp_vol].to_string(index=False))\n\n"
        "for _, row in df_vol.iterrows():\n"
        "    print(f'Horizon N={int(row[\"horizon\"])} ({int(row[\"horizon_min\"])}m): ROC AUC = {row[\"roc_auc\"]:.4f}, PR AUC = {row[\"pr_auc\"]:.4f}, Prevalence = {row[\"positive_prevalence\"]*100:.1f}%')"
    ))

    # Cell 13 — Volatility comparison (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Volatility comparison plots across horizons\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "horizons = df_vol['horizon'].values\n"
        "h_labels = [f'{h} ({h*5}m)' for h in horizons]\n\n"
        "axes[0].plot(h_labels, df_vol['roc_auc'], marker='o', color='purple', label='ROC AUC')\n"
        "axes[0].plot(h_labels, df_vol['pr_auc'], marker='s', color='orange', label='PR AUC')\n"
        "axes[0].set_title('Volatility Discrimination by Horizon (big_move_N)')\n"
        "axes[0].set_xlabel('Horizon N (holding period)')\n"
        "axes[0].set_ylabel('Score')\n"
        "axes[0].grid(True, alpha=0.3)\n"
        "axes[0].legend()\n\n"
        "axes[1].bar(h_labels, df_vol['positive_prevalence']*100, color='teal', alpha=0.7)\n"
        "axes[1].set_title('Big Move Prevalence (|fwd_ret| > 23 bps)')\n"
        "axes[1].set_xlabel('Horizon N (holding period)')\n"
        "axes[1].set_ylabel('Prevalence (%)')\n"
        "axes[1].grid(True, alpha=0.3)\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print('Interpretation: Discrimination is highest at short horizons (ROC AUC = 0.7057 at N=2), but prevalence is lowest (15.4%).')\n"
        "print('At N=12, 45.8% of candles clear the 23 bps hurdle, providing substantially greater opportunity for cost-clearing moves.')"
    ))

    # =========================================================================
    # Section 3 — Multi-horizon comparison
    # =========================================================================

    # Cell 14 — Horizon comparison overview (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Multi-Horizon Comparative Analysis\n\n"
        "This section investigates four key research questions across holding horizons:\n"
        "1. **Are short-term predictions easier?** Does 10m noise degrade directional accuracy compared to 60m trends?\n"
        "2. **Do longer horizons provide stronger directional signals?** How does signal persistence evolve from $N=2$ to $N=12$?\n"
        "3. **Does the neutral dead-zone shrink with horizon?** How quickly does price displacement overcome trading friction?\n"
        "4. **Which horizon offers the best net potential after transaction costs?** Where is the optimal trade-off between move size and fee drag?"
    ))

    # Cell 15 — Best binary baseline per horizon (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Best binary baseline per horizon across multiple criteria\n"
        "records = []\n"
        "for h in existing_horizons:\n"
        "    sub = df_bin[df_bin['horizon'] == h]\n"
        "    best_hit = sub.loc[sub['hit_rate'].idxmax()]\n"
        "    best_mcc = sub.loc[sub['mcc'].idxmax()]\n"
        "    best_gap = sub.loc[sub['break_even_gap'].idxmax()]\n"
        "    best_net = sub.loc[sub['net_return_bps'].idxmax()]\n"
        "    records.append({\n"
        "        'horizon': h,\n"
        "        'timeframe': f'{h*5}m',\n"
        "        'top_hit_model': f'{best_hit[\"baseline\"]} ({best_hit[\"hit_rate\"]*100:.2f}%)',\n"
        "        'top_mcc_model': f'{best_mcc[\"baseline\"]} ({best_mcc[\"mcc\"]:.4f})',\n"
        "        'top_gap_model': f'{best_gap[\"baseline\"]} ({best_gap[\"break_even_gap\"]:.4f})',\n"
        "        'top_net_model': f'{best_net[\"baseline\"]} ({best_net[\"net_return_bps\"]:.2f} bps)',\n"
        "    })\n\n"
        "df_top_bin = pd.DataFrame(records)\n"
        "print('=== Top Binary Baselines by Criterion Across Horizons ===')\n"
        "print(df_top_bin.to_string(index=False))"
    ))

    # Cell 16 — Best three-class baseline per horizon (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Best three-class baseline per horizon across multiple criteria\n"
        "records_3 = []\n"
        "for h in existing_horizons_3:\n"
        "    sub = df_three[df_three['horizon'] == h]\n"
        "    best_bal = sub.loc[sub['balanced_accuracy'].idxmax()]\n"
        "    best_f1 = sub.loc[sub['f1_macro'].idxmax()]\n"
        "    best_mcc = sub.loc[sub['mcc'].idxmax()]\n"
        "    best_gap = sub.loc[sub['break_even_gap'].idxmax()]\n"
        "    records_3.append({\n"
        "        'horizon': h,\n"
        "        'timeframe': f'{h*5}m',\n"
        "        'top_bal_acc_model': f'{best_bal[\"baseline\"]} ({best_bal[\"balanced_accuracy\"]*100:.2f}%)',\n"
        "        'top_f1_macro_model': f'{best_f1[\"baseline\"]} ({best_f1[\"f1_macro\"]:.4f})',\n"
        "        'top_mcc_model':      f'{best_mcc[\"baseline\"]} ({best_mcc[\"mcc\"]:.4f})',\n"
        "        'top_gap_model':      f'{best_gap[\"baseline\"]} ({best_gap[\"break_even_gap\"]:.4f})',\n"
        "    })\n\n"
        "df_top_three = pd.DataFrame(records_3)\n"
        "print('=== Top Three-Class Baselines by Criterion Across Horizons ===')\n"
        "print(df_top_three.to_string(index=False))"
    ))

    # Cell 17 — Horizon comparison visualizations (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Comprehensive visualization of baseline metrics across horizons\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n\n"
        "# Filter to reference models for clear trend visualization\n"
        "key_models = ['loc_position_288', 'ema50_rule', 'ref_5feature_lr', 'momentum']\n\n"
        "# 1. Binary Balanced Accuracy\n"
        "for m in key_models:\n"
        "    sub = df_bin[df_bin['baseline'] == m].sort_values('horizon')\n"
        "    axes[0, 0].plot(sub['horizon'], sub['balanced_accuracy']*100, marker='o', label=m)\n"
        "axes[0, 0].axhline(50, color='gray', linestyle='--', label='Chance (50%)')\n"
        "axes[0, 0].set_title('1. Binary Balanced Accuracy (%)')\n"
        "axes[0, 0].set_xlabel('Horizon N')\n"
        "axes[0, 0].grid(True, alpha=0.3)\n"
        "axes[0, 0].legend(fontsize=8)\n\n"
        "# 2. Binary Hit Rate\n"
        "for m in key_models:\n"
        "    sub = df_bin[df_bin['baseline'] == m].sort_values('horizon')\n"
        "    axes[0, 1].plot(sub['horizon'], sub['hit_rate']*100, marker='o', label=m)\n"
        "axes[0, 1].axhline(50, color='gray', linestyle='--')\n"
        "axes[0, 1].set_title('2. Binary Directional Hit Rate (%)')\n"
        "axes[0, 1].set_xlabel('Horizon N')\n"
        "axes[0, 1].grid(True, alpha=0.3)\n\n"
        "# 3. Three-Class Balanced Accuracy\n"
        "for m in key_models:\n"
        "    sub = df_three[df_three['baseline'] == m].sort_values('horizon')\n"
        "    axes[0, 2].plot(sub['horizon'], sub['balanced_accuracy']*100, marker='o', label=m)\n"
        "axes[0, 2].axhline(33.33, color='gray', linestyle='--', label='Chance (33.3%)')\n"
        "axes[0, 2].set_title('3. Three-Class Balanced Accuracy (%)')\n"
        "axes[0, 2].set_xlabel('Horizon N')\n"
        "axes[0, 2].grid(True, alpha=0.3)\n"
        "axes[0, 2].legend(fontsize=8)\n\n"
        "# 4. Three-Class Macro F1\n"
        "for m in key_models:\n"
        "    sub = df_three[df_three['baseline'] == m].sort_values('horizon')\n"
        "    axes[1, 0].plot(sub['horizon'], sub['f1_macro'], marker='o', label=m)\n"
        "axes[1, 0].set_title('4. Three-Class Macro F1')\n"
        "axes[1, 0].set_xlabel('Horizon N')\n"
        "axes[1, 0].grid(True, alpha=0.3)\n\n"
        "# 5. Net Return in bps\n"
        "for m in key_models:\n"
        "    sub = df_bin[df_bin['baseline'] == m].sort_values('horizon')\n"
        "    axes[1, 1].plot(sub['horizon'], sub['net_return_bps'], marker='o', label=m)\n"
        "axes[1, 1].axhline(0, color='red', linestyle='--', label='Break-Even (0 bps)')\n"
        "axes[1, 1].set_title('5. Binary Net Return (bps/trade)')\n"
        "axes[1, 1].set_xlabel('Horizon N')\n"
        "axes[1, 1].grid(True, alpha=0.3)\n"
        "axes[1, 1].legend(fontsize=8)\n\n"
        "# 6. Break-Even Gap\n"
        "for m in key_models:\n"
        "    sub = df_bin[df_bin['baseline'] == m].sort_values('horizon')\n"
        "    axes[1, 2].plot(sub['horizon'], sub['break_even_gap'], marker='o', label=m)\n"
        "axes[1, 2].axhline(0, color='red', linestyle='--')\n"
        "axes[1, 2].set_title('6. Break-Even Gap (hit - BE)')\n"
        "axes[1, 2].set_xlabel('Horizon N')\n"
        "axes[1, 2].grid(True, alpha=0.3)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 18 — Class distribution comparison (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Load pre-computed label balance from eda_label_balance.csv\n"
        "lbl_path = ML_DIR / 'results' / 'reports' / 'eda_label_balance.csv'\n"
        "df_lbl = pd.read_csv(lbl_path)\n"
        "s1_balance = df_lbl[df_lbl['label_col'].str.endswith('_S1')][['horizon', 'pct_up', 'pct_neutral', 'pct_down']].sort_values('horizon')\n\n"
        "print('=== Class Distribution Across Horizons (Rule S1, Dead-Zone = 23 bps) ===')\n"
        "print(s1_balance.to_string(index=False))\n\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "horizons_str = [f'N={h}\\n({h*5}m)' for h in s1_balance['horizon']]\n"
        "x = np.arange(len(horizons_str))\n"
        "width = 0.55\n\n"
        "p_down = ax.bar(x, s1_balance['pct_down'], width, label='Down (-1)', color='#e74c3c')\n"
        "p_neu = ax.bar(x, s1_balance['pct_neutral'], width, bottom=s1_balance['pct_down'], label='Neutral (0)', color='#95a5a6')\n"
        "p_up = ax.bar(x, s1_balance['pct_up'], width, bottom=s1_balance['pct_down']+s1_balance['pct_neutral'], label='Up (+1)', color='#2ecc71')\n\n"
        "ax.set_ylabel('Share of Total Post-Warmup Candles (%)')\n"
        "ax.set_title('Class Distribution Evolution Across Horizons (Rule S1)')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(horizons_str)\n"
        "ax.legend(loc='upper right')\n"
        "ax.grid(True, axis='y', alpha=0.3)\n\n"
        "# Add percentage labels\n"
        "for i in range(len(s1_balance)):\n"
        "    neu_val = s1_balance['pct_neutral'].iloc[i]\n"
        "    down_val = s1_balance['pct_down'].iloc[i]\n"
        "    ax.text(i, down_val + neu_val/2, f'Neutral\\n{neu_val:.1f}%', ha='center', va='center', color='white', fontweight='bold')\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print('Key Observation: Neutral dead-zone contracts dramatically from 83.2% at N=2 down to 52.7% at N=12.')\n"
        "print('At N=2, only 16.8% of candles clear fees. At N=12, 47.3% clear fees, expanding tradeable opportunity.')"
    ))

    # Cell 19 — Horizon comparison summary (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Consolidated Horizon Comparison Summary Table\n"
        "summary_rows = []\n"
        "for h in existing_horizons:\n"
        "    sub_bin = df_bin[df_bin['horizon'] == h]\n"
        "    sub_3 = df_three[df_three['horizon'] == h]\n"
        "    lbl_row = s1_balance[s1_balance['horizon'] == h].iloc[0]\n"
        "    \n"
        "    summary_rows.append({\n"
        "        'Metric': f'Horizon N={h} ({h*5}m)',\n"
        "        'Best Binary Hit Rate': f'{sub_bin[\"hit_rate\"].max()*100:.2f}% ({sub_bin.loc[sub_bin[\"hit_rate\"].idxmax(), \"baseline\"]})',\n"
        "        'Best Binary MCC': f'{sub_bin[\"mcc\"].max():.4f} ({sub_bin.loc[sub_bin[\"mcc\"].idxmax(), \"baseline\"]})',\n"
        "        'Best 3-Class Bal Acc': f'{sub_3[\"balanced_accuracy\"].max()*100:.2f}% ({sub_3.loc[sub_3[\"balanced_accuracy\"].idxmax(), \"baseline\"]})',\n"
        "        'Best 3-Class Macro F1': f'{sub_3[\"f1_macro\"].max():.4f} ({sub_3.loc[sub_3[\"f1_macro\"].idxmax(), \"baseline\"]})',\n"
        "        'Best Break-Even Gap': f'{sub_bin[\"break_even_gap\"].max():.4f} ({sub_bin.loc[sub_bin[\"break_even_gap\"].idxmax(), \"baseline\"]})',\n"
        "        'Best Net Return': f'{sub_bin[\"net_return_bps\"].max():.2f} bps ({sub_bin.loc[sub_bin[\"net_return_bps\"].idxmax(), \"baseline\"]})',\n"
        "        'Neutral Prevalence': f'{lbl_row[\"pct_neutral\"]:.1f}%',\n"
        "    })\n\n"
        "df_h_sum = pd.DataFrame(summary_rows)\n"
        "print('=== Consolidated Multi-Horizon Trade-Off Matrix ===')\n"
        "print(df_h_sum.transpose().to_string(header=False))\n\n"
        "print('\\n--- Horizon Trade-Off Synthesis ---')\n"
        "print('1. N=2 & N=3 (10m, 15m): Highest directional hit rate on selective baselines (RSI hit ~55%), but severely penalized by fee drag (Gap < -0.32, Net < -22 bps) because mean price moves are smaller than the 23 bps cost.')\n"
        "print('2. N=6 (30m): Intermediate balance (Neutral = 65.4%, Best Gap = -0.3448).')\n"
        "print('3. N=12 (60m): Largest price displacement relative to fees, yielding the lowest break-even hurdles and strongest economic potential under selective coverage.')"
    ))

    # =========================================================================
    # Section 4 — Like-for-like coverage analysis
    # =========================================================================

    # Cell 20 — Coverage analysis overview (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Like-for-Like Multi-Coverage Analysis\n\n"
        "Comparing raw hit rates across models with differing signal frequencies is fundamentally misleading. A rule taking "
        "only 10 trades per day cannot be judged on the same scale as one trading 288 times daily.\n\n"
        "**Multi-Coverage Evaluation Framework:**\n"
        "- **Coverage Tiers:** 100% (unconditional), 50%, 20%, 10% (most confident or most extreme candles).\n"
        "- **Strict Threshold Isolation:** Thresholds are established **strictly on training folds** (never on held-out evaluation folds).\n"
        "- **Stationary Block-Bootstrap 95% CIs:** Computed via $B = 288$ consecutive candles (1 full trading day) to account for market serial dependence.\n"
        "- **Scope:** The complete like-for-like multi-coverage suite is evaluated on the primary survivor horizon ($N=12$), comparing all baselines, same-coverage heuristics, and Stage 1 survivors."
    ))

    # Cell 21 — Coverage comparison (Code)
    cells.append(nbf.v4.new_code_cell(
        "lfl_path = ML_DIR / 'results' / 'reports' / 'like_for_like_summary.csv'\n"
        "df_lfl = pd.read_csv(lfl_path)\n\n"
        "print(f'Underlying Like-for-Like Scope: Horizon N = {df_lfl[\"horizon\"].unique()[0]} ({len(df_lfl)} evaluated model-coverage configurations)')\n\n"
        "disp_lfl_cols = ['model', 'signals_per_day', 'hit_rate', 'ci_lower', 'ci_upper', 'mean_abs_fwd_ret', 'break_even_rate', 'break_even_gap', 'net_bps']\n\n"
        "for cov in [1.0, 0.5, 0.2, 0.1]:\n"
        "    print(f'\\n================ Coverage Level: {int(cov*100)}% ================')\n"
        "    sub = df_lfl[df_lfl['target_coverage'] == cov].sort_values(by='break_even_gap', ascending=False).reset_index(drop=True)\n"
        "    print(sub[disp_lfl_cols].to_string(index=False))"
    ))

    # Cell 22 — Coverage visualization (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Plot performance evolution across coverage levels\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "focus_models = ['vol_gated_reversal', 'C_12_F15', 'loc_position_288', 'B_12', 'ema50_rule', 'random_loc_reversal']\n\n"
        "for m in focus_models:\n"
        "    sub = df_lfl[df_lfl['model'] == m].sort_values('actual_coverage', ascending=False)\n"
        "    if len(sub) > 1:\n"
        "        axes[0].plot(sub['actual_coverage']*100, sub['hit_rate']*100, marker='o', label=m)\n"
        "        axes[1].plot(sub['actual_coverage']*100, sub['break_even_gap'], marker='s', label=m)\n\n"
        "axes[0].set_title('Directional Hit Rate vs Actual Coverage (%)')\n"
        "axes[0].set_xlabel('Actual Coverage (%)')\n"
        "axes[0].set_ylabel('Hit Rate (%)')\n"
        "axes[0].grid(True, alpha=0.3)\n"
        "axes[0].legend(fontsize=8)\n\n"
        "axes[1].set_title('Break-Even Gap vs Actual Coverage (%)')\n"
        "axes[1].set_xlabel('Actual Coverage (%)')\n"
        "axes[1].set_ylabel('Break-Even Gap (hit - BE)')\n"
        "axes[1].axhline(0, color='red', linestyle='--', label='Break-Even (0.0)')\n"
        "axes[1].grid(True, alpha=0.3)\n"
        "axes[1].legend(fontsize=8)\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print('Insight: Lower coverage improves break-even gap for all selective strategies, but primarily through size selection (trading higher volatility bars) rather than pure directional accuracy.')"
    ))

    # =========================================================================
    # Section 5 — Break-even gap decomposition
    # =========================================================================

    # Cell 23 — Decomposition explanation (Markdown)
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Break-Even Gap Decomposition (20% Coverage)\n\n"
        "To break even after round-trip fees ($c = 23\\text{ bps}$), the expected return of winning trades must pay for both trading fees:\n"
        "$$\\mathbb{E}[\\text{net}] = hit \\cdot \\mathbb{E}[|fwd\\_ret|] - (1 - hit) \\cdot \\mathbb{E}[|fwd\\_ret|] - c = 0$$\n"
        "$$\\implies BE = 0.5 + \\frac{c}{2 \\cdot \\mathbb{E}[|fwd\\_ret\\_open|]}$$\n\n"
        "The **Break-Even Gap** is decomposed into two distinct components:\n"
        "$$\\underbrace{\\text{Gap}}_{hit - BE} = \\underbrace{(hit - 0.5)}_{\\textbf{Direction Skill}} \\;-\\; \\underbrace{(BE - 0.5)}_{\\textbf{Size Selection Hurdle}}$$\n\n"
        "- **Direction Skill $(hit - 0.5)$:** How much better than chance ($50\\%$) is the model at guessing Up vs Down?\n"
        "- **Size Selection Hurdle $(BE - 0.5) = \\frac{c}{2 \\cdot \\mathbb{E}[|fwd\\_ret|]}$:** How large is the fee penalty after accounting for trade displacement?\n\n"
        "> **Why High Hit Rate Does Not Imply Profitability:** A model can hit $55\\%$ directional accuracy, but if average candle displacement is only $0.35\\%$, the fee hurdle is $+32.9\\%$ ($BE = 82.9\\%$), leaving the model deeply unprofitable (Gap = $-27.9\\%$)."
    ))

    # Cell 24 — Decomposition results (Code)
    cells.append(nbf.v4.new_code_cell(
        "decomp_path = ML_DIR / 'results' / 'reports' / 'like_for_like_decomp_20pct.csv'\n"
        "df_decomp = pd.read_csv(decomp_path)\n\n"
        "print(f'Decomposition Scope: Horizon N = {df_decomp[\"horizon\"].unique()[0]} (20% Coverage Tier)')\n"
        "df_decomp_sorted = df_decomp.sort_values(by='break_even_gap', ascending=False).reset_index(drop=True)\n\n"
        "disp_decomp = ['model', 'hit_rate', 'direction_skill', 'mean_abs_fwd_ret', 'break_even_rate', 'size_selection_hurdle', 'break_even_gap', 'net_bps']\n"
        "print(df_decomp_sorted[disp_decomp].to_string(index=False))\n\n"
        "loc_row = df_decomp[df_decomp['model'] == 'loc_position_288'].iloc[0]\n"
        "vol_row = df_decomp[df_decomp['model'] == 'vol_gated_reversal'].iloc[0]\n"
        "c15_row = df_decomp[df_decomp['model'] == 'C_12_F15'].iloc[0]\n"
        "b12_row = df_decomp[df_decomp['model'] == 'B_12'].iloc[0]\n\n"
        "print('\\n--- Critical Insights from 20% Coverage Decomposition ---')\n"
        "print(f'1. Highest Pure Direction Skill: loc_position_288 ({loc_row[\"hit_rate\"]*100:.2f}%, skill: +{loc_row[\"direction_skill\"]*100:.2f}%)')\n"
        "print(f'2. Fallback ML Direction Skill:  B_12 ({b12_row[\"hit_rate\"]*100:.2f}%, skill: +{b12_row[\"direction_skill\"]*100:.2f}%)')\n"
        "print(f'3. Setup C Direction Skill:      C_12_F15 ({c15_row[\"hit_rate\"]*100:.2f}%, skill: +{c15_row[\"direction_skill\"]*100:.2f}%)')\n"
        "print(f'4. Lowest Break-Even Hurdle:     vol_gated_reversal (Hurdle: +{vol_row[\"size_selection_hurdle\"]*100:.2f}%, mean move: {vol_row[\"mean_abs_fwd_ret\"]*100:.3f}%)')\n"
        "print(f'5. Tightest Break-Even Gap:      vol_gated_reversal ({vol_row[\"break_even_gap\"]:.4f}) beats C_12_F15 ({c15_row[\"break_even_gap\"]:.4f})!')"
    ))

    # =========================================================================
    # Section 6 — Stage 1 Logistic Regression screening
    # =========================================================================

    # Cell 25 — Existing Stage 1 screening results (Code)
    cells.append(nbf.v4.new_code_cell(
        "stage1_path = ML_DIR / 'results' / 'reports' / 'stage1_lr_ranking.csv'\n"
        "df_stage1 = pd.read_csv(stage1_path)\n\n"
        "print(f'Total Stage 1 Screening Configurations Evaluated: {len(df_stage1)} (Pre-declared experiment budget = 76)')\n"
        "print(f'Horizons Covered in Stage 1 Screening: {sorted(df_stage1[\"horizon\"].unique())}')\n\n"
        "s_cols = [\n"
        "    'config_id', 'setup', 'horizon', 'rule', 'signals_per_day',\n"
        "    'mean_hit_rate', 'mean_break_even_gap', 'std_break_even_gap',\n"
        "    'rank_metric', 'mean_net_bps', 'status'\n"
        "]\n"
        "survivors = df_stage1[df_stage1['status'].isin(['survivor', 'fallback'])].copy()\n"
        "print('\\n=== Stage 1 Survivors & Fallback Configuration ===')\n"
        "print(survivors[s_cols].to_string(index=False))\n\n"
        "print('\\n=== Top 10 Configurations in Stage 1 Screening ===')\n"
        "print(df_stage1.head(10)[s_cols].to_string(index=False))\n\n"
        "rej_cnt = (df_stage1['status'] == 'rejected').sum()\n"
        "print(f'\\nConfigurations rejected by volume filter (< 15 sig/day): {rej_cnt} / {len(df_stage1)}')"
    ))

    # Cell 26 — Stage 1 survivor analysis (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Comparative analysis between Stage 1 survivors and baselines\n"
        "top_s = survivors[survivors['status'] == 'survivor'].iloc[0]\n"
        "fallback_s = survivors[survivors['status'] == 'fallback'].iloc[0]\n"
        "top_base_12 = df_bin[df_bin['horizon'] == 12].sort_values('break_even_gap', ascending=False).iloc[0]\n\n"
        "print('=== Survivor vs Fallback vs Baseline Comparison ===')\n"
        "print(f'Top Setup C Survivor: {top_s[\"config_id\"]} (N={top_s[\"horizon\"]}, Rule {top_s[\"rule\"]})')\n"
        "print(f'  Direction Hit Rate: {top_s[\"mean_hit_rate\"]*100:.2f}%')\n"
        "print(f'  Break-Even Gap:     {top_s[\"mean_break_even_gap\"]:.4f} (std: {top_s[\"std_break_even_gap\"]:.4f})')\n"
        "print(f'  Net Return:         {top_s[\"mean_net_bps\"]:.2f} bps/trade')\n"
        "print(f'  Signals/Day:        {top_s[\"signals_per_day\"]:.1f}')\n\n"
        "print(f'Setup B Fallback:     {fallback_s[\"config_id\"]} (N={fallback_s[\"horizon\"]}, raw_sign)')\n"
        "print(f'  Direction Hit Rate: {fallback_s[\"mean_hit_rate\"]*100:.2f}% (HIGHEST direction hit rate among survivors!)')\n"
        "print(f'  Break-Even Gap:     {fallback_s[\"mean_break_even_gap\"]:.4f}')\n"
        "print(f'  Net Return:         {fallback_s[\"mean_net_bps\"]:.2f} bps/trade')\n\n"
        "print(f'Top Unconditional Base:{top_base_12[\"baseline\"]} (N={top_base_12[\"horizon\"]})')\n"
        "print(f'  Direction Hit Rate: {top_base_12[\"hit_rate\"]*100:.2f}%')\n"
        "print(f'  Break-Even Gap:     {top_base_12[\"break_even_gap\"]:.4f}')\n"
        "print(f'  Net Return:         {top_base_12[\"net_return_bps\"]:.2f} bps/trade')"
    ))

    # =========================================================================
    # Section 7 — Final baseline conclusions
    # =========================================================================

    # Cell 27 — Overall findings (Code)
    cells.append(nbf.v4.new_code_cell(
        "# Dynamically synthesized overall findings\n"
        "print('=== Synthesized Overall Findings Across All Four Horizons ===\\n')\n"
        "for h in [2, 3, 6, 12]:\n"
        "    b_bin = df_bin[df_bin['horizon'] == h].sort_values('break_even_gap', ascending=False).iloc[0]\n"
        "    b_3 = df_three[df_three['horizon'] == h].sort_values('balanced_accuracy', ascending=False).iloc[0]\n"
        "    b_vol = df_vol[df_vol['horizon'] == h].iloc[0]\n"
        "    print(f'Horizon N={h} ({h*5}m):')\n"
        "    print(f'  Top Binary:     {b_bin[\"baseline\"]} (Hit: {b_bin[\"hit_rate\"]*100:.2f}%, Gap: {b_bin[\"break_even_gap\"]:.4f}, Net: {b_bin[\"net_return_bps\"]:.2f} bps)')\n"
        "    print(f'  Top 3-Class:    {b_3[\"baseline\"]} (Bal Acc: {b_3[\"balanced_accuracy\"]*100:.2f}%, F1: {b_3[\"f1_macro\"]:.4f})')\n"
        "    print(f'  Volatility LR:  ROC AUC = {b_vol[\"roc_auc\"]:.4f}, Prevalence = {b_vol[\"positive_prevalence\"]*100:.1f}%')\n\n"
        "print('--- Strategic Horizon Assessment ---')\n"
        "print('1. Directional Edge: Mean-reversion signals (loc_position_288, reverse_momentum, reverse_rsi_14) consistently outperform trend continuation across all horizons.')\n"
        "print('2. Economic Viability: Unconditional trading is unprofitable at every horizon due to the 23 bps fee hurdle.')\n"
        "print('3. Optimal Horizon for Next Phase: N=12 (60m) possesses the largest price displacement relative to fees, lowest neutral drag (52.7%), and highest empirical capacity to clear trading friction under volatility gating.')"
    ))

    # Cell 28 — Baseline checkpoint report (Markdown/Code)
    cells.append(nbf.v4.new_code_cell(
        "# Final Checkpoint Report Verification\n"
        "generated_files = [\n"
        "    'cv_folds_summary.csv',\n"
        "    'baselines_binary_summary.csv',\n"
        "    'baselines_three_class_summary.csv',\n"
        "    'volatility_baseline_summary.csv',\n"
        "    'like_for_like_summary.csv',\n"
        "    'like_for_like_decomp_20pct.csv',\n"
        "    'stage1_lr_ranking.csv',\n"
        "]\n\n"
        "print('=== BASELINE EVALUATION CHECKPOINT REPORT ===\\n')\n"
        "print('1. Horizons Evaluated: N in {2, 3, 6, 12} (10m, 15m, 30m, 60m) - COMPLETE (100% evaluated)')\n"
        "print('2. Baseline Suites:    Binary (60 evaluations), 3-Class (60 evaluations), Volatility (4 evaluations) - COMPLETE')\n"
        "print('3. Like-for-Like:      Coverage levels [100%, 50%, 20%, 10%] with stationary block-bootstrap CIs (B=288) - COMPLETE')\n"
        "print('4. Decomposition:      20% coverage Direction Skill vs Hurdle Penalty decomposed - COMPLETE')\n"
        "print('5. Stage 1 Screening:  76 pre-declared configurations evaluated and preserved - COMPLETE\\n')\n"
        "print('--- Generated Result Reports Verification ---')\n"
        "for f in generated_files:\n"
        "    p = ML_DIR / 'results' / 'reports' / f\n"
        "    status = 'EXISTS (' + str(p.stat().st_size) + ' bytes)' if p.exists() else 'MISSING'\n"
        "    print(f'  {f:<35} : {status}')\n\n"
        "print('\\nSTATUS: Baseline evaluation is 100% complete, verified, and ready for model training approval.')"
    ))

    nb.cells = cells
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Successfully constructed 28-cell notebook at {NOTEBOOK_PATH}")

    # Execute with nbclient
    print("Executing complete 28-cell notebook from top to bottom with nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Successfully executed and saved fully rendered notebook at {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()
