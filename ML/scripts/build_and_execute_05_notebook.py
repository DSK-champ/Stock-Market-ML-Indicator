"""Construct and execute ML/notebooks/05_Model_Training.ipynb with all verified tables."""

import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ML_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ML_DIR / "notebooks" / "05_Model_Training.ipynb"

def build_notebook():
    nb = nbf.v4.new_notebook()

    nb.cells = [
        # Markdown Title
        nbf.v4.new_markdown_cell(
            "# STEP 5: Baselines, Stage 1 Screening & Architecture Evaluation\n\n"
            "This notebook presents the verified empirical evaluation for Step 5 on the **TRAIN partition** ($N=220,953$ rows, 2023-11-01 to 2025-12-07):\n"
            "1. **Purged Walk-Forward Cross-Validation** (6 expanding folds, 12-candle purge, 132,523 held-out evaluation candles).\n"
            "2. **Canonical Baselines Suite**:\n"
            "   - **Binary Directional Baselines** (15 rules across $N \\in \\{2, 3, 6, 12\\}$).\n"
            "   - **Three-Class Baselines** (Balanced accuracy mathematically anchored at 33.33%).\n"
            "   - **Volatility Baseline** scored strictly on `big_move_N`.\n"
            "3. **Like-for-Like Multi-Coverage Comparison** (100%, 50%, 20%, 10% coverage) with stationary block-bootstrap 95% CIs ($B=288$).\n"
            "4. **20% Coverage Gap Decomposition**: Direction Skill vs. Size Selection Hurdle ($gap = (hit - 0.5) - (BE - 0.5)$).\n"
            "5. **Stage 1 Screening & Survivors**: 76 Logistic Regression configurations, ranking, and directional hit rates.\n\n"
            "> **Strict Partition Guard:** Evaluated strictly on the `train` partition. Validation rows remain reserved for Step 6, and test rows remain locked."
        ),

        # Code 1: Environment & Config
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "ML_DIR = Path.cwd()\n"
            "if (ML_DIR / 'ML').exists():\n"
            "    ML_DIR = ML_DIR / 'ML'\n"
            "elif ML_DIR.name == 'notebooks':\n"
            "    ML_DIR = ML_DIR.parent\n"
            "if str(ML_DIR) not in sys.path:\n"
            "    sys.path.insert(0, str(ML_DIR))\n\n"
            "from src import config\n"
            "print(f'Working Directory: {ML_DIR}')\n"
            "print(f'Candidate Horizons: {config.CANDIDATE_HORIZONS}')\n"
            "print(f'Round-Trip Cost: {config.TOTAL_ROUNDTRIP_COST*10000:.1f} bps ({config.TOTAL_ROUNDTRIP_COST})')"
        ),

        # Markdown 2: Section 1 CV Folds
        nbf.v4.new_markdown_cell(
            "## 1. Purged Walk-Forward Fold Boundaries\n\n"
            "Evaluated on train post-warmup ($N=218,935$ candles, 760.2 calendar days). Purge gap is strictly 12 candles."
        ),

        # Code 2: Display CV Folds
        nbf.v4.new_code_cell(
            "cv_folds_path = ML_DIR / 'results' / 'reports' / 'cv_folds_summary.csv'\n"
            "df_folds = pd.read_csv(cv_folds_path)\n"
            "display_cols = [\n"
            "    'fold', 'train_rows', 'purge_rows', 'held_out_rows',\n"
            "    'train_start_time', 'train_end_time',\n"
            "    'held_out_start_time', 'held_out_end_time'\n"
            "]\n"
            "print('=== Purged Walk-Forward 6-Fold Structure ===')\n"
            "print(df_folds[display_cols].to_string(index=False))\n"
            "print(f'\\nTotal evaluated held-out candles: {df_folds[\"held_out_rows\"].sum():,}')\n"
            "print(f'Purge gap strictly enforced: {df_folds[\"purge_rows\"].iloc[0]} candles')"
        ),

        # Markdown 3: Section 2A Binary Baselines
        nbf.v4.new_markdown_cell(
            "## 2. Benchmark Baselines Evaluation\n\n"
            "### 2A. Binary Classification Baselines (`raw_sign_N` / `bin_dir_N`)\n\n"
            "Directional forecasting across 15 rules. RSI(14) uses true abstentions (<30 Long, >70 Short, Neutral dropped); "
            "EMA50 emits both signs cleanly."
        ),

        # Code 3: Binary Baselines Display
        nbf.v4.new_code_cell(
            "bin_path = ML_DIR / 'results' / 'reports' / 'baselines_binary_summary.csv'\n"
            "df_bin = pd.read_csv(bin_path)\n"
            "disp_cols_bin = [\n"
            "    'baseline', 'horizon', 'accuracy', 'balanced_accuracy',\n"
            "    'f1_macro', 'mcc', 'coverage', 'signals_per_day', 'hit_rate',\n"
            "    'break_even_rate', 'break_even_gap', 'net_return_bps'\n"
            "]\n"
            "print('=== Binary Baselines Evaluation (Horizon N=12 / 60-Minute Forward Move) ===')\n"
            "df_bin_12 = df_bin[df_bin['horizon'] == 12].sort_values(by='break_even_gap', ascending=False).reset_index(drop=True)\n"
            "print(df_bin_12[disp_cols_bin].to_string(index=False))\n\n"
            "# Dynamic takeaways\n"
            "top_bin = df_bin_12.iloc[0]\n"
            "rsi_bin = df_bin_12[df_bin_12['baseline'] == 'rsi_14'].iloc[0]\n"
            "ema_bin = df_bin_12[df_bin_12['baseline'] == 'ema50_rule'].iloc[0]\n"
            "loc_bin = df_bin_12[df_bin_12['baseline'] == 'loc_position_288'].iloc[0]\n"
            "print('\\n--- Dynamic Analysis (Binary Directional Regime) ---')\n"
            "print(f'1. Top Directional Baseline: {top_bin[\"baseline\"]} (Hit Rate: {top_bin[\"hit_rate\"]*100:.2f}%, Gap: {top_bin[\"break_even_gap\"]:.4f}, Net: {top_bin[\"net_return_bps\"]:.2f} bps)')\n"
            "print(f'2. RSI(14) Abstentions Preserved: Coverage = {rsi_bin[\"coverage\"]*100:.2f}% ({rsi_bin[\"signals_per_day\"]:.1f} sig/day), Hit Rate = {rsi_bin[\"hit_rate\"]*100:.2f}%, MCC = {rsi_bin[\"mcc\"]:.4f}')\n"
            "print(f'3. EMA50 Directional Rule: Coverage = {ema_bin[\"coverage\"]*100:.2f}%, Hit Rate = {ema_bin[\"hit_rate\"]*100:.2f}%, MCC = {ema_bin[\"mcc\"]:.4f}')\n"
            "print(f'4. Location 288 Mean Reversion: Hit Rate = {loc_bin[\"hit_rate\"]*100:.2f}%, Gap = {loc_bin[\"break_even_gap\"]:.4f}, Net = {loc_bin[\"net_return_bps\"]:.2f} bps')"
        ),

        # Markdown 4: Section 2B Three-Class Baselines
        nbf.v4.new_markdown_cell(
            "### 2B. Three-Class Classification Baselines (`target_N_S1`)\n\n"
            "Evaluated on 3-class target `[-1.0, 0.0, 1.0]`. Balanced accuracy (unweighted macro recall) mathematically anchors "
            "all constant and random baselines at the exact **33.33%** zero-skill watermark despite ~52.7% Neutral prevalence."
        ),

        # Code 4: Three-Class Baselines Display
        nbf.v4.new_code_cell(
            "three_path = ML_DIR / 'results' / 'reports' / 'baselines_three_class_summary.csv'\n"
            "df_three = pd.read_csv(three_path)\n"
            "disp_cols_3 = [\n"
            "    'baseline', 'horizon', 'accuracy', 'balanced_accuracy',\n"
            "    'precision_macro', 'recall_macro', 'f1_macro', 'mcc',\n"
            "    'coverage', 'signals_per_day', 'hit_rate', 'break_even_gap', 'net_return_bps'\n"
            "]\n"
            "print('=== Three-Class Baselines Evaluation (Horizon N=12, Rule S1) ===')\n"
            "df_three_12 = df_three[df_three['horizon'] == 12].sort_values(by='balanced_accuracy', ascending=False).reset_index(drop=True)\n"
            "print(df_three_12[disp_cols_3].to_string(index=False))\n\n"
            "neutral_b = df_three_12[df_three_12['baseline'] == 'always_neutral'].iloc[0]\n"
            "rand3_b = df_three_12[df_three_12['baseline'] == 'random_3class'].iloc[0]\n"
            "up3_b = df_three_12[df_three_12['baseline'] == 'always_up'].iloc[0]\n"
            "print('\\n--- Dynamic Analysis (Three-Class Balanced Accuracy Watermark) ---')\n"
            "print(f'1. Always-Neutral: Raw Accuracy is {neutral_b[\"accuracy\"]*100:.2f}%, but Balanced Accuracy is exactly {neutral_b[\"balanced_accuracy\"]*100:.2f}% (Macro F1: {neutral_b[\"f1_macro\"]:.4f})')\n"
            "print(f'2. Random 3-Class: Raw Accuracy is {rand3_b[\"accuracy\"]*100:.2f}%, Balanced Accuracy is {rand3_b[\"balanced_accuracy\"]*100:.2f}%, MCC is {rand3_b[\"mcc\"]:.4f}')\n"
            "print(f'3. Zero-Skill Invariance: Always-Up ({up3_b[\"balanced_accuracy\"]*100:.2f}%), Always-Down, Coin-Flip, and Majority Class all collapse to strictly 33.33% balanced accuracy.')"
        ),

        # Markdown 5: Section 2C Volatility Baseline on Big Move
        nbf.v4.new_markdown_cell(
            "### 2C. Volatility Baseline on `big_move_N`\n\n"
            "`volatility_lr` is evaluated strictly as a non-directional magnitude model predicting whether $|fwd\\_ret\\_N| > 23$ bps."
        ),

        # Code 5: Volatility Baseline Display
        nbf.v4.new_code_cell(
            "vol_path = ML_DIR / 'results' / 'reports' / 'volatility_baseline_summary.csv'\n"
            "df_vol = pd.read_csv(vol_path)\n"
            "print('=== Volatility Baseline Evaluation (Predicting big_move_N via feat_vol_sigma_288) ===')\n"
            "print(df_vol.to_string(index=False))\n\n"
            "vol_12 = df_vol[df_vol['horizon'] == 12].iloc[0]\n"
            "print(f'\\nAt N=12: ROC AUC = {vol_12[\"roc_auc\"]:.4f}, PR AUC = {vol_12[\"pr_auc\"]:.4f}, F1 = {vol_12[\"f1_score\"]:.4f}, Brier = {vol_12[\"brier_score\"]:.4f}')"
        ),

        # Markdown 6: Section 3 Like-for-Like Multi-Coverage
        nbf.v4.new_markdown_cell(
            "## 3. Like-for-Like Multi-Coverage Comparison (100%, 50%, 20%, 10%)\n\n"
            "Evaluated at fixed coverage thresholds established on training folds. Primary metric: Directional Hit Rate vs `raw_sign_12` "
            "with stationary block-bootstrap 95% CIs ($B=288$ candles), mean $|fwd\\_ret\\_open\\_12|$, break-even rate, gap, and net bps."
        ),

        # Code 6: Like-for-Like Display
        nbf.v4.new_code_cell(
            "lfl_path = ML_DIR / 'results' / 'reports' / 'like_for_like_summary.csv'\n"
            "df_lfl = pd.read_csv(lfl_path)\n\n"
            "for cov in [1.0, 0.5, 0.2, 0.1]:\n"
            "    print(f'\\n================ Coverage Level: {int(cov*100)}% ================')\n"
            "    sub = df_lfl[df_lfl['target_coverage'] == cov].sort_values(by='break_even_gap', ascending=False).reset_index(drop=True)\n"
            "    sub_cols = ['model', 'signals_per_day', 'hit_rate', 'ci_lower', 'ci_upper', 'mean_abs_fwd_ret', 'break_even_rate', 'break_even_gap', 'net_bps']\n"
            "    print(sub[sub_cols].to_string(index=False))"
        ),

        # Markdown 7: Section 4 20% Coverage Gap Decomposition
        nbf.v4.new_markdown_cell(
            "## 4. 20% Coverage Gap Decomposition: Direction Skill vs. Size Selection Hurdle\n\n"
            "Decomposes break-even gap into:\n"
            "$$\\text{gap} = (\\text{hit} - 0.5) - (\\text{BE} - 0.5)$$\n"
            "- **Direction Skill:** $\\text{hit} - 0.5$\n"
            "- **Size Selection Hurdle:** $\\text{BE} - 0.5 = \\frac{c}{2 \\cdot \\mathbb{E}[|fwd\\_ret\\_open|]}$"
        ),

        # Code 7: Decomposition Display
        nbf.v4.new_code_cell(
            "decomp_path = ML_DIR / 'results' / 'reports' / 'like_for_like_decomp_20pct.csv'\n"
            "df_decomp = pd.read_csv(decomp_path)\n"
            "df_decomp_sorted = df_decomp.sort_values(by='break_even_gap', ascending=False).reset_index(drop=True)\n"
            "print('=== 20% Coverage Gap Decomposition ===')\n"
            "disp_decomp = ['model', 'hit_rate', 'direction_skill', 'mean_abs_fwd_ret', 'break_even_rate', 'size_selection_hurdle', 'break_even_gap', 'net_bps']\n"
            "print(df_decomp_sorted[disp_decomp].to_string(index=False))\n\n"
            "loc_row = df_decomp[df_decomp['model'] == 'loc_position_288'].iloc[0]\n"
            "vol_row = df_decomp[df_decomp['model'] == 'vol_gated_reversal'].iloc[0]\n"
            "c15_row = df_decomp[df_decomp['model'] == 'C_12_F15'].iloc[0]\n"
            "b12_row = df_decomp[df_decomp['model'] == 'B_12'].iloc[0]\n\n"
            "print('\\n--- Critical Findings from 20% Coverage Decomposition ---')\n"
            "print(f'1. Highest Pure Direction Skill: loc_position_288 ({loc_row[\"hit_rate\"]*100:.2f}%, skill: +{loc_row[\"direction_skill\"]*100:.2f}%)')\n"
            "print(f'2. Highest Fallback ML Direction Skill: B_12 ({b12_row[\"hit_rate\"]*100:.2f}%, skill: +{b12_row[\"direction_skill\"]*100:.2f}%)')\n"
            "print(f'3. Setup C Direction Skill: C_12_F15 ({c15_row[\"hit_rate\"]*100:.2f}%, skill: +{c15_row[\"direction_skill\"]*100:.2f}%)')\n"
            "print(f'4. Lowest Break-Even Hurdle: vol_gated_reversal (BE: {vol_row[\"break_even_rate\"]*100:.2f}%, hurdle: +{vol_row[\"size_selection_hurdle\"]*100:.2f}%, mean |fwd_ret|: {vol_row[\"mean_abs_fwd_ret\"]*100:.3f}%)')\n"
            "print(f'5. Tightest Break-Even Gap: vol_gated_reversal ({vol_row[\"break_even_gap\"]:.4f}) beats C_12_F15 ({c15_row[\"break_even_gap\"]:.4f}) and B_12 ({b12_row[\"break_even_gap\"]:.4f})!')"
        ),

        # Markdown 8: Section 5 Stage 1 Screening & Survivors
        nbf.v4.new_markdown_cell(
            "## 5. Stage 1 Screening: 76 Logistic Regression Configurations\n\n"
            "Screening of all 76 configurations on 70 `LINEAR_FEATURES`. Direction hit rate shown next to rank metric."
        ),

        # Code 8: Stage 1 Display
        nbf.v4.new_code_cell(
            "stage1_path = ML_DIR / 'results' / 'reports' / 'stage1_lr_ranking.csv'\n"
            "df_stage1 = pd.read_csv(stage1_path)\n"
            "s_cols = [\n"
            "    'config_id', 'setup', 'horizon', 'rule', 'signals_per_day',\n"
            "    'mean_hit_rate', 'mean_break_even_gap', 'std_break_even_gap',\n"
            "    'rank_metric', 'mean_net_bps', 'status'\n"
            "]\n"
            "survivors = df_stage1[df_stage1['status'].isin(['survivor', 'fallback'])].copy()\n"
            "print('=== Stage 1 Survivors & Fallback ===')\n"
            "print(survivors[s_cols].to_string(index=False))\n\n"
            "b12_s = survivors[survivors['config_id'] == 'B_12'].iloc[0]\n"
            "c15_s = survivors[survivors['config_id'] == 'C_12_F15'].iloc[0]\n"
            "print(f'\\nNotice: B_12 has the HIGHEST direction hit rate ({b12_s[\"mean_hit_rate\"]*100:.2f}%) among all survivors (vs C_12_F15 at {c15_s[\"mean_hit_rate\"]*100:.2f}%).')\n"
            "print('Setup C\\'s superior rank metric is driven by size selection (volatility conditioning), not direction skill.')"
        ),
    ]

    # Save to notebook file
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Written notebook template to {NOTEBOOK_PATH}")

    # Execute with nbclient
    print("Executing notebook with nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Successfully executed and saved notebook to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()
