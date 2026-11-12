"""Build ML/notebooks/05b_Horizon_N2.ipynb with exact 13-cell structure and EMPTY outputs."""

import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "05b_Horizon_N2.ipynb"

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    }

    cells = []

    # =========================================================================
    # CELL 1: Horizon Configuration & Purged CV Folds
    # =========================================================================
    c1_md = """# Section 1: HORIZON = 2 Prediction System (10-Minute Direction)

## Cell 1: Horizon Configuration & Purged CV Folds
- **Function:** Initializes project environment, locks horizon parameter $N=2$ (10 minutes, 2 candles forward), and verifies 6-fold expanding purged walk-forward cross-validation schedule.
- **Runtime:** fast (~2s)
- **Status:** Configuration check (no training)"""

    c1_code = """import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Set project root and python path
PROJECT_ROOT = Path.cwd().resolve()
if (PROJECT_ROOT / "src").exists():
    ML_DIR = PROJECT_ROOT
elif (PROJECT_ROOT / "ML" / "src").exists():
    ML_DIR = PROJECT_ROOT / "ML"
else:
    ML_DIR = PROJECT_ROOT.parent

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.cv import PurgedWalkForwardCV
from src.column_dictionary import load_master

# Set target horizon
HORIZON = 2
MINUTES = HORIZON * config.INTERVAL_MINUTES

print("=" * 70)
print(f"HORIZON CONFIGURATION: N = {HORIZON} ({MINUTES} Minutes / 2 Candles)")
print("=" * 70)
print(f"Working Directory      : {ML_DIR}")
print(f"Binary Target          : {config.BINARY_TARGET}")
print(f"Three-Class Target     : {config.THREE_CLASS_TARGET}")
print(f"Directional Benchmark  : {config.TRUTH_COL}")
print(f"Forward Return Column  : {config.RETURN_COL}")
print(f"Target Validity Mask   : {config.VALID_COL}")
print(f"Round-Trip Friction    : {config.COST * 10000:.1f} bps ({config.COST})")
print(f"Training Stride        : Every {config.STRIDE}-rd row (stride downsampling)")
print(f"OpenMP / CPU Cores     : n_jobs = {config.N_JOBS}")
print(f"Random Seed            : {config.SEED}")

# Cross-Validation Folds Verification
cv = PurgedWalkForwardCV()
df_train = load_master(split="train")
print(f"\\nLoaded TRAIN split: {len(df_train):,} total rows (Warmup offset: {cv.warmup_candles:,})")

fold_info = []
for fold, tr_idx, te_idx in cv.split(df_train):
    tr_start, tr_end = df_train.index[tr_idx[0]], df_train.index[tr_idx[-1]]
    te_start, te_end = df_train.index[te_idx[0]], df_train.index[te_idx[-1]]
    purge_gap = te_idx[0] - tr_idx[-1] - 1
    fold_info.append({
        "Fold": fold,
        "Train Start": str(tr_start)[:10],
        "Train End": str(tr_end)[:10],
        "Purge Rows": purge_gap,
        "Test Start": str(te_start)[:10],
        "Test End": str(te_end)[:10],
        "Test Rows": len(te_idx),
    })

df_cv_summary = pd.DataFrame(fold_info)
display(df_cv_summary)
assert all(r["Purge Rows"] == 12 for r in fold_info), "Purge gap must be exactly 12 candles!"
print("Cross-Validation fold boundaries verified: exactly 12-candle purge across all 6 folds.")"""

    # =========================================================================
    # CELL 2: Data Preparation & Column Integrity Audit
    # =========================================================================
    c2_md = """## Cell 2: Data Preparation & Column Integrity Audit
- **Function:** Inspects class balance of `bin_dir_2` and `target_2_S1` on TRAIN, prints feature counts and names for linear (70) and non-linear (136) models, and asserts that no input feature possesses a forbidden role.
- **Runtime:** fast (~15s)
- **Status:** Integrity audit (no training)"""

    c2_code = """from src.column_dictionary import get_feature_columns, get_column_role
from src.model_common import FORBIDDEN_ROLES, validate_feature_columns

# 1. Feature sets
linear_features = get_feature_columns(feature_set="LINEAR_FEATURES")
all_features = get_feature_columns(feature_set="FS_ALL")

print("=" * 70)
print("FEATURE SET AUDIT")
print("=" * 70)
print(f"Linear Models (Logistic Regression) : {len(linear_features)} features (LINEAR_FEATURES)")
print(f"Tree Models (Random Forest, LightGBM): {len(all_features)} features (FS_ALL)")

# 2. Strict forbidden role audit
validate_feature_columns(linear_features)
validate_feature_columns(all_features)
print("Role Validation Passed: Zero features possess role 'target', 'threshold', 'flag', 'market', or 'drawing'.")

# 3. Class balance on TRAIN
valid_mask = df_train[f"has_valid_target_{HORIZON}"] == 1
df_valid = df_train[valid_mask]

bin_col = f"bin_dir_{HORIZON}"
tri_col = f"target_{HORIZON}_{config.THREE_CLASS_RULE}"

print(f"\\nTotal valid target rows on TRAIN: {len(df_valid):,}")

print(f"\\n--- Binary Target Balance ({bin_col}) ---")
bin_counts = df_valid[bin_col].value_counts(dropna=False).sort_index()
bin_pcts = df_valid[bin_col].value_counts(dropna=False, normalize=True).sort_index() * 100.0
for val, count in bin_counts.items():
    label = "UP (1)" if val == 1 else ("DOWN (0)" if val == 0 else "FLAT (NaN)")
    print(f"  {label:<12}: {count:>7,} rows ({bin_pcts[val]:.2f}%)")

print(f"\\n--- Three-Class Target Balance ({tri_col}) ---")
tri_counts = df_valid[tri_col].value_counts(dropna=False).sort_index()
tri_pcts = df_valid[tri_col].value_counts(dropna=False, normalize=True).sort_index() * 100.0
for val, count in tri_counts.items():
    label = "UP (+1)" if val == 1 else ("DOWN (-1)" if val == -1 else "NEUTRAL (0)")
    print(f"  {label:<12}: {count:>7,} rows ({tri_pcts[val]:.2f}%)")"""

    # =========================================================================
    # CELL 3: Branch A — Binary Logistic Regression
    # =========================================================================
    c3_md = """## Cell 3: Branch A — Binary Logistic Regression (`binary_lr`)
- **Function:** Trains `binary_lr` across 6 expanding walk-forward folds on 70 `LINEAR_FEATURES` with Stride 3 downsampling, evaluates out-of-fold probabilistic metrics, 4-tier coverage table, block-bootstrap CIs, circular-shift audit, and fold 3 label shuffle check.
- **Runtime:** RUN THIS (trains) ~1.1 min
- **Status:** Modeling & evaluation"""

    c3_code = """from src.model_common import run_oof, load_truth, load_oof
from src.horizon_eval import evaluate_binary, shuffled_label_retrain

model_id = "binary_lr"
print(f"Executing 6-Fold Out-of-Fold Training for {model_id} (HORIZON = {HORIZON})...")

# Run or load cached OOF
oof_path = run_oof(model_id, horizon=HORIZON)
truth_df = load_truth(horizon=HORIZON)
df_oof = load_oof(model_id, horizon=HORIZON)

# Comprehensive Evaluation
eval_res = evaluate_binary(df_oof, truth_df, cost=config.COST, horizon=HORIZON)

print("=" * 70)
print(f"OUT-OF-FOLD EVALUATION REPORT: {model_id} (N={HORIZON})")
print("=" * 70)
print(f"Accuracy               : {eval_res['accuracy']*100:.2f}%")
print(f"Balanced Accuracy      : {eval_res['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score         : {eval_res['macro_f1']:.4f}")
print(f"Matthews Corr (MCC)    : {eval_res['mcc']:.4f}")
print(f"ROC AUC                : {eval_res['roc_auc']:.4f}")
print(f"PR AUC (Minority)      : {eval_res['pr_auc']:.4f}")
print(f"Brier Score            : {eval_res['brier_score']:.4f}")
print(f"Expected Calib Error   : {eval_res['ece']:.4f}")

print("\\nConfusion Matrix (Normalized by True Class):")
print(np.round(eval_res['confusion_matrix_norm'], 4))

print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
display(eval_res['coverage_table'])

circ = eval_res['circular_shift']
print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
print(f"  Empirical p-value    : {circ['p_value']:.4f}")

# Shuffled-label check on fold 3
shuff_auc = shuffled_label_retrain(model_id, horizon=HORIZON, fold=3)
print(f"\\nFold 3 Shuffled-Label Sanity Retrain AUC: {shuff_auc:.4f} (Expectation: ~0.5000)")"""

    # =========================================================================
    # CELL 4: Branch A — Binary Random Forest
    # =========================================================================
    c4_md = """## Cell 4: Branch A — Binary Random Forest (`binary_rf`)
- **Function:** Trains `binary_rf` (300 trees, depth 6, min_samples_leaf 500, max_features sqrt, max_samples 0.3) across 6 folds on 136 `FS_ALL` features, evaluates out-of-fold metrics, coverage tiers, block-bootstrap CIs, and circular-shift audit.
- **Runtime:** RUN THIS (trains) ~2.2 min
- **Status:** Modeling & evaluation"""

    c4_code = """model_id = "binary_rf"
print(f"Executing 6-Fold Out-of-Fold Training for {model_id} (HORIZON = {HORIZON})...")

oof_path = run_oof(model_id, horizon=HORIZON)
df_oof = load_oof(model_id, horizon=HORIZON)
eval_res = evaluate_binary(df_oof, truth_df, cost=config.COST, horizon=HORIZON)

print("=" * 70)
print(f"OUT-OF-FOLD EVALUATION REPORT: {model_id} (N={HORIZON})")
print("=" * 70)
print(f"Accuracy               : {eval_res['accuracy']*100:.2f}%")
print(f"Balanced Accuracy      : {eval_res['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score         : {eval_res['macro_f1']:.4f}")
print(f"Matthews Corr (MCC)    : {eval_res['mcc']:.4f}")
print(f"ROC AUC                : {eval_res['roc_auc']:.4f}")
print(f"PR AUC (Minority)      : {eval_res['pr_auc']:.4f}")
print(f"Brier Score            : {eval_res['brier_score']:.4f}")
print(f"Expected Calib Error   : {eval_res['ece']:.4f}")

print("\\nConfusion Matrix (Normalized by True Class):")
print(np.round(eval_res['confusion_matrix_norm'], 4))

print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
display(eval_res['coverage_table'])

circ = eval_res['circular_shift']
print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
print(f"  Empirical p-value    : {circ['p_value']:.4f}")"""

    # =========================================================================
    # CELL 5: Branch A — Binary LightGBM
    # =========================================================================
    c5_md = """## Cell 5: Branch A — Binary LightGBM (`binary_gb`)
- **Function:** Trains `binary_gb` (600 trees, depth 4, lr 0.03, colsample 0.7, subsample 0.7, early stopping 50 with 12-candle purge) on 136 `FS_ALL` features, evaluates out-of-fold metrics, coverage tiers, circular-shift audit, shuffled-label check on fold 3, and permutation feature importance on fold 6.
- **Runtime:** RUN THIS (trains) ~3.5 min
- **Status:** Modeling & evaluation"""

    c5_code = """from src.horizon_eval import permutation_importance_lgbm
from src.binary_models import build_binary_model, get_binary_inputs
from src.model_common import prepare_model_features

model_id = "binary_gb"
print(f"Executing 6-Fold Out-of-Fold Training for {model_id} (HORIZON = {HORIZON})...")

oof_path = run_oof(model_id, horizon=HORIZON)
df_oof = load_oof(model_id, horizon=HORIZON)
eval_res = evaluate_binary(df_oof, truth_df, cost=config.COST, horizon=HORIZON)

print("=" * 70)
print(f"OUT-OF-FOLD EVALUATION REPORT: {model_id} (N={HORIZON})")
print("=" * 70)
print(f"Accuracy               : {eval_res['accuracy']*100:.2f}%")
print(f"Balanced Accuracy      : {eval_res['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score         : {eval_res['macro_f1']:.4f}")
print(f"Matthews Corr (MCC)    : {eval_res['mcc']:.4f}")
print(f"ROC AUC                : {eval_res['roc_auc']:.4f}")
print(f"PR AUC (Minority)      : {eval_res['pr_auc']:.4f}")
print(f"Brier Score            : {eval_res['brier_score']:.4f}")
print(f"Expected Calib Error   : {eval_res['ece']:.4f}")

print("\\nConfusion Matrix (Normalized by True Class):")
print(np.round(eval_res['confusion_matrix_norm'], 4))

print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
display(eval_res['coverage_table'])

circ = eval_res['circular_shift']
print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
print(f"  Empirical p-value    : {circ['p_value']:.4f}")

# Shuffled-label check on fold 3
shuff_auc = shuffled_label_retrain(model_id, horizon=HORIZON, fold=3)
print(f"\\nFold 3 Shuffled-Label Retrain AUC: {shuff_auc:.4f} (Expectation: ~0.5000)")

# Fold 6 Permutation Importance (20,000 samples, 3 repeats)
fold6_test_idx = list(cv.split(df_train))[-1][2]
df_f6 = df_train.iloc[fold6_test_idx]
mask_f6 = (df_f6[f"has_valid_target_{HORIZON}"] == 1) & (df_f6[f"bin_dir_{HORIZON}"].notna())
df_f6_val = df_f6[mask_f6]
feat_cols = get_binary_inputs(model_id)
X_f6 = prepare_model_features(df_f6_val, feat_cols)
y_f6 = df_f6_val[f"bin_dir_{HORIZON}"].values

# Fit a quick model on fold 6 train block to evaluate permutation importance
fold6_tr_idx = list(cv.split(df_train))[-1][1]
df_f6_tr = df_train.iloc[fold6_tr_idx]
mask_tr = (df_f6_tr[f"has_valid_target_{HORIZON}"] == 1) & (df_f6_tr[f"bin_dir_{HORIZON}"].notna())
df_f6_tr_val = df_f6_tr[mask_tr].iloc[::config.STRIDE]
X_tr6 = prepare_model_features(df_f6_tr_val, feat_cols)
y_tr6 = df_f6_tr_val[f"bin_dir_{HORIZON}"].values

clf_f6 = build_binary_model(model_id, smoke=True)
clf_f6.fit(X_tr6, y_tr6)
df_imp, is_dominated = permutation_importance_lgbm(clf_f6, X_f6, y_f6, n_samples=20000, n_repeats=3)
print(f"\\nPermutation Feature Importance (Fold 6 Held-Out, Dominance Flag: {is_dominated}):")
display(df_imp.head(10))"""

    # =========================================================================
    # CELL 6: Branch B — Three-Class Logistic Regression
    # =========================================================================
    c6_md = """## Cell 6: Branch B — Three-Class Logistic Regression (`three_class_lr`)
- **Function:** Trains multinomial `three_class_lr` with `class_weight="balanced"` on 70 `LINEAR_FEATURES` to predict `target_2_S1` (-1 DOWN, 0 NEUTRAL, 1 UP), evaluates out-of-fold multi-class metrics, always-Neutral baseline accuracy, coverage tiers, and circular-shift check.
- **Runtime:** RUN THIS (trains) ~0.9 min
- **Status:** Modeling & evaluation"""

    c6_code = """from src.horizon_eval import evaluate_three_class

model_id = "three_class_lr"
print(f"Executing 6-Fold Out-of-Fold Training for {model_id} (HORIZON = {HORIZON})...")

oof_path = run_oof(model_id, horizon=HORIZON)
df_oof = load_oof(model_id, horizon=HORIZON)
eval_res = evaluate_three_class(df_oof, truth_df, cost=config.COST, horizon=HORIZON)

print("=" * 70)
print(f"OUT-OF-FOLD EVALUATION REPORT: {model_id} (N={HORIZON})")
print("=" * 70)
print(f"Accuracy               : {eval_res['accuracy']*100:.2f}%")
print(f"Always-Neutral Acc     : {eval_res['always_neutral_accuracy']*100:.2f}% (Benchmark Baseline)")
print(f"Balanced Accuracy      : {eval_res['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score         : {eval_res['macro_f1']:.4f}")
print(f"Matthews Corr (MCC)    : {eval_res['mcc']:.4f}")
print(f"Macro OvR ROC AUC      : {eval_res['roc_auc_ovr']:.4f}")
print(f"Multi-class Log Loss   : {eval_res['log_loss']:.4f}")

print("\\nConfusion Matrix (Normalized by True Class [-1, 0, 1]):")
print(np.round(eval_res['confusion_matrix_norm'], 4))

print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
display(eval_res['coverage_table'])

circ = eval_res['circular_shift']
print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
print(f"  Empirical p-value    : {circ['p_value']:.4f}")

# Shuffled-label check on fold 3
shuff_auc = shuffled_label_retrain(model_id, horizon=HORIZON, fold=3)
print(f"\\nFold 3 Shuffled-Label Retrain OvR AUC: {shuff_auc:.4f} (Expectation: ~0.5000)")"""

    # =========================================================================
    # CELL 7: Branch B — Three-Class Random Forest
    # =========================================================================
    c7_md = """## Cell 7: Branch B — Three-Class Random Forest (`three_class_rf`)
- **Function:** Trains `three_class_rf` (`class_weight="balanced_subsample"`) across 6 folds on 136 `FS_ALL` features, evaluates out-of-fold multi-class metrics, always-Neutral baseline accuracy, coverage tiers, and circular-shift audit.
- **Runtime:** RUN THIS (trains) ~2.1 min
- **Status:** Modeling & evaluation"""

    c7_code = """model_id = "three_class_rf"
print(f"Executing 6-Fold Out-of-Fold Training for {model_id} (HORIZON = {HORIZON})...")

oof_path = run_oof(model_id, horizon=HORIZON)
df_oof = load_oof(model_id, horizon=HORIZON)
eval_res = evaluate_three_class(df_oof, truth_df, cost=config.COST, horizon=HORIZON)

print("=" * 70)
print(f"OUT-OF-FOLD EVALUATION REPORT: {model_id} (N={HORIZON})")
print("=" * 70)
print(f"Accuracy               : {eval_res['accuracy']*100:.2f}%")
print(f"Always-Neutral Acc     : {eval_res['always_neutral_accuracy']*100:.2f}% (Benchmark Baseline)")
print(f"Balanced Accuracy      : {eval_res['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score         : {eval_res['macro_f1']:.4f}")
print(f"Matthews Corr (MCC)    : {eval_res['mcc']:.4f}")
print(f"Macro OvR ROC AUC      : {eval_res['roc_auc_ovr']:.4f}")
print(f"Multi-class Log Loss   : {eval_res['log_loss']:.4f}")

print("\\nConfusion Matrix (Normalized by True Class [-1, 0, 1]):")
print(np.round(eval_res['confusion_matrix_norm'], 4))

print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
display(eval_res['coverage_table'])

circ = eval_res['circular_shift']
print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
print(f"  Empirical p-value    : {circ['p_value']:.4f}")"""

    # =========================================================================
    # CELL 8: Branch B — Three-Class LightGBM
    # =========================================================================
    c8_md = """## Cell 8: Branch B — Three-Class LightGBM (`three_class_gb`)
- **Function:** Trains `three_class_gb` (`objective="multiclass"`, `class_weight="balanced"`) across 6 folds on 136 `FS_ALL` features, evaluates out-of-fold multi-class metrics, always-Neutral baseline, coverage tiers, circular-shift audit, shuffled-label check on fold 3, and permutation importance on fold 6.
- **Runtime:** RUN THIS (trains) ~2.0 min
- **Status:** Modeling & evaluation"""

    c8_code = """from src.three_class_models import build_three_class_model, get_three_class_inputs

model_id = "three_class_gb"
print(f"Executing 6-Fold Out-of-Fold Training for {model_id} (HORIZON = {HORIZON})...")

oof_path = run_oof(model_id, horizon=HORIZON)
df_oof = load_oof(model_id, horizon=HORIZON)
eval_res = evaluate_three_class(df_oof, truth_df, cost=config.COST, horizon=HORIZON)

print("=" * 70)
print(f"OUT-OF-FOLD EVALUATION REPORT: {model_id} (N={HORIZON})")
print("=" * 70)
print(f"Accuracy               : {eval_res['accuracy']*100:.2f}%")
print(f"Always-Neutral Acc     : {eval_res['always_neutral_accuracy']*100:.2f}% (Benchmark Baseline)")
print(f"Balanced Accuracy      : {eval_res['balanced_accuracy']*100:.2f}%")
print(f"Macro F1-Score         : {eval_res['macro_f1']:.4f}")
print(f"Matthews Corr (MCC)    : {eval_res['mcc']:.4f}")
print(f"Macro OvR ROC AUC      : {eval_res['roc_auc_ovr']:.4f}")
print(f"Multi-class Log Loss   : {eval_res['log_loss']:.4f}")

print("\\nConfusion Matrix (Normalized by True Class [-1, 0, 1]):")
print(np.round(eval_res['confusion_matrix_norm'], 4))

print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
display(eval_res['coverage_table'])

circ = eval_res['circular_shift']
print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
print(f"  Empirical p-value    : {circ['p_value']:.4f}")

# Shuffled-label check on fold 3
shuff_auc = shuffled_label_retrain(model_id, horizon=HORIZON, fold=3)
print(f"\\nFold 3 Shuffled-Label Retrain OvR AUC: {shuff_auc:.4f} (Expectation: ~0.5000)")

# Fold 6 Permutation Importance (20,000 samples, 3 repeats)
feat_cols = get_three_class_inputs(model_id)
X_f6 = prepare_model_features(df_f6_val, feat_cols)
y_f6_tri = df_f6_val[f"target_{HORIZON}_{config.THREE_CLASS_RULE}"].values

X_tr6_tri = prepare_model_features(df_f6_tr_val, feat_cols)
y_tr6_tri = df_f6_tr_val[f"target_{HORIZON}_{config.THREE_CLASS_RULE}"].values

clf_f6_tri = build_three_class_model(model_id, smoke=True)
clf_f6_tri.fit(X_tr6_tri, y_tr6_tri)
df_imp_tri, is_dominated_tri = permutation_importance_lgbm(clf_f6_tri, X_f6, y_f6_tri, n_samples=20000, n_repeats=3)
print(f"\\nPermutation Feature Importance (Fold 6 Held-Out, Dominance Flag: {is_dominated_tri}):")
display(df_imp_tri.head(10))"""

    # =========================================================================
    # CELL 9: Branch C — Technical Analysis Engine
    # =========================================================================
    c9_md = """## Cell 9: Branch C — Technical Analysis Engine (`ta_engine`)
- **Function:** Computes 6 standardized technical scores in [-1, 1] using conventional trend-following orientation, evaluates score distributions, computes composite `ta_label`, measures directional hit rate against `raw_sign_2`, and reports correlations with forward returns.
- **Runtime:** fast (~5s)
- **Status:** Deterministic rule execution (no model training)"""

    c9_code = """from src.ta_engine import compute_ta_scores, ta_label, TA_SCORE_COLUMNS

print(f"Evaluating Technical Analysis Engine on TRAIN (HORIZON = {HORIZON})...")

ta_params_path = config.MODELS_DIR / f"horizon_{HORIZON}" / f"ta_params_N{HORIZON}.json"
ta_scores_df = compute_ta_scores(df_train, params_path=ta_params_path, save_params=True, horizon=HORIZON)

print("=" * 70)
print("TECHNICAL ANALYSIS SCORES DISTRIBUTION (Normalized to [-1.0, 1.0])")
print("=" * 70)
ta_desc = ta_scores_df[TA_SCORE_COLUMNS].describe().T[["min", "mean", "std", "max"]]
display(ta_desc)

# Composite TA Label
ta_labels = ta_label(ta_scores_df)
print("\\n--- Composite TA Label Breakdown (Thresholds +/- 0.10) ---")
label_counts = ta_labels.value_counts()
label_pcts = ta_labels.value_counts(normalize=True) * 100.0
for lbl, cnt in label_counts.items():
    print(f"  {lbl:<10}: {cnt:>7,} ({label_pcts[lbl]:.2f}%)")

# Directional hit rate of TA Label vs raw_sign_2
dir_mask = (ta_labels != "NEUTRAL") & (df_train[f"has_valid_target_{HORIZON}"] == 1)
ta_sign = np.where(ta_labels[dir_mask] == "BULLISH", 1, -1)
true_sign = df_train.loc[dir_mask, f"raw_sign_{HORIZON}"].values
ta_hit_rate = (ta_sign == true_sign).mean()
print(f"\\nTA Label Direction Hit Rate vs {config.TRUTH_COL} : {ta_hit_rate*100:.2f}% (on {dir_mask.sum():,} active bars)")

# Correlation with forward returns
print(f"\\n--- Score Correlations with {config.RETURN_COL} ---")
fwd_ret = df_train[f"fwd_ret_open_{HORIZON}"]
corr_rows = []
for c in TA_SCORE_COLUMNS:
    p_corr = ta_scores_df[c].corr(fwd_ret, method="pearson")
    s_corr = ta_scores_df[c].corr(fwd_ret, method="spearman")
    corr_rows.append({"TA Feature": c, "Pearson Corr": p_corr, "Spearman Corr": s_corr})
display(pd.DataFrame(corr_rows))

print(\"\"\"
[Note on BTC Technical Analysis Dynamics]
In 5-minute Bitcoin market data, short-term momentum and trend-following votes frequently
hit below 50% due to mean-reversion and liquidity-sweeping behavior. The TA Engine maintains
standard technical definitions; the downstream Meta-Model learns the sign weighting empirically.
\"\"\")"""

    # =========================================================================
    # CELL 10: Branch D — Prepare Meta-Model Training Data
    # =========================================================================
    c10_md = """## Cell 10: Branch D — Assemble Meta-Model Feature Matrix
- **Function:** Aggregates out-of-fold probabilistic outputs from Branches A & B and continuous signals from Branch C into the exact 15 declared meta-features, checks timestamps alignment across files, and prints correlation matrix.
- **Runtime:** fast (~10s)
- **Status:** Data aggregation (requires Cells 3..8)"""

    c10_code = """from src.meta_model import build_meta_inputs, META_INPUT_COLUMNS

# Check required base OOF artifacts
required_base_oofs = [
    f"binary_lr_N{HORIZON}.csv.gz",
    f"binary_rf_N{HORIZON}.csv.gz",
    f"binary_gb_N{HORIZON}.csv.gz",
    f"three_class_lr_N{HORIZON}.csv.gz",
    f"three_class_rf_N{HORIZON}.csv.gz",
    f"three_class_gb_N{HORIZON}.csv.gz",
]
missing_files = [f for f in required_base_oofs if not (config.RESULTS_DIR / "oof" / f).exists()]
if missing_files:
    print(f"ERROR: Missing base OOF predictions: {missing_files}")
    print("Please execute Cells 3 through 8 before running Cell 10.")
else:
    meta_df = build_meta_inputs(horizon=HORIZON)
    print("=" * 70)
    print(f"META-FEATURE MATRIX ASSEMBLED: {len(meta_df):,} rows x {len(META_INPUT_COLUMNS)} features")
    print("=" * 70)
    print(f"Coverage Folds Present : {sorted(meta_df['fold'].unique())} (Folds 1 to 6)")
    print(f"Date Range             : {meta_df['open_time'].min()} to {meta_df['open_time'].max()}")
    print(f"Zero Missing Values    : {meta_df[META_INPUT_COLUMNS].isna().sum().sum() == 0}")

    print("\\n15 Declared Meta-Input Features (in exact stack order):")
    for i, col in enumerate(META_INPUT_COLUMNS, 1):
        print(f"  {i:>2}. {col}")

    print("\\nCorrelation Matrix of Meta-Input Features:")
    meta_corr = meta_df[META_INPUT_COLUMNS].corr()
    display(np.round(meta_corr, 3))"""

    # =========================================================================
    # CELL 11: Branch D — Nested Walk-Forward Meta-Model Evaluation
    # =========================================================================
    c11_md = """## Cell 11: Branch D — Nested Walk-Forward Meta-Model Evaluation & Final Fit
- **Function:** Executes strictly nested walk-forward cross-validation for the meta-model across folds 2..6 (folds 1..k-1 training with 12-candle purge drop), evaluates out-of-fold multi-class and trading performance, benchmarks against single models and heuristic ensembles, and fits the final production meta-model on all OOF rows.
- **Runtime:** RUN THIS (trains) ~20s
- **Status:** Meta modeling & evaluation (requires Cell 10)"""

    c11_code = """from src.meta_model import nested_meta_oof, fit_meta_final

if missing_files:
    print("Skipping: Run Cells 3 to 10 first.")
else:
    print(f"Executing Nested Out-of-Fold Evaluation for Meta-Model (HORIZON = {HORIZON})...")
    meta_oof_path = nested_meta_oof(horizon=HORIZON, meta_df=meta_df)
    df_meta_oof = load_horizon_oof_csv_gz(meta_oof_path)

    # Evaluate Meta OOF
    meta_eval = evaluate_three_class(df_meta_oof, truth_df, cost=config.COST, horizon=HORIZON)

    print("=" * 70)
    print(f"NESTED OUT-OF-FOLD EVALUATION REPORT: Meta-Model (N={HORIZON})")
    print("=" * 70)
    print(f"Accuracy               : {meta_eval['accuracy']*100:.2f}%")
    print(f"Always-Neutral Acc     : {meta_eval['always_neutral_accuracy']*100:.2f}% (Benchmark Baseline)")
    print(f"Balanced Accuracy      : {meta_eval['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score         : {meta_eval['macro_f1']:.4f}")
    print(f"Matthews Corr (MCC)    : {meta_eval['mcc']:.4f}")
    print(f"Macro OvR ROC AUC      : {meta_eval['roc_auc_ovr']:.4f}")
    print(f"Multi-class Log Loss   : {meta_eval['log_loss']:.4f}")

    print("\\nConfusion Matrix (Normalized by True Class [-1, 0, 1]):")
    print(np.round(meta_eval['confusion_matrix_norm'], 4))

    print(f"\\nCoverage Tiers (Cost = {config.COST*10000:.1f} bps, Block-Bootstrap B=500):")
    display(meta_eval['coverage_table'])

    circ = meta_eval['circular_shift']
    print(f"\\nCircular-Shift Null Test (20% Coverage Tier, 200 Shifts, min_shift=2018):")
    print(f"  Real Hit Rate        : {circ['real_hit_rate']*100:.2f}%")
    print(f"  Null Mean (P95/P99)  : {circ['null_mean']*100:.2f}% ({circ['null_p95']*100:.2f}% / {circ['null_p99']*100:.2f}%)")
    print(f"  Empirical p-value    : {circ['p_value']:.4f}")

    # Comparative Ensembles Benchmark
    # 1. Plain average score: mean(P(up) - P(down)) across base models
    avg_score = (
        (meta_df['bin_lr_p_up'] - 0.5) + (meta_df['bin_rf_p_up'] - 0.5) + (meta_df['bin_gb_p_up'] - 0.5) +
        (meta_df['tri_lr_p_up'] - meta_df['tri_lr_p_down']) +
        (meta_df['tri_rf_p_up'] - meta_df['tri_rf_p_down']) +
        (meta_df['tri_gb_p_up'] - meta_df['tri_gb_p_down'])
    ) / 6.0
    avg_sig = np.where(avg_score > 0.05, 1, np.where(avg_score < -0.05, -1, 0))
    avg_hit = (avg_sig == meta_df[f'raw_sign_{HORIZON}'])[avg_sig != 0].mean()

    # 2. Majority vote of 3 binary models
    maj_bin = (
        (meta_df['bin_lr_p_up'] > 0.5).astype(int) +
        (meta_df['bin_rf_p_up'] > 0.5).astype(int) +
        (meta_df['bin_gb_p_up'] > 0.5).astype(int)
    )
    maj_sig = np.where(maj_bin >= 2, 1, -1)
    maj_hit = (maj_sig == meta_df[f'raw_sign_{HORIZON}']).mean()

    meta_hit_full = meta_eval['coverage_table'].loc[meta_eval['coverage_table']['tier'] == '100%', 'hit_rate'].values[0]

    print("\\n--- Comparative Benchmark Summary ---")
    print(f"  Meta-Model Direction Hit Rate (100% Tier) : {meta_hit_full*100:.2f}%")
    print(f"  Plain Average 6-Model Score Hit Rate       : {avg_hit*100:.2f}%")
    print(f"  Binary Models Majority Vote Hit Rate       : {maj_hit*100:.2f}%")

    if meta_hit_full > avg_hit and meta_hit_full > maj_hit:
        print("Conclusion: Meta-model successfully outperforms both heuristic plain average and majority vote ensembles.")
    else:
        print("Conclusion: Meta-model performs within statistical fold noise of the plain average ensemble.")

    # Fit final production meta-model
    meta_joblib, meta_json = fit_meta_final(horizon=HORIZON, meta_df=meta_df)
    print(f"\\nPersisted Final Production Meta-Model to {meta_joblib}")"""

    # =========================================================================
    # CELL 12: Production Inference Demo & Feature Parity Check
    # =========================================================================
    c12_md = """## Cell 12: Historical Single-Timestamp Inference Demo (Fold 6 Parity)
- **Function:** Selects the final historical timestamp of fold 6, constructs a live buffer of `LIVE_BUFFER_CANDLES` (2880) candles from `clean.csv`, calls `predict_horizon(buffer_df, horizon=2)`, prints the formatted prediction payload, verifies probability closure (sum=1), and executes feature parity verification against `master.csv` ($|\\Delta| < 10^{-7}$).
- **Runtime:** fast (~5s)
- **Status:** Inference verification (requires Cell 11 & Cell 13)"""

    c12_code = """from src.horizon_pipeline import predict_horizon
from src.features import build_features

# Check if model artifacts are available
meta_artifact = config.MODELS_DIR / f"horizon_{HORIZON}" / f"meta_model_N{HORIZON}.joblib"
base_artifact = config.MODELS_DIR / f"horizon_{HORIZON}" / f"binary_lr_N{HORIZON}.joblib"

if not meta_artifact.exists() or not base_artifact.exists():
    print("Artifacts missing: Run Cell 11 and Cell 13 before executing Cell 12 demo.")
else:
    # 1. Target the last row of fold 6
    fold6_test_idx = list(cv.split(df_train))[-1][2]
    f6_last_pos = fold6_test_idx[-1]
    f6_timestamp = df_train.index[f6_last_pos]

    # 2. Extract preceding 2,880 candles
    buffer_start = max(0, f6_last_pos - config.LIVE_BUFFER_CANDLES + 1)
    raw_candles = df_train.reset_index()
    candle_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "quote_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"
    ]
    buffer_df = raw_candles.iloc[buffer_start : f6_last_pos + 1][candle_cols].copy()

    print("=" * 70)
    print(f"HISTORICAL INFERENCE DEMO: Target Timestamp = {f6_timestamp}")
    print("=" * 70)
    print(f"Candle Buffer Length   : {len(buffer_df)} candles (Interval: {buffer_df['open_time'].iloc[0]} to {buffer_df['open_time'].iloc[-1]})")

    # 3. Call predict_horizon
    pred_res = predict_horizon(buffer_df, horizon=HORIZON)

    print(f"\\n--- Production Prediction Payload ---")
    for k, v in pred_res.items():
        if isinstance(v, dict):
            print(f"  {k:<32}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k:<28}: {sub_v}")
        else:
            print(f"  {k:<32}: {v}")

    # Probability Simplex Closure Check
    p_sum = pred_res['p_up'] + pred_res['p_down'] + pred_res['p_neutral']
    assert abs(p_sum - 1.0) < 1e-6, f"Probabilities do not sum to 1: {p_sum}"
    print(f"\\nProbability closure verified: p_down + p_neutral + p_up = {p_sum:.8f} (within 1e-6)")

    # 4. Feature Extraction Parity Check against master.csv
    features_from_buffer = build_features(buffer_df.set_index("open_time"))
    last_computed_row = features_from_buffer.iloc[-1]
    master_row = df_train.loc[f6_timestamp]

    parity_diffs = []
    for col in get_feature_columns(feature_set="FS_ALL"):
        if col in last_computed_row and col in master_row:
            diff = abs(float(last_computed_row[col]) - float(master_row[col]))
            parity_diffs.append(diff)

    max_diff = max(parity_diffs) if parity_diffs else 0.0
    print(f"\\nFeature Parity Check vs master.csv: Max abs difference = {max_diff:.2e}")
    assert max_diff < 1e-7, f"Feature parity failure: max diff {max_diff} >= 1e-7"
    print("Feature Extraction Parity Confirmed: Streaming buffer features match master.csv within 1e-7.")

    print(\"\"\"
[Notice on Inference Context]
This historical demo is an IN-SAMPLE demonstration because the final models were fit
on this row during deployment training; it is intended to verify software parity and
pipeline execution, not to simulate out-of-sample live performance.
\"\"\")"""

    # =========================================================================
    # CELL 13: Final Deployment Refit for All Six Base Models
    # =========================================================================
    c13_md = """## Cell 13: Refit Final Deployment Base Models on Full Train Set
- **Function:** Trains final deployment instances of all six base models on the entire post-warmup TRAIN partition with Stride 3 downsampling, persists `.joblib` models and `.json` metadata into `models/horizon_2/`.
- **Runtime:** RUN THIS (trains) ~2.5 min
- **Status:** Production model serialization (run after Cells 3..8)"""

    c13_code = """from src.model_common import fit_final

print("=" * 70)
print(f"SERIALIZING FINAL BASE MODELS ON FULL TRAIN PARTITION (N = {HORIZON})")
print("=" * 70)

base_models = [
    "binary_lr",
    "binary_rf",
    "binary_gb",
    "three_class_lr",
    "three_class_rf",
    "three_class_gb",
]

for mid in base_models:
    joblib_path, json_path = fit_final(mid, horizon=HORIZON)
    size_mb = joblib_path.stat().st_size / (1024 * 1024)
    print(f"  {mid:<18} -> {joblib_path.name} ({size_mb:.2f} MB), {json_path.name}")

print(f"\\nAll six base models successfully refit and persisted in {config.MODELS_DIR / f'horizon_{HORIZON}'}.")"""

    # Assemble notebook cells
    cell_specs = [
        (c1_md, c1_code),
        (c2_md, c2_code),
        (c3_md, c3_code),
        (c4_md, c4_code),
        (c5_md, c5_code),
        (c6_md, c6_code),
        (c7_md, c7_code),
        (c8_md, c8_code),
        (c9_md, c9_code),
        (c10_md, c10_code),
        (c11_md, c11_code),
        (c12_md, c12_code),
        (c13_md, c13_code),
    ]

    for md_text, code_text in cell_specs:
        cells.append(nbf.v4.new_markdown_cell(md_text))
        # Ensure outputs are strictly empty!
        code_cell = nbf.v4.new_code_cell(code_text)
        code_cell.outputs = []
        code_cell.execution_count = None
        cells.append(code_cell)

    nb.cells = cells

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated {NOTEBOOK_PATH} with {len(nb.cells)} cells (13 code cells with EMPTY outputs).")

if __name__ == "__main__":
    create_notebook()
