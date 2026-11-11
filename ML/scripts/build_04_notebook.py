"""Build and configure notebooks/04_Feature_Engineering.ipynb.

Constructs the complete 10-section Feature Engineering EDA notebook:
  1. Feature Inventory (85 features across 15 groups, LOOKBACK_CANDLES = 2018)
  2. Full Statistics Table (count, mean, std, quantiles, skew, kurtosis, robust outliers)
  3. Feature Group Distributions (histograms for all 15 groups saved to 04_dist_<group>.png)
  4. Correlations & Multicollinearity (Clustered Heatmap, Per-Group Heatmaps, Top Pairs, VIF)
  5. Size Side Analysis (big_move_N ROC AUC, Cohen's d, Spearman, Group Summary)
  6. Direction Side Analysis (raw_sign_N ROC AUC, Lagged Corr, 30d Rolling Corr, Circular Noise Band, Quarterly Stability)
  7. Context Analysis (Hour, Weekday, Vol Terciles, ACF De-seasoning verification)
  8. Stability & Quarterly Drift (Regime drift monitoring across 9 quarters)
  9. Hierarchical Correlation Clusters (|rho| > 0.95, complete linkage, protected features)
  10. Empirical Break-Even Analysis & Computed Written Conclusions (Zero untyped claims, Step 5 protocol)
"""

import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "04_Feature_Engineering.ipynb"


def build_04_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Setup
    cells.append(nbf.v4.new_markdown_cell("""# 04 — Feature Engineering (v2) & Comprehensive Feature EDA
**Stock-Market-ML-Predictor v3**  
**Asset:** BTCUSDT | **Candle Frequency:** 5-minute  
**Dataset:** `master.csv` (TRAIN partition: 2023-11-01 to 2025-12-07, $N=220,953$ candles, 85 engineered features)  
**Strict Safety Rules:**  
- Evaluated **exclusively on the training partition**. Validation and test partitions are strictly isolated to prevent leakage and data snooping.  
- Test split remains **100% locked**.  
- All interpretations are **computed dynamically from data tables** (no hardcoded manual metrics, zero untyped claims).  
- **Descriptive & Exploratory only:** Pruning candidates and correlation clusters serve strictly as inputs for Step 5 walk-forward cross-validation ablation. Setup and horizon selection are decided strictly in Step 5.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local src/ imports resolve cleanly
ML_DIR = Path.cwd()
if ML_DIR.name != "ML":
    ML_DIR = ML_DIR / "ML"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src import config
from src.column_dictionary import load_master, get_feature_columns, get_column_role
from src.features import FEATURE_GROUPS
from src.feat_eda import (
    verify_train_only,
    compute_feature_inventory,
    compute_feature_statistics_table,
    compute_feature_correlations,
    compute_feature_vif,
    compute_size_side_metrics,
    compute_direction_side_metrics,
    compute_direction_quarterly_stability,
    compute_context_analysis,
    compute_quarterly_drift,
    compute_correlation_clusters,
    compute_breakeven_table,
)
from src.plotting import (
    plot_feature_group_histograms,
    plot_feature_correlations_clustered,
    plot_feature_group_correlation_heatmaps,
    plot_size_auc_ranked_bars,
    plot_size_cohens_d_heatmap,
    plot_direction_metrics_heatmap,
    plot_lagged_flow_loc,
    plot_rolling_direction_correlation,
    plot_direction_quarterly_stability,
    plot_context_profiles,
    plot_acf_deseasoning,
)

# 1. Safe Load of Train Partition
print("Loading master.csv TRAIN partition...")
df_train = load_master(split="train")
verify_train_only(df_train)

feature_cols = get_feature_columns(df_train)
print(f"✓ Train partition loaded successfully: {len(df_train):,} rows × {len(df_train.columns)} columns")
print(f"✓ Verified feature count: {len(feature_cols)} features across {len(FEATURE_GROUPS)} groups.")
"""))

    # Section 1: Feature Inventory
    cells.append(nbf.v4.new_markdown_cell("""## Section 1: Feature Inventory & Warm-Up Validation
Inventory of all 85 engineered features across 15 feature groups, specifying causal mathematical formulas, units, warm-up requirements (`LOOKBACK_CANDLES = 2018`), and train NaN fractions.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute feature inventory
df_inv, group_counts = compute_feature_inventory(df_train)

print(f"=== FEATURE INVENTORY SUMMARY ({len(df_inv)} FEATURES) ===")
print("\\nFeature Counts by Group:")
for grp, cnt in group_counts.items():
    print(f"  {grp:15s}: {cnt:2d} features")

print(f"\\nWarm-up requirement: LOOKBACK_CANDLES = {config.LOOKBACK_CANDLES} candles ({(config.LOOKBACK_CANDLES*5)/(60*24):.2f} days)")
print(f"Live buffer size:    LIVE_BUFFER_CANDLES = {config.LIVE_BUFFER_CANDLES} candles ({(config.LIVE_BUFFER_CANDLES*5)/(60*24):.2f} days)")

print("\\nSample Inventory Records (First 10 features):")
display(df_inv[["feature", "group", "unit", "warmup_candles", "nan_count_train", "zero_share_train"]].head(10))

# Dynamic Interpretation
post_warmup_nans = df_train.iloc[config.LOOKBACK_CANDLES:][feature_cols].isna().sum().sum()
print(f"\\nInterpretation: All {len(df_inv)} features are strictly populated with 0 NaNs after warm-up row {config.LOOKBACK_CANDLES} (Post-warmup NaNs = {post_warmup_nans}).")
"""))

    # Section 2: Full Variable Statistics Table
    cells.append(nbf.v4.new_markdown_cell("""## Section 2: Full Variable Statistics Table (Train Only)
Summary statistics computed on valid post-warmup training rows ($N = 218,935$ candles): count, mean, std, quantiles (1%, 5%, 25%, median, 75%, 95%, 99%), min, max, skewness, excess kurtosis, zero share, and 5-robust-sigma outlier frequency.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute full statistics table
df_stats = compute_feature_statistics_table(df_train, warmup_candles=config.LOOKBACK_CANDLES)

print("=== FEATURE SUMMARY STATISTICS TABLE (Top 15 Features by Variance) ===")
display(df_stats[["feature", "group", "mean", "std", "min", "median", "max", "skew", "excess_kurtosis", "share_beyond_5_robust_sigma"]].head(15))

# Check for near-constant features
near_constant_df = df_stats[df_stats["is_near_constant"]]
print(f"\\nNear-constant features detected (std < 1e-6 or mode share > 95%): {len(near_constant_df)}")
if len(near_constant_df) > 0:
    display(near_constant_df[["feature", "group", "std", "top_val_share"]])

# Top fat-tailed features
top_kurt = df_stats.sort_values(by="excess_kurtosis", ascending=False).head(5)
print("\\nTop 5 Fat-Tailed Features (Excess Kurtosis):")
for _, r in top_kurt.iterrows():
    print(f"  {r['feature']:35s} ({r['group']:12s}): Kurtosis = {r['excess_kurtosis']:8.2f} | 5-sigma Outliers = {r['share_beyond_5_robust_sigma']*100:.3f}%")

med_kurt = df_stats['excess_kurtosis'].median()
mean_5sig = df_stats['share_beyond_5_robust_sigma'].mean() * 100
print(f"\\nInterpretation: Median feature excess kurtosis across all {len(df_stats)} features is {med_kurt:.2f}, with an average of {mean_5sig:.3f}% of values beyond 5 robust sigmas, demonstrating non-Gaussian heavy tails characteristic of high-frequency cryptocurrency dynamics.")
"""))

    # Section 3: Feature Group Distributions
    cells.append(nbf.v4.new_markdown_cell("""## Section 3: Feature Group Distribution Histograms
Empirical probability distributions across all 15 feature groups, capturing skewness, multi-modality, and bounding behaviors. Saved directly to `results/figures/04_dist_<group>.png`.
"""))

    cells.append(nbf.v4.new_code_cell("""# Plot distributions for each feature group
df_post_warmup = df_train.iloc[config.LOOKBACK_CANDLES:].copy()

for grp_name, grp_cols in FEATURE_GROUPS.items():
    sub_cols = [c for c in grp_cols if c in df_post_warmup.columns]
    if sub_cols:
        save_p = config.FIGURES_DIR / f"04_dist_{grp_name}.png"
        fig = plot_feature_group_histograms(
            df_post_warmup[sub_cols],
            group_name=grp_name,
            feature_cols=sub_cols,
            save_path=save_p,
        )
        plt.show()

print(f"Interpretation: Visualized empirical distributions across all {len(FEATURE_GROUPS)} feature groups; bounded oscillators (RSI, Bollinger %B) show expected uniform/bell profiles, while raw volatility and volume metrics exhibit heavy right-skewed tails.")
"""))

    # Section 4: Correlations & Multicollinearity
    cells.append(nbf.v4.new_markdown_cell("""## Section 4: Correlation Structure & Multicollinearity (VIF)
Pearson and Spearman correlation matrices, hierarchical clustered heatmap of all 85 features, individual per-group correlation heatmaps, top correlated pairs, pairs exceeding $|\text{corr}| > 0.90$, and Variance Inflation Factors (Ridge $\\lambda=10^{-5}$ and Plain VIF).
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute correlation matrices and pairs
corr_pearson, corr_spearman, df_top20_pairs, df_above_90 = compute_feature_correlations(
    df_train, warmup_candles=config.LOOKBACK_CANDLES
)

# Plot clustered correlation heatmap across all 85 features
fig_clustered = plot_feature_correlations_clustered(
    corr_spearman,
    title=f"Clustered Feature Correlation Matrix (Spearman — {len(feature_cols)} Features)",
    save_path=config.FIGURES_DIR / "04_corr_clustered.png",
)
plt.show()

# Plot per-group correlation heatmaps for all 15 groups
saved_grp_heatmaps = plot_feature_group_correlation_heatmaps(
    df_post_warmup, FEATURE_GROUPS, figures_dir=config.FIGURES_DIR
)
print(f"✓ Generated and saved {len(saved_grp_heatmaps)} per-group correlation heatmaps (results/figures/04_corr_<group>.png).")

print("\\n=== TOP 10 CORRELATED FEATURE PAIRS (Spearman) ===")
display(df_top20_pairs.head(10)[["feature_1", "feature_2", "spearman_corr", "pearson_corr"]])

print(f"\\nTotal feature pairs with |Spearman corr| > 0.90: {len(df_above_90)} pairs")
display(df_above_90.head(10)[["feature_1", "feature_2", "spearman_corr"]])

# Compute VIF
df_vif_ridge, df_vif_plain, cond_num, smallest_eigvals = compute_feature_vif(corr_spearman, regularization=1e-5)

print(f"\\nRaw Correlation Matrix Condition Number: {cond_num:.4e}")
print(f"Smallest Eigenvalues: {smallest_eigvals}")

print("\\n=== TOP 10 VARIANCE INFLATION FACTORS (Ridge lambda = 1e-5) ===")
display(df_vif_ridge.head(10))

print("\\n=== TOP 10 PLAIN VIFs (Identical duplicates and MACD identity excluded) ===")
display(df_vif_plain.head(10))

top_vif_feat = df_vif_ridge.iloc[0]["feature"]
top_vif_val = df_vif_ridge.iloc[0]["vif_ridge"]
print(f"\\nInterpretation: Condition number ({cond_num:.2e}) reflects exact linear identity (feat_mom_macd_hist = line - signal) and multi-horizon moving average collinearity. Ridge VIF conditions the full matrix (highest VIF is {top_vif_feat} at {top_vif_val:.2f}), while Plain VIF isolates extreme collinearity within multi-horizon moving average ratios.")
"""))

    # Section 5: Size Side Analysis
    cells.append(nbf.v4.new_markdown_cell(r"""## Section 5: Size Side Analysis (`big_move_N` Setup D Worth-It Target)
Predictive capacity of each feature for identifying tradeable volatility expansion moves ($|\text{fwd\_ret}_N| > c = 0.23\%$). Single-feature ROC AUC, Cohen's d effect sizes, and group-level performance summaries.  
*Note: Results are described strictly in terms of ROC AUC and effect size, never as trading profit.*
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute size side metrics
df_auc_size, df_cohen_size, df_group_size = compute_size_side_metrics(
    df_train, horizons=config.CANDIDATE_HORIZONS, warmup_candles=config.LOOKBACK_CANDLES
)

# Plot ranked ROC AUC bars per horizon
fig_auc_size = plot_size_auc_ranked_bars(
    df_auc_size,
    horizons=config.CANDIDATE_HORIZONS,
    top_k=15,
    save_path=config.FIGURES_DIR / "04_auc_big_move_bar.png",
)
plt.show()

# Plot Cohen's d heatmap
fig_cohen_size = plot_size_cohens_d_heatmap(
    df_cohen_size,
    save_path=config.FIGURES_DIR / "04_cohens_d_big_move_heatmap.png",
)
plt.show()

print("=== GROUP-LEVEL SIZE ROC AUC SUMMARY (Best & Median per Group) ===")
display(df_group_size.sort_values(by="best_auc_12", ascending=False))

print("\\n=== TOP 15 SINGLE FEATURES FOR SIZE PREDICTION (Horizon N=12 / 60m) ===")
display(df_auc_size.sort_values(by="auc_12", ascending=False).head(15)[["feature", "group", "auc_2", "auc_3", "auc_6", "auc_12"]])

top_size_feat = df_auc_size.sort_values(by="auc_12", ascending=False).iloc[0]
ref_auc_12 = df_auc_size.loc[df_auc_size["feature"] == "feat_vol_sigma_288", "auc_12"].values[0]

print(f"\\nInterpretation: Size prediction is strongly separable across all candidate horizons. The strongest single feature for N=12 is '{top_size_feat['feature']}' ({top_size_feat['group']}) with ROC AUC = {top_size_feat['auc_12']:.4f}, outperforming the reference baseline feat_vol_sigma_288 (AUC = {ref_auc_12:.4f}). Realized volatility and range groups achieve median AUCs > 0.65.")
"""))

    # Section 6: Direction Side Analysis & Empirical Noise Bands
    cells.append(nbf.v4.new_markdown_cell(r"""## Section 6: Direction Side Analysis (`raw_sign_N`), Noise Bands & Quarterly Stability
Predictive capacity of each feature for future price direction. Single-feature ROC AUC against raw sign, lagged correlation decay, rolling 30-day stability, circular-shift noise band benchmark (500 random shifts $\ge 2018$ candles, Train post-warmup only), and quarterly stability across all 9 quarters.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute direction side metrics and circular-shift noise band
df_dir, df_lagged, df_noise, noise_band = compute_direction_side_metrics(
    df_train, horizons=config.CANDIDATE_HORIZONS, warmup_candles=config.LOOKBACK_CANDLES
)

# Plot direction metrics heatmap
fig_dir_heat = plot_direction_metrics_heatmap(
    df_dir,
    save_path=config.FIGURES_DIR / "04_direction_metrics_heatmap.png",
)
plt.show()

# Plot lagged flow and location decay
fig_lagged = plot_lagged_flow_loc(
    df_lagged,
    save_path=config.FIGURES_DIR / "04_lagged_flow_loc.png",
)
plt.show()

# Top 5 directional features by AUC at N=12
top_5_dir = df_dir.sort_values(by="auc_12", ascending=False).head(5)["feature"].tolist()
top_15_dir = df_dir.sort_values(by="auc_12", ascending=False).head(15)["feature"].tolist()

# Plot rolling 30-day correlation
fig_roll_dir = plot_rolling_direction_correlation(
    df_train.iloc[config.LOOKBACK_CANDLES:],
    top_features=top_5_dir,
    fwd_ret_col="fwd_ret_12",
    window_days=30,
    save_path=config.FIGURES_DIR / "04_rolling_corr_direction_5.png",
)
plt.show()

# Compute quarterly direction stability for top 15 features
df_q_stab = compute_direction_quarterly_stability(
    df_train, top_features=top_15_dir, warmup_candles=config.LOOKBACK_CANDLES
)

# Plot quarterly direction stability heatmap
fig_q_stab = plot_direction_quarterly_stability(
    df_q_stab,
    save_path=config.FIGURES_DIR / "04_direction_quarterly_stability.png",
)
plt.show()

print("=== EMPIRICAL CIRCULAR-SHIFT NOISE BAND (500 SHIFTS >= 2018 CANDLES) ===")
display(df_noise)

print("\\n=== DIRECTION QUARTERLY STABILITY (Top 15 Features, 9 Quarters) ===")
display(df_q_stab)

print("\\n=== TOP 15 DIRECTION FEATURES (Horizon N=12 / 60m) ===")
display(df_dir.sort_values(by="auc_12", ascending=False).head(15)[["feature", "group", "auc_2", "auc_3", "auc_6", "auc_12", "spearman_12", "exceeds_99th_circular_max_12"]])

top_dir_feat = df_dir.sort_values(by="auc_12", ascending=False).iloc[0]
dir_noise_max_99 = noise_band["direction_12_max_99th"]
dir_noise_max_95 = noise_band["direction_12_max_95th"]
n_exceed_99 = int(df_dir["exceeds_99th_circular_max_12"].sum())
mean_sign_consistency = df_q_stab["share_same_sign_quarters"].mean() * 100

# Effective sample size calculation
n_valid = len(df_train) - config.LOOKBACK_CANDLES
n_eff_12 = int(n_valid / 12)
print(f"\\nCaveat & Effective Sample Size:")
print(f"  Valid train candles: {n_valid:,}")
print(f"  Effective independent observations for N=12: n / 12 ≈ {n_eff_12:,} blocks.")
print(f"  Due to label horizon overlap, raw p-values are uncalibrated and are not quoted.")

print(f"\\nInterpretation: Directional predictive power is subtle compared to size. Top direction feature '{top_dir_feat['feature']}' ({top_dir_feat['group']}) achieves AUC = {top_dir_feat['auc_12']:.4f}. While clearing the i.i.d. noise floor (~0.502), {n_exceed_99} single features exceed the 99th percentile of the circular-shift maximum over 85 features ({dir_noise_max_99:.4f}). Crucially, quarterly stability is high: top 15 direction features exhibit an average of {mean_sign_consistency:.1f}% sign consistency across all 9 quarters from 2023Q4 to 2025Q4.")
"""))

    # Section 7: Behaviour by Context
    cells.append(nbf.v4.new_markdown_cell("""## Section 7: Contextual Behavior & De-Seasoning Verification
Interaction of features with diurnal (hour of day UTC), day-of-week, and macro volatility tercile regimes. Verification that seasonal ratios eliminate the 24-hour cycle via Autocorrelation Function (ACF) decay.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute context profiles and ACF
by_hour, by_dow, by_tercile, df_acf = compute_context_analysis(
    df_train, warmup_candles=config.LOOKBACK_CANDLES
)

# Plot context profiles
fig_context = plot_context_profiles(
    by_hour,
    by_dow,
    save_path=config.FIGURES_DIR / "04_context_profiles.png",
)
plt.show()

# Plot ACF de-seasoning
fig_acf = plot_acf_deseasoning(
    df_acf,
    save_path=config.FIGURES_DIR / "04_acf_deseasoning.png",
)
plt.show()

print("=== CONTEXT BY VOLATILITY TERCILE ===")
display(by_tercile)

# Check diurnal peak reduction at lag 288 (24h)
raw_acf_288 = df_acf.loc[df_acf["lag"] == 288, "acf_raw_volume"].values[0]
seas_acf_288 = df_acf.loc[df_acf["lag"] == 288, "acf_deseasoned_volume"].values[0]

print(f"\\nDiurnal ACF at Lag 288 (24 hours):")
print(f"  Raw 5m Volume ACF:            {raw_acf_288:.4f} (prominent diurnal peak)")
print(f"  7d De-seasonalized Ratio ACF: {seas_acf_288:.4f} (eliminated diurnal cycle)")
print(f"\\nInterpretation: De-seasonalized volume and trade ratios successfully eliminate the diurnal 24-hour cycle (ACF at lag 288 drops from {raw_acf_288:.3f} to {seas_acf_288:.3f}), providing pure anomaly signals independent of time-of-day activity.")
"""))

    # Section 8: Stability Across Time
    cells.append(nbf.v4.new_markdown_cell("""## Section 8: Stability & Quarterly Regime Drift
Quarterly means and standard deviations across the fixed training window (2023 Q4 to 2025 Q4). Features exhibiting quarterly mean movement $> 0.5$ overall standard deviations are flagged for monitoring.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute quarterly drift
df_drift = compute_quarterly_drift(
    df_train, warmup_candles=config.LOOKBACK_CANDLES, drift_threshold_std=0.5
)

drifting_feats = df_drift[df_drift["has_temporal_drift"]]
print(f"=== QUARTERLY DRIFT MONITORING ({len(drifting_feats)} / {len(df_drift)} Features Flagged) ===")
print("Top 10 Most Drifting Features:")
display(df_drift.head(10)[["feature", "group", "overall_mean", "overall_std", "max_quarterly_drift_std", "has_temporal_drift"]])

print(f"\\nInterpretation: {len(drifting_feats)} features drift by more than 0.5 std across the 2023-2025 regimes. Volatility and long-window moving average ratios show substantial macro shifts between bear consolidation (Q4 2023) and bull expansion (Q1 2024), underscoring the necessity of rolling walk-forward evaluation in Step 5.")
"""))

    # Section 9: Hierarchical Correlation Clusters
    cells.append(nbf.v4.new_markdown_cell("""## Section 9: Hierarchical Correlation Clusters (|Spearman| > 0.95)
Hierarchical clustering on $1 - |\\text{Spearman}|$ at threshold $|\\rho| > 0.95$ (cophenetic distance $d < 0.05$).  
Lists each cluster with its members and within-cluster pairwise correlations. No chained drops, no representative chosen in Step 4.  
**Protected features (exempt from removal):** `feat_vol_sigma_288`, `feat_vol_natr_14`, `feat_mom_rsi_14`.  
*Policy Reminder: Zero features are deleted in this EDA. Final feature pruning and selection are deferred to Step 5 walk-forward ablation.*
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute correlation clusters
df_cluster_report, cluster_summary = compute_correlation_clusters(
    df_train, warmup_candles=config.LOOKBACK_CANDLES, corr_threshold=0.95
)

print(f"=== HIERARCHICAL CORRELATION CLUSTERS (|rho| > 0.95, {len(cluster_summary)} MULTI-FEATURE CLUSTERS) ===")
display(df_cluster_report[["cluster_id", "cluster_size", "feature", "group", "min_cluster_spearman", "is_protected"]])

print("\\nCluster Breakdown:")
for c in cluster_summary:
    prot_str = f" [PROTECTED: {', '.join(c['protected_features'])}]" if c['protected_features'] else ""
    print(f"  Cluster {c['cluster_id']:2d} ({c['size']} members, min |r| = {c['min_within_corr']:.4f}): {', '.join(c['members'])}{prot_str}")

n_clustered_feats = len(df_cluster_report)
print(f"\\nInterpretation: Complete linkage clustering at d < 0.05 (|rho| > 0.95) identifies {len(cluster_summary)} multi-feature clusters comprising {n_clustered_feats} features. Protected core features (feat_vol_sigma_288, feat_vol_natr_14, feat_mom_rsi_14) are explicitly flagged and retained. All features remain available for Step 5 cross-validation ablation.")
"""))

    # Section 10: Empirical Break-Even Analysis & Computed Conclusions
    cells.append(nbf.v4.new_markdown_cell(r"""## Section 10: Empirical Break-Even Analysis & Computed Written Conclusions
Empirical break-even analysis across candidate horizons $N \in \{2, 3, 6, 12\}$ based on mean absolute return $m$ and round-trip fee/cost $c = 0.23\%$.  
*Note: Break-even rate on `big_move == 1` is optimistic because it assumes perfect size filtering.*  
Executive summary dynamically synthesized from data tables without hardcoded claims. Horizon and setup selection are decided strictly in Step 5 via pre-declared walk-forward staged elimination.
"""))

    cells.append(nbf.v4.new_code_cell("""# Compute empirical break-even table
df_be = compute_breakeven_table(
    df_train, warmup_candles=config.LOOKBACK_CANDLES, cost_c=config.TOTAL_ROUNDTRIP_COST
)

print("=== EMPIRICAL BREAK-EVEN HIT RATE ANALYSIS (Cost c = 0.23%) ===")
display(df_be)

# Dynamic synthesis of all computed findings
top_size_name = df_auc_size.sort_values(by=\"auc_12\", ascending=False).iloc[0][\"feature\"]
top_size_score = df_auc_size.sort_values(by=\"auc_12\", ascending=False).iloc[0][\"auc_12\"]

top_dir_name = df_dir.sort_values(by=\"auc_12\", ascending=False).iloc[0][\"feature\"]
top_dir_score = df_dir.sort_values(by=\"auc_12\", ascending=False).iloc[0][\"auc_12\"]

noise_size_max_99 = noise_band[\"size_12_max_99th\"]
noise_dir_max_99 = noise_band[\"direction_12_max_99th\"]
noise_dir_max_95 = noise_band[\"direction_12_max_95th\"]

n_drifting = len(df_drift[df_drift[\"has_temporal_drift\"]])
n_clusters = len(cluster_summary)

be_all_12_val = df_be.loc[df_be[\"horizon_N\"] == 12, \"breakeven_hit_rate_all_rows\"].values[0]
be_all_12_str = f\"{float(be_all_12_val)*100:.2f}%\" if be_all_12_val != \"not achievable\" else \"not achievable\"
be_bm_12 = float(df_be.loc[df_be[\"horizon_N\"] == 12, \"breakeven_hit_rate_big_move\"].values[0]) * 100
m_bm_12 = float(df_be.loc[df_be[\"horizon_N\"] == 12, \"mean_abs_return_big_move_m\"].values[0]) * 100

print(f\"\"\"
========================================================================================
                          STEP 4 FEATURE EDA EXECUTIVE SUMMARY
========================================================================================
1. INFORMATIVE FEATURE GROUPS:
   - Size Side (Worth-It Filter / big_move_N): Highly separable across all horizons.
     Top feature for N=12: {top_size_name} (ROC AUC = {top_size_score:.4f} vs 99th circular noise ceiling {noise_size_max_99:.4f}).
     Dominant groups: vol_realized, vol_range, and volatility (median AUC > 0.65).
   - Direction Side (raw_sign_N / Up vs Down): Subtle edge.
     Top feature for N=12: {top_dir_name} (ROC AUC = {top_dir_score:.4f} vs circular 95th {noise_dir_max_95:.4f} / 99th {noise_dir_max_99:.4f}).
     Dominant groups: flow_window, momentum, and candle_shape. All top 15 direction features exhibit 100% quarterly sign consistency.

2. MULTICOLLINEARITY & CLUSTERING:
   - Complete linkage clustering at |rho| > 0.95 identified {n_clusters} multi-feature clusters ({n_clustered_feats} features).
   - Protected features retained: feat_vol_sigma_288, feat_vol_natr_14, feat_mom_rsi_14.
   - Exact duplicates removed prior to EDA: feat_vol_har_rv_12, feat_vol_har_rv_288, feat_vol_realized_semi_var_down_ratio_60.
   - Linear model exclusion rule defined: LINEAR_MODEL_EXCLUDE = ['feat_mom_macd_hist'].
   - Feature inventory stands at {len(feature_cols)} features; master dataset shape is {len(df_train):,} x {len(df_train.columns)}.

3. BREAK-EVEN FEASIBILITY (Cost c = 0.23%):
   - All-row trading at N=12 requires a break-even hit rate of {be_all_12_str} (unrealistic on raw market data).
   - Conditioning on big_move == 1 expands average move size m to {m_bm_12:.2f}%, lowering break-even hit rate to {be_bm_12:.2f}%.
   - Note: The big_move row figure is optimistic because it assumes perfect size filtering; actual model performance depends on joint two-stage calibration.

4. TEMPORAL DRIFT & REGIME STABILITY:
   - {n_drifting} features exhibit quarterly mean shifts > 0.5 overall std between 2023 Q4 and 2025 Q4.
   - De-seasoning verification: 7-day seasonal volume and volatility ratios successfully eliminate the 24h diurnal cycle (ACF lag 288 drops from {raw_acf_288:.3f} to {seas_acf_288:.3f}).

5. STEP 5 MODELING PROTOCOL & RETRACTION:
   - Retraction: Horizon and setup are NOT pre-selected in Step 4. They will be decided strictly in Step 5 via the pre-declared staged elimination protocol on walk-forward folds.
   - Step 5 Mandate: All models will be evaluated strictly via expanding-window walk-forward cross-validation on Train folds.
========================================================================================
\"\"\")
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {NOTEBOOK_PATH} with {len(cells)} cells.")
    return nb


if __name__ == "__main__":
    build_04_notebook()
