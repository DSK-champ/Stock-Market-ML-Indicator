import json
from pathlib import Path

nb_path = Path("ML/notebooks/02_Define_Targets.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find if subcost cell was already inserted
sub_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "EXTRA TRADABLE ROWS AND SUB-COST LOSS SHARE" in src:
        sub_idx = i
        break

sub_cost_source = """print('=' * 80)
print('EXTRA TRADABLE ROWS AND SUB-COST LOSS SHARE (Train Split)')
print('=' * 80)

subcost_records = []
for N in CANDIDATE_HORIZONS:
    valid = train_targets[f'has_valid_target_{N}']
    fwd_ret = train_targets[f'fwd_ret_{N}'][valid]
    abs_ret = fwd_ret.abs()
    
    # Baseline S1 (c = 0.23%)
    tgt_s1 = train_targets[f'target_{N}_S1'][valid]
    n_act_s1 = int((tgt_s1 != 0.0).sum())
    
    for rule in ['S1', 'F15', 'F10', 'F05']:
        col = f'target_{N}_{rule}'
        tgt = train_targets[col][valid]
        n_act = int((tgt != 0.0).sum())
        pct_act = n_act / len(valid) * 100.0
        extra_rows = n_act - n_act_s1
        extra_pct = (extra_rows / n_act_s1 * 100.0) if n_act_s1 > 0 else 0.0
        
        # Sub-cost signals: active signal where move < roundtrip cost c (0.23%)
        sub_cost_active = int(((tgt != 0.0) & (abs_ret < COST)).sum())
        sub_cost_loss_share = (sub_cost_active / n_act * 100.0) if n_act > 0 else 0.0
        
        subcost_records.append({
            'horizon': N,
            'rule': rule,
            'active_rows': n_act,
            'active_pct': round(pct_act, 2),
            'extra_rows_vs_S1': extra_rows,
            'extra_pct_vs_S1': round(extra_pct, 2),
            'sub_cost_loss_rows': sub_cost_active,
            'sub_cost_loss_share_pct': round(sub_cost_loss_share, 2),
        })

df_subcost = pd.DataFrame(subcost_records)
print(df_subcost.to_string(index=False))

print('\\n' + '=' * 80)
print('BIG_MOVE_N TARGET SUMMARY (Train Split)')
print('=' * 80)
bm_records = []
for N in CANDIDATE_HORIZONS:
    valid = train_targets[f'has_valid_target_{N}']
    bm = train_targets[f'big_move_{N}'][valid]
    n_ones = int((bm == 1.0).sum())
    n_zeros = int((bm == 0.0).sum())
    bm_records.append({
        'horizon': N,
        'valid_candles': len(valid),
        'big_move_1 (moves > c)': n_ones,
        'big_move_0 (moves <= c)': n_zeros,
        'share_big_move_pct': round(n_ones / len(valid) * 100.0, 2),
    })
print(pd.DataFrame(bm_records).to_string(index=False))
"""

sub_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in sub_cost_source.splitlines()]
}

if sub_idx is not None:
    nb["cells"][sub_idx] = sub_cell
else:
    nb["cells"].insert(7, sub_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 02_Define_Targets.ipynb successfully.")
