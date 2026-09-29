import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, chi2_contingency
from itertools import combinations

model    = 'GPT-4o-mini'
n_rounds = 10
groups = ['18-29_Female', '18-29_Male', '30-49_Female',
          '30-49_Male',   '50-70_Female', '50-70_Male']

def find_project_root(folder):
    path = Path(__file__).resolve().parent
    while not (path / folder).exists():
        if path.parent == path:
            raise FileNotFoundError(f"Could not find a '{folder}' folder above {__file__}")
        path = path.parent
    return path

BASE_DIR = find_project_root("llm_predictions")
INPUT_FILE = BASE_DIR / "llm_predictions" / "gpt-4o-mini" / "forced_choice" / "gpt-4o-mini_fc_direct_prompt_dataset1.csv" 
HUMAN_FILE = BASE_DIR / "data" / "selected_216_words" / "human_annotation_dataset1.csv" 

df    = pd.read_csv(INPUT_FILE).set_index('word')
rates = df[groups]
human_data     = pd.read_csv(HUMAN_FILE).set_index('word')
common_words = rates.index.intersection(human_data.index)


print("- - - - Mean Selection gap - - - -")
gaps = rates.max(axis=1) - rates.min(axis=1)
print(f"\nMean selection gap: {gaps.mean():.4f}")
print(f"Max selection gap: {gaps.max():.4f}")

print("\n")
print("- - - - Pairwise Selection Rate Gap - - - -")
pairwise_gaps = []
for g1, g2 in combinations(groups, 2):
    pair_gap = np.mean(np.abs(rates[g1] - rates[g2]))
    pairwise_gaps.append((g1, g2, pair_gap))
 
print(f"\n  {'Group 1':20s} {'Group 2':20s} {'Pairwise gap':>10}")
print("  " + "-"*52)
for g1, g2, pair_gap in sorted(pairwise_gaps, key=lambda x: x[2]):
    print(f"  {g1:20s} {g2:20s} {pair_gap:>10.4f}")
 
mean_pairwise_gap = np.mean([pg[2] for pg in pairwise_gaps])
max_pair = max(pairwise_gaps, key=lambda x: x[2])

print("\n")
print("- - - - Variance - - - -")
var = rates.var()
print(f"\n  {'Group':20s} {'Variance':>12}")
print("  " + "-"*34)
for g in groups:
    print(f"  {g:20s} {var[g]:>12.6f}")

print("\n")
print("- - - - Spearman between groups (internal) - - - -")
corr  = rates.corr(method='spearman').round(3)
pairs = sorted([(a, b, corr.loc[a,b])
                for a,b in combinations(groups, 2)],
               key=lambda x: x[2])
r_vals = [pr[2] for pr in pairs]

header = f"{'':20s}" + ''.join([f"{g:>14s}" for g in groups])
print(f"\n{header}")
for g1 in groups:
    row = f"  {g1:18s}" + ''.join([f"{corr.loc[g1,g2]:>14.3f}" for g2 in groups])
    print(row)

print("\n")
print("- - - - Spearman (llm vs human) - - - -")
print(f"\n  {'Group':20s} {'r':>8}")
print("  " + "-"*48)
r_list = []
for g in groups:
    r, _ = spearmanr(rates.loc[common_words, g], human_data.loc[common_words, g])
    print(f"  {g:20s} {r:>8.3f}")
    r_list.append(r)
r_vs_human_mean = np.mean(r_list)

print("\n")
print("- - - - MAE - - - -")
print(f"\n  {'Group':20s} {'MAE':>8}")
print("  " + "-"*32)
mae_list = []
for g in groups:
    mae = np.mean(np.abs(rates.loc[common_words, g] - human_data.loc[common_words, g]))
    print(f"  {g:20s} {mae:>8.4f}")
    mae_list.append(mae)
print(f"\nMean MAE: {np.mean(mae_list):.4f}")

print("\n")
print("- - - - Chi Square - - - -")
sig_flags, sig_words = [], []
for word in rates.index:
    contingency = []
    for g in groups:
        yes = round(rates.loc[word, g] * n_rounds)
        contingency.append([yes, n_rounds - yes])
    try:
        chi2, p, _, _ = chi2_contingency(contingency)
        sig_flags.append(p < 0.05)
        if p < 0.05:
            sig_words.append((word, chi2, p))
    except ValueError:
        pass

chi_pct = sum(sig_flags) / len(sig_flags) * 100 if sig_flags else 0
print(f"\nWords tested:{len(sig_flags)}")
print(f"Significant (p<0.05): {sum(sig_flags)} ({chi_pct:.1f}%)")
