import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

model  = 'GPT-4o-mini'
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
INPUT_FILE = BASE_DIR / "llm_predictions" / "gpt-4o-mini" / "rating_scale" / "gpt-4o-mini_rs_cot_prompt_dataset2.csv"

df      = pd.read_csv(INPUT_FILE).set_index('word')
ratings = df[groups]


print("- - - - Mean Rating per Group - - - -")
means = ratings.mean()
print(f"\n  {'Group':20s} {'Mean':>10}")
print("  " + "-"*32)
for g in groups:
    print(f"  {g:20s} {means[g]:>10.3f}")

print("\n")
print("- - - - Mean Rating gap - - - -")
gaps = ratings.max(axis=1) - ratings.min(axis=1)
print(f"\nMean rating gap: {gaps.mean():.4f}")
print(f"Max rating gap: {gaps.max():.4f}")

print("\n")
print("- - - - Pairwise Rating Gap - - - -")
pairwise_gaps = []
for g1, g2 in combinations(groups, 2):
    pair_gap = np.mean(np.abs(ratings[g1] - ratings[g2]))
    pairwise_gaps.append((g1, g2, pair_gap))

print(f"\n  {'Group 1':20s} {'Group 2':20s} {'Pairwise gap':>10}")
print("  " + "-"*52)
for g1, g2, pair_gap in sorted(pairwise_gaps, key=lambda x: x[2]):
    print(f"  {g1:20s} {g2:20s} {pair_gap:>10.4f}")

print("\n")
print("- - - - Variance - - - -")
var = ratings.var()
print(f"\n  {'Group':20s} {'Variance':>12}")
print("  " + "-"*34)
for g in groups:
    print(f"  {g:20s} {var[g]:>12.6f}")

print("\n")
print("- - - - Spearman between groups (internal) - - - -")
corr  = ratings.corr(method='spearman').round(3)
pairs = sorted([(a, b, corr.loc[a,b])
                for a,b in combinations(groups, 2)],
               key=lambda x: x[2])
r_vals = [pr[2] for pr in pairs]

header = f"{'':20s}" + ''.join([f"{g:>14s}" for g in groups])
print(f"\n{header}")
for g1 in groups:
    row = f"  {g1:18s}" + ''.join([f"{corr.loc[g1,g2]:>14.3f}" for g2 in groups])
    print(row)