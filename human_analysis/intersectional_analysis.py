import pandas as pd
import numpy as np
import krippendorff
from pathlib import Path
from scipy.stats import spearmanr, chi2_contingency
from itertools import combinations

def find_project_root(folder):
    path = Path(__file__).resolve().parent
    while not (path / folder).exists():
        if path.parent == path:
            raise FileNotFoundError(f"Could not find a '{folder}' folder above {__file__}")
        path = path.parent
    return path

BASE_DIR = find_project_root("data")
DATA_DIR = BASE_DIR / "data" / "selected_216_words"

human_dataset_1 = pd.read_csv(DATA_DIR / "human_annotation_dataset1.csv").set_index('word')
human_raw_data_1 = pd.read_csv(DATA_DIR /"human_individual_annotaion_dataset1.csv")
human_raw_data_1['selected'] = (human_raw_data_1['rating'] > 0).astype(int)

HUMAN_GROUPS = ['18-29_Female', '18-29_Male', '30-49_Female',
                '30-49_Male', '50-70_Female', '50-70_Male']

print("- - - - Mean Selection Rates - - - -")

gaps = human_dataset_1[HUMAN_GROUPS].max(axis=1) - human_dataset_1[HUMAN_GROUPS].min(axis=1)
print(f"Mean selection gap: {gaps.mean():.4f}")
print(f"Max selection gap: {gaps.max():.4f}")


print("\n")
print("- - - - Pairwise Selection Rate Gap - - - -")
pairwise_gaps = []
for g1, g2 in combinations(HUMAN_GROUPS, 2):
    pair_gap = np.mean(np.abs(human_dataset_1[g1] - human_dataset_1[g2]))
    pairwise_gaps.append((g1, g2, pair_gap))
 
print(f"\n  {'Group 1':20s} {'Group 2':20s} {'Pairwise gap':>10}")
print("  " + "-"*52)
for g1, g2, pair_gap in sorted(pairwise_gaps, key=lambda x: x[2]):
    print(f"  {g1:20s} {g2:20s} {pair_gap:>10.4f}")
 
mean_pairwise_gap = np.mean([pg[2] for pg in pairwise_gaps])
max_pair = max(pairwise_gaps, key=lambda x: x[2])


print("\n")
print("- - - - Variance - - - -")
var = human_dataset_1[HUMAN_GROUPS].var()
for g in HUMAN_GROUPS:
    print(f"{g:20s}: {var[g]:.6f}")

print("\n")
print("- - - - Spearman rank - - - -")
corr = human_dataset_1[HUMAN_GROUPS].corr(method='spearman').round(3)
pairs = sorted([(a, b, corr.loc[a, b]) for a, b in combinations(HUMAN_GROUPS, 2)],
               key=lambda x: x[2])

print(corr.to_string())

print("\n")
print("- - - - Chi Square - - - -")
chi_results = []
for word, sub in human_raw_data_1.groupby('word'):
    tab = sub.groupby('age_gender_label')['selected'].agg(['sum', 'count'])
    tab['not_selected'] = tab['count'] - tab['sum']
    contingency = tab[['sum', 'not_selected']].values
    try:
        chi2, p, _, _ = chi2_contingency(contingency)
        chi_results.append({'word': word, 'chi2': round(chi2, 4),
                             'p_value': round(p, 6), 'significant': p < 0.05})
    except ValueError:
        pass

chi_df = pd.DataFrame(chi_results).sort_values('p_value')
sig, total = chi_df['significant'].sum(), len(chi_df)
print(f"  Total words tested:   {total}")
print(f"  Significant (p<0.05): {sig} ({sig/total*100:.1f}%)")


print("\n")
print("- - - - Krippendorff Alpha - - - -")
def alpha_for(sub):
    mat = sub.pivot_table(index='user_id', columns='word', values='selected', aggfunc='mean')
    return krippendorff.alpha(reliability_data=mat.values, level_of_measurement='nominal')

inter_alphas = {}
for g, sub in human_raw_data_1.groupby('age_gender_label'):
    a = alpha_for(sub)
    inter_alphas[g] = a
    print(f"{g:15s}: {a:.4f}  (n={sub['user_id'].nunique()})")
print(f"Mean alphal:  {np.mean(list(inter_alphas.values())):.4f}")