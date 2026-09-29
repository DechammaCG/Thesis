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
human_data = pd.read_csv(DATA_DIR /"human_individual_annotaion_dataset1.csv")
human_data['selected'] = (human_data['rating'] > 0).astype(int)
human_data['gender'] = human_data['age_gender_label'].str.split('_').str[1]

GENDER_GROUPS = ['Female', 'Male']
rates = human_data.pivot_table(index='word', columns='gender',values='selected', aggfunc='mean')
rates = rates[GENDER_GROUPS]

print("- - - - Mean Selection Rates - - - -")

gaps = rates.max(axis=1) - rates.min(axis=1)
print(f"Mean selection gap: {gaps.mean():.4f}")
print(f"Max selection gap: {gaps.max():.4f}")

print("\n")
print("- - - - Variance - - - -")
var = rates.var()
for g in GENDER_GROUPS:
    print(f"{g:5s}: {var[g]:.6f}")

print("\n")
print("- - - - Spearman rank - - - -")
corr = rates.corr(method='spearman').round(3)
pairs = sorted([(a, b, corr.loc[a, b]) for a, b in combinations(GENDER_GROUPS, 2)],
               key=lambda x: x[2])
print(corr.to_string())

print("\n")
print("- - - - Chi Square - - - -")
chi_results = []
for word, sub in human_data.groupby('word'):
    tab = sub.groupby('gender')['selected'].agg(['sum', 'count'])
    tab['not_selected'] = tab['count'] - tab['sum']
    cont_table = tab[['sum', 'not_selected']].values
    try:
        chi2, p, _, _ = chi2_contingency(cont_table)
        chi_results.append({'word': word, 'chi2': round(chi2, 4),
                             'p_value': round(p, 6), 'significant': p < 0.05})
    except ValueError:
        pass

chi_df = pd.DataFrame(chi_results).sort_values('p_value')
sig, total = chi_df['significant'].sum(), len(chi_df)
print(f"Total words tested: {total}")
print(f"Significant (p<0.05): {sig} ({sig/total*100:.1f}%)")

print("\n")
print("- - - - Krippendorff Alpha - - - -")
def alpha_for(sub):
    mat = sub.pivot_table(index='user_id', columns='word', values='selected', aggfunc='mean')
    return krippendorff.alpha(reliability_data=mat.values, level_of_measurement='nominal')
gender_alphas = {}
for g, sub in human_data.groupby('gender'):
    a = alpha_for(sub)
    gender_alphas[g] = a
    print(f"{g:5s}: {a:.4f} (n={sub['user_id'].nunique()})")
print(f"Mean alpha: {np.mean(list(gender_alphas.values())):.4f}")