"""
Test hierarchical prior: species borrow temporal signal from their taxonomic group.

P(species | time) = λ * P(species | time)_empirical
                  + (1-λ) * P(group | time) * P(species | group)

where λ = n_species / (n_species + k), tuned via LOO-CV.
"""

import re
import itertools
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

LABELS_PATH = 'scratch/train_soundscapes_labels.csv'
TAXONOMY_PATH = 'scratch/taxonomy.csv'

TIME_BIN_HOURS = 2
N_TIME_BINS = 24 // TIME_BIN_HOURS
TIME_SIGMA = 1.0
MONTH_SIGMA = 1.0
LOG_EPS = 1e-10

K_VALUES = [1, 2, 5, 10, 20, 50]  # mixing parameter to sweep


def parse_filename(fn):
    m = re.match(r'.*_(S\d+)_(\d{8})_(\d{6})\.ogg', fn)
    if m:
        site = m.group(1)
        month = int(m.group(2)[4:6])
        time_bin = int(m.group(3)[:2]) // TIME_BIN_HOURS
        return site, time_bin, month
    return None, None, None


def smooth_circular(counts_df, n_bins, all_labels, sigma):
    arr = counts_df.reindex(index=range(n_bins), columns=all_labels, fill_value=0).astype(float).values
    arr_tiled = np.tile(arr, (3, 1))
    arr_smoothed = gaussian_filter1d(arr_tiled, sigma=sigma, axis=0)
    arr_smoothed = arr_smoothed[n_bins:2 * n_bins]
    arr_smoothed += 1
    df_s = pd.DataFrame(arr_smoothed, index=range(n_bins), columns=all_labels)
    return df_s.div(df_s.sum(axis=1), axis=0)


def compute_priors(df_exp, all_labels, taxonomy, k):
    # Global
    global_counts = df_exp['label'].value_counts()
    global_prior = pd.Series(0.0, index=all_labels)
    global_prior.update(global_counts)
    global_prior = (global_prior + 1) / (global_prior + 1).sum()

    # Site
    site_counts = df_exp.groupby(['site', 'label']).size().unstack(fill_value=0)
    site_counts = site_counts.reindex(columns=all_labels, fill_value=0) + 1
    p_site = site_counts.div(site_counts.sum(axis=1), axis=0)

    # Species-level time prior (smoothed)
    time_counts = df_exp.groupby(['time_bin', 'label']).size().unstack(fill_value=0)
    p_time_species = smooth_circular(time_counts, N_TIME_BINS, all_labels, TIME_SIGMA)

    # Group-level time prior
    group_map = taxonomy.set_index('primary_label')['class_name'].to_dict()
    df_exp_g = df_exp.copy()
    df_exp_g['group'] = df_exp_g['label'].map(group_map)
    all_groups = sorted(df_exp_g['group'].dropna().unique())

    group_time_counts = df_exp_g.groupby(['time_bin', 'group']).size().unstack(fill_value=0)
    p_time_group = smooth_circular(group_time_counts, N_TIME_BINS, all_groups, TIME_SIGMA)

    # P(species | group)
    species_group_counts = df_exp_g.groupby(['group', 'label']).size().unstack(fill_value=0)
    species_group_counts = species_group_counts.reindex(columns=all_labels, fill_value=0) + 1
    p_species_given_group = species_group_counts.div(species_group_counts.sum(axis=1), axis=0)

    # Species observation counts for mixing weight
    n_species = df_exp.groupby('label').size().reindex(all_labels, fill_value=0)

    # Hierarchical time prior: mix species-level with group-level
    p_time_hierarchical = pd.DataFrame(0.0, index=range(N_TIME_BINS), columns=all_labels)
    for label in all_labels:
        group = group_map.get(label)
        n_j = n_species[label]
        lam = n_j / (n_j + k)  # data-driven mixing weight

        species_component = p_time_species[label].values  # (N_TIME_BINS,)

        if group in all_groups:
            group_component = (
                p_time_group[group].values *           # P(group | time)
                p_species_given_group.loc[group, label] # P(species | group)
            )
            # renormalise group component so it sums to 1 over time
            group_component = group_component / (group_component.sum() + 1e-10)
        else:
            group_component = species_component

        p_time_hierarchical[label] = lam * species_component + (1 - lam) * group_component

    # Renormalise rows
    p_time_hierarchical = p_time_hierarchical.div(p_time_hierarchical.sum(axis=1), axis=0)

    # Month prior
    month_counts = df_exp.groupby(['month', 'label']).size().unstack(fill_value=0)
    month_counts.index = month_counts.index - 1
    p_month = smooth_circular(month_counts, 12, all_labels, MONTH_SIGMA)

    return global_prior, p_site, p_time_hierarchical, p_month


def get_scores(site, time_bin, month, global_prior, p_site, p_time, p_month):
    log_global = np.log(global_prior.values + LOG_EPS)
    known_sites = set(p_site.index)

    terms = []
    n_terms = 0
    if site in known_sites:
        terms.append(np.log(p_site.loc[site].values + LOG_EPS))
        n_terms += 1
    if time_bin is not None:
        terms.append(np.log(p_time.iloc[time_bin].values + LOG_EPS))
        n_terms += 1
    if month is not None:
        terms.append(np.log(p_month.iloc[month - 1].values + LOG_EPS))
        n_terms += 1

    if n_terms == 0:
        log_scores = log_global
    else:
        log_scores = sum(terms) - (n_terms - 1) * log_global

    log_scores = log_scores - log_scores.max()
    scores = np.exp(log_scores)
    scores /= scores.sum()
    return scores


def loo_auc(df, df_exp, all_labels, taxonomy, k):
    files = df['filename'].unique()
    aucs = []
    for val_file in files:
        train_exp = df_exp[df_exp['filename'] != val_file]
        global_prior, p_site, p_time, p_month = compute_priors(train_exp, all_labels, taxonomy, k)

        site, time_bin, month = parse_filename(val_file)
        scores = get_scores(site, time_bin, month, global_prior, p_site, p_time, p_month)

        val_exp = df_exp[df_exp['filename'] == val_file]
        present = set(val_exp['label'].unique())
        y_true = np.array([1 if l in present else 0 for l in all_labels])

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        aucs.append(roc_auc_score(y_true, scores))

    return np.mean(aucs)


def main():
    df = pd.read_csv(LABELS_PATH)
    taxonomy = pd.read_csv(TAXONOMY_PATH)

    df['site'], df['time_bin'], df['month'] = zip(*df['filename'].map(parse_filename))

    df_exp = df.copy()
    df_exp['label'] = df_exp['primary_label'].str.split(';')
    df_exp = df_exp.explode('label')

    all_labels = sorted(df_exp['label'].unique())
    print(f'{len(all_labels)} labels, {df["filename"].nunique()} files\n')

    # Baseline (no hierarchical, equivalent to k=inf)
    baseline = loo_auc(df, df_exp, all_labels, taxonomy, k=1000)
    print(f'Baseline (no hierarchical): AUC={baseline:.4f}\n')

    results = []
    for k in K_VALUES:
        auc = loo_auc(df, df_exp, all_labels, taxonomy, k)
        results.append((k, auc))
        print(f'k={k:4d}  AUC={auc:.4f}')

    best = max(results, key=lambda x: x[1])
    print(f'\nBest: k={best[0]}  AUC={best[1]:.4f}')


if __name__ == '__main__':
    main()
