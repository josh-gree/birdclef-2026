"""
Leave-one-out CV to tune Gaussian smoothing sigmas for the crazy Bayes prior submission.
Sweeps time bin resolution (1h, 2h, 6h), TIME_SIGMA and MONTH_SIGMA.
Reports macro ROC-AUC for each combo.
"""

import re
import itertools
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

LABELS_PATH = 'scratch/train_soundscapes_labels.csv'

# Time bin sizes in hours -> (n_bins, bin_size_hours)
TIME_BIN_SIZES = [1, 2, 6]

# Sigma grids to sweep (in units of time bins)
TIME_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
MONTH_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0]

LOG_EPS = 1e-10


def parse_filename(fn, bin_size=1):
    m = re.match(r'.*_(S\d+)_(\d{8})_(\d{6})\.ogg', fn)
    if m:
        site = m.group(1)
        month = int(m.group(2)[4:6])
        hour = int(m.group(3)[:2])
        time_bin = hour // bin_size
        return site, time_bin, month
    return None, None, None


def smooth_circular(counts_df, n_bins, all_labels, sigma):
    arr = counts_df.reindex(index=range(n_bins), columns=all_labels, fill_value=0).astype(float).values
    arr_tiled = np.tile(arr, (3, 1))
    arr_smoothed = gaussian_filter1d(arr_tiled, sigma=sigma, axis=0)
    arr_smoothed = arr_smoothed[n_bins:2 * n_bins]
    arr_smoothed += 1  # Laplace smoothing
    df_smoothed = pd.DataFrame(arr_smoothed, index=range(n_bins), columns=all_labels)
    return df_smoothed.div(df_smoothed.sum(axis=1), axis=0)


def compute_priors(df_exp, all_labels, n_bins, time_sigma, month_sigma):
    # Global
    global_counts = df_exp['label'].value_counts()
    global_prior = pd.Series(0.0, index=all_labels)
    global_prior.update(global_counts)
    global_prior = (global_prior + 1) / (global_prior + 1).sum()

    # Site
    site_counts = df_exp.groupby(['site', 'label']).size().unstack(fill_value=0)
    site_counts = site_counts.reindex(columns=all_labels, fill_value=0) + 1
    p_site = site_counts.div(site_counts.sum(axis=1), axis=0)

    # Time bin
    time_counts = df_exp.groupby(['time_bin', 'label']).size().unstack(fill_value=0)
    p_time = smooth_circular(time_counts, n_bins, all_labels, time_sigma)

    # Month
    month_counts = df_exp.groupby(['month', 'label']).size().unstack(fill_value=0)
    month_counts.index = month_counts.index - 1
    p_month = smooth_circular(month_counts, 12, all_labels, month_sigma)

    return global_prior, p_site, p_time, p_month


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


def loo_auc(df, df_exp, all_labels, bin_size, time_sigma, month_sigma):
    n_bins = 24 // bin_size
    files = df['filename'].unique()
    aucs = []

    for val_file in files:
        train_exp = df_exp[df_exp['filename'] != val_file]

        global_prior, p_site, p_time, p_month = compute_priors(
            train_exp, all_labels, n_bins, time_sigma, month_sigma
        )

        site, time_bin, month = parse_filename(val_file, bin_size)
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

    df_exp = df.copy()
    df_exp['label'] = df_exp['primary_label'].str.split(';')
    df_exp = df_exp.explode('label')

    all_labels = sorted(df_exp['label'].unique())
    print(f'{len(all_labels)} labels, {df["filename"].nunique()} files\n')

    results = []
    combos = list(itertools.product(TIME_BIN_SIZES, TIME_SIGMAS, MONTH_SIGMAS))
    for i, (bs, ts, ms) in enumerate(combos):
        # Add time_bin column for this bin size
        df_exp['site'], df_exp['time_bin'], df_exp['month'] = zip(
            *df_exp['filename'].map(lambda fn: parse_filename(fn, bs))
        )
        df['site'], df['time_bin'], df['month'] = zip(
            *df['filename'].map(lambda fn: parse_filename(fn, bs))
        )
        auc = loo_auc(df, df_exp, all_labels, bs, ts, ms)
        results.append((bs, ts, ms, auc))
        print(f'[{i+1}/{len(combos)}] bin={bs}h  time_sigma={ts:.1f}  month_sigma={ms:.1f}  AUC={auc:.4f}')

    results.sort(key=lambda x: -x[3])
    print('\n--- Top 10 ---')
    for bs, ts, ms, auc in results[:10]:
        print(f'  bin={bs}h  time_sigma={ts:.1f}  month_sigma={ms:.1f}  AUC={auc:.4f}')

    best = results[0]
    print(f'\nBest: bin={best[0]}h  time_sigma={best[1]}  month_sigma={best[2]}  AUC={best[3]:.4f}')


if __name__ == '__main__':
    main()
