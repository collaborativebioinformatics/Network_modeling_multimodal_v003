# #!/usr/bin/env python3
“””
PROTEOMICS — PREPROCESS_TYPE_B

Computes per-feature control baselines and z-scores for proteomic
expression data, then produces a Type B similarity score per anchor
for consumption by EXPAND_NODES.

## Processing steps

1. Separate control and condition samples via condition_label.
1. Per-feature baseline (control samples only):
   a. Compute initial mean and SD across control samples.
   b. Exclude outlier control samples (|z| > n_sd_outlier per feature).
   c. Recompute mean and SD on the remaining control samples.
   d. Store as baseline_mean and baseline_sd per feature.
   Features with baseline_sd == 0 (no variation in control) are retained
   but z-scores will be 0 for those features — they contribute no signal
   to similarity.
1. Z-score condition samples:
   z_ij = (x_ij - baseline_mean_j) / baseline_sd_j
   where i = sample, j = feature.
   Features with baseline_sd == 0 get z = 0.
1. Anchor signature:
   The anchor z-score vector is the mean z-score across all condition
   samples, with a second round of outlier exclusion (|z_i| > n_sd_outlier
   for the per-sample mean z-score). This gives the “typical” deviation
   profile for this condition relative to control.
1. Per-sample cosine similarity:
   Compute cosine similarity between the anchor signature and each
   condition sample’s z-score vector. Report the mean similarity
   as the Type B score for this anchor.
   
   This score is passed to EXPAND_NODES where it is compared against
   params.similarity to determine whether Type B fires for a given
   synthetic node candidate.

## Outputs

type_b_scores.parquet : anchor_id, similarity_score, n_features_used,
n_condition_samples, n_control_samples
baseline_stats.parquet: feature_id, baseline_mean, baseline_sd,
n_control_used, excluded_outlier_count
zscores.parquet       : sample_id x feature z-score matrix
(condition samples only)
“””

import argparse
import numpy as np

try:
import cudf
import cupy as cp
GPU = True
print(”[INFO] RAPIDS detected — preprocessing on GPU.”)
except ImportError:
import pandas as cudf
import numpy as cp
GPU = False
print(”[INFO] RAPIDS not found — falling back to pandas / NumPy.”)

# —————————————————————————

# Baseline computation

# —————————————————————————

def compute_baseline(control_mat, n_sd_outlier):
“””
Compute per-feature baseline mean and SD from control sample matrix,
with two-pass outlier exclusion.

```
Parameters
----------
control_mat : np.ndarray, shape (n_control, n_features)
n_sd_outlier : float

Returns
-------
baseline_mean : np.ndarray (n_features,)
baseline_sd   : np.ndarray (n_features,)  — replaced 0s with 1e-9
n_used        : np.ndarray (n_features,)  — control samples used per feature
n_excluded    : np.ndarray (n_features,)  — outliers excluded per feature
"""
n_ctrl, n_feat = control_mat.shape

# Pass 1: initial mean and SD
mean1 = np.nanmean(control_mat, axis=0)
sd1   = np.nanstd(control_mat,  axis=0)
sd1   = np.where(sd1 == 0, 1e-9, sd1)

# Outlier mask: True = keep
z1   = np.abs((control_mat - mean1) / sd1)
keep = z1 <= n_sd_outlier   # shape (n_ctrl, n_feat)

# Pass 2: recompute mean/SD on non-outlier values
masked = np.where(keep, control_mat, np.nan)
baseline_mean = np.nanmean(masked, axis=0)
baseline_sd   = np.nanstd(masked,  axis=0)

n_used     = np.sum(keep, axis=0).astype(float)
n_excluded = n_ctrl - n_used

# Replace zero SD with small epsilon to avoid division by zero
baseline_sd = np.where(baseline_sd == 0, 1e-9, baseline_sd)

return baseline_mean, baseline_sd, n_used, n_excluded
```

# —————————————————————————

# Z-scoring

# —————————————————————————

def zscore_samples(sample_mat, baseline_mean, baseline_sd):
“””
Z-score a sample matrix against baseline_mean and baseline_sd.

```
Parameters
----------
sample_mat   : np.ndarray (n_samples, n_features)
baseline_mean: np.ndarray (n_features,)
baseline_sd  : np.ndarray (n_features,)

Returns
-------
z_mat : np.ndarray (n_samples, n_features)
"""
return (sample_mat - baseline_mean) / baseline_sd
```

# —————————————————————————

# Anchor signature

# —————————————————————————

def compute_anchor_signature(z_mat, n_sd_outlier):
“””
Compute the anchor z-score signature as the mean z-score vector
across condition samples, with outlier exclusion on per-sample
mean absolute z-score.

```
Samples whose mean |z| across all features exceeds n_sd_outlier
relative to other condition samples are excluded. Note: a sample
that is a uniform outlier in magnitude but consistent in direction
will not be excluded — this is intentional, as it represents a
genuine (if extreme) expression of the same subcondition profile.
are excluded as outlier profiles.

Returns
-------
signature      : np.ndarray (n_features,)
n_samples_used : int
"""
per_sample_mean_absz = np.mean(np.abs(z_mat), axis=1)
global_mean = np.mean(per_sample_mean_absz)
global_sd   = np.std(per_sample_mean_absz)
global_sd   = global_sd if global_sd > 0 else 1e-9

keep = np.abs(
    (per_sample_mean_absz - global_mean) / global_sd
) <= n_sd_outlier

z_kept = z_mat[keep]
signature = np.mean(z_kept, axis=0) if len(z_kept) > 0 \
            else np.mean(z_mat, axis=0)
return signature, int(np.sum(keep))
```

# —————————————————————————

# Cosine similarity

# —————————————————————————

def cosine_similarity_batch(signature, z_mat):
“””
Compute cosine similarity between signature and each row of z_mat.
Returns mean similarity across all samples.
“””
sig_norm = np.linalg.norm(signature)
if sig_norm == 0:
return 0.0

```
row_norms = np.linalg.norm(z_mat, axis=1)
row_norms = np.where(row_norms == 0, 1e-9, row_norms)
sims = (z_mat @ signature) / (row_norms * sig_norm)
return float(np.mean(sims))
```

# —————————————————————————

# Main

# —————————————————————————

def main():
parser = argparse.ArgumentParser(
description=“Preprocess Type B proteomics data”
)
parser.add_argument(’–features’,     required=True)   # validated .parquet
parser.add_argument(’–samples’,      required=True)   # core samples .parquet
parser.add_argument(’–n_sd_outlier’, type=float, default=2.0)
parser.add_argument(’–similarity’,   type=float, default=0.8)
parser.add_argument(’–out_scores’,   required=True)
parser.add_argument(’–out_baseline’, required=True)
parser.add_argument(’–out_zscores’,  required=True)
args = parser.parse_args()

```
feat_df    = cudf.read_parquet(args.features)
samples_df = cudf.read_parquet(args.samples)

# Ensure condition_label is present (join if missing)
if 'condition_label' not in feat_df.columns:
    samples_df['sample_id'] = samples_df['sample_id'].astype(str)
    feat_df['sample_id']    = feat_df['sample_id'].astype(str)
    feat_df = feat_df.merge(
        samples_df[['sample_id', 'condition_label']],
        on='sample_id', how='left'
    )

feat_pd = feat_df.to_pandas() if GPU else feat_df

vector_cols = sorted(
    [c for c in feat_pd.columns if c.startswith('v_')],
    key=lambda c: int(c.split('_')[1])
)

# Identify control label
labels = feat_pd['condition_label'].unique().tolist()
control_label = next(
    (l for l in labels if 'control' in str(l).lower()), labels[0]
)
condition_labels = [l for l in labels if l != control_label]

print(f"Control label    : {control_label}")
print(f"Condition labels : {condition_labels}")
print(f"Feature dims     : {len(vector_cols)}")

control_pd   = feat_pd[feat_pd['condition_label'] == control_label]
condition_pd = feat_pd[feat_pd['condition_label'] != control_label]

control_mat   = control_pd[vector_cols].values.astype(np.float32)
condition_mat = condition_pd[vector_cols].values.astype(np.float32)
condition_ids = condition_pd['sample_id'].tolist()

# -------------------------------------------------------------------
# Step 1: Compute baseline from control
# -------------------------------------------------------------------
print(f"Computing baseline from {len(control_mat)} control samples...")
baseline_mean, baseline_sd, n_used, n_excluded = compute_baseline(
    control_mat, args.n_sd_outlier
)

# -------------------------------------------------------------------
# Step 2: Z-score condition samples
# -------------------------------------------------------------------
print(f"Z-scoring {len(condition_mat)} condition samples...")
z_mat = zscore_samples(condition_mat, baseline_mean, baseline_sd)

# -------------------------------------------------------------------
# Step 3: Anchor signature (mean z-score, outlier-excluded)
# -------------------------------------------------------------------
signature, n_sig_samples = compute_anchor_signature(
    z_mat, args.n_sd_outlier
)
print(f"Anchor signature computed from {n_sig_samples} samples "
      f"(after outlier exclusion).")

# -------------------------------------------------------------------
# Step 4: Per-sample cosine similarity → anchor score
# -------------------------------------------------------------------
mean_sim = cosine_similarity_batch(signature, z_mat)
print(f"Anchor Type B similarity score: {mean_sim:.4f} "
      f"(threshold: {args.similarity})")

# -------------------------------------------------------------------
# Build output DataFrames
# -------------------------------------------------------------------
import pandas as pd

# One row per anchor (single condition for now;
# extend to multi-condition by groupby condition_label)
anchor_id = condition_labels[0] if condition_labels else 'condition'
scores_pd = pd.DataFrame([{
    'anchor_id':           anchor_id,
    'similarity_score':    round(mean_sim, 6),
    'n_features_used':     len(vector_cols),
    'n_condition_samples': len(condition_mat),
    'n_control_samples':   len(control_mat),
    'n_sig_samples':       n_sig_samples
}])

baseline_pd = pd.DataFrame({
    'feature_id':            vector_cols,
    'baseline_mean':         baseline_mean.tolist(),
    'baseline_sd':           baseline_sd.tolist(),
    'n_control_used':        n_used.tolist(),
    'excluded_outlier_count': n_excluded.tolist()
})

zscores_pd = pd.DataFrame(z_mat, columns=vector_cols)
zscores_pd.insert(0, 'sample_id', condition_ids)

# Convert to cuDF if GPU
to_df = lambda df: cudf.DataFrame.from_pandas(df) if GPU else df

to_df(scores_pd).to_parquet(args.out_scores,   index=False)
to_df(baseline_pd).to_parquet(args.out_baseline, index=False)
to_df(zscores_pd).to_parquet(args.out_zscores,  index=False)

print(f"Type B preprocessing complete.")
print(f"  Baseline stats   → {args.out_baseline}")
print(f"  Z-scores         → {args.out_zscores}")
print(f"  Anchor scores    → {args.out_scores}")
```

if **name** == ‘**main**’:
main()
