# #!/usr/bin/env python3
“””
PROTEOMICS — VALIDATE_TYPE_B

Validates the raw proteomic expression matrix against the core
sample table. Ensures:

- sample_id column present and matched to core samples
- Feature columns named v_0..v_n (raw expression, not z-scored)
- No entirely-zero feature columns (likely unmeasured assay slots)
- Condition/control split is preserved via join with core samples

This module is optional — the pipeline runs without it if
–proteomics_features is not provided.

Raw values represent absolute expression levels (~3k features typical
for this data type, though the module is agnostic to feature count).
Baseline subtraction and z-scoring happen in PREPROCESS_TYPE_B.
“””

import argparse
import sys

try:
import cudf
GPU = True
print(”[INFO] RAPIDS detected — reading with cuDF.”)
except ImportError:
import pandas as cudf
GPU = False
print(”[INFO] RAPIDS not found — falling back to pandas.”)

def detect_delimiter(path):
import csv
with open(path, newline=’’) as f:
sample = f.read(4096)
sniffer = csv.Sniffer()
try:
return sniffer.sniff(sample, delimiters=’,\t’).delimiter
except csv.Error:
return ‘,’

def main():
parser = argparse.ArgumentParser(description=“Validate Type B proteomics input”)
parser.add_argument(’–features’,     required=True)
parser.add_argument(’–core_samples’, required=True)
parser.add_argument(’–out_features’, required=True)
parser.add_argument(’–report’,       required=True)
args = parser.parse_args()

```
errors = []

# Read inputs
feat_df    = cudf.read_csv(args.features, sep=detect_delimiter(args.features))
samples_df = cudf.read_parquet(args.core_samples)

print(f"Read {len(feat_df)} proteomics rows, "
      f"{len(samples_df)} core samples.")

# Check sample_id column
if 'sample_id' not in feat_df.columns:
    sys.exit("ERROR: proteomics features file must contain 'sample_id' column.")

feat_df['sample_id'] = feat_df['sample_id'].astype(str)
samples_df['sample_id'] = samples_df['sample_id'].astype(str)

# Feature columns
vector_cols = sorted(
    [c for c in feat_df.columns if c.startswith('v_')],
    key=lambda c: int(c.split('_')[1])
)
if not vector_cols:
    sys.exit("ERROR: proteomics features must contain columns v_0, v_1, ...")

# Referential integrity — all proteomics samples must be in core
core_ids   = samples_df['sample_id']
bad_ids    = feat_df[~feat_df['sample_id'].isin(core_ids)]
if len(bad_ids):
    errors.append(
        f"Proteomics: {len(bad_ids)} sample_id(s) not in core samples table"
    )

# Check for entirely-zero feature columns (unmeasured assay slots)
feat_pd = feat_df[vector_cols].to_pandas() if GPU else feat_df[vector_cols]
zero_cols = [c for c in vector_cols if (feat_pd[c] == 0).all()]
if zero_cols:
    errors.append(
        f"Proteomics: {len(zero_cols)} entirely-zero feature column(s) — "
        f"likely unmeasured. These will be excluded from similarity scoring."
    )

# Null checks
for col in vector_cols:
    nulls = int(feat_df[col].isnull().sum())
    if nulls:
        errors.append(f"Proteomics: {nulls} null(s) in column '{col}'")

# Join condition_label from core samples so downstream has it
clean = feat_df[feat_df['sample_id'].isin(core_ids)].copy()
clean = clean.merge(
    samples_df[['sample_id', 'condition_label']],
    on='sample_id', how='left'
)

# Remove entirely-zero columns
active_cols = [c for c in vector_cols if c not in zero_cols]
keep_cols   = ['sample_id', 'condition_label'] + active_cols
clean = clean[keep_cols]

# Cast to float32
for col in active_cols:
    clean[col] = clean[col].astype('float32')

# Identify control and condition counts
labels = clean['condition_label'].unique().tolist()
control_label = next(
    (l for l in labels if 'control' in str(l).lower()), labels[0]
)
n_control   = int((clean['condition_label'] == control_label).sum())
n_condition = int((clean['condition_label'] != control_label).sum())

# Report
with open(args.report, 'w') as f:
    f.write("VALIDATION REPORT — PROTEOMICS (Type B)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total samples       : {len(clean)}\n")
    f.write(f"  Condition         : {n_condition}\n")
    f.write(f"  Control           : {n_control}\n")
    f.write(f"Feature dims (raw)  : {len(vector_cols)}\n")
    f.write(f"Feature dims (used) : {len(active_cols)}\n")
    f.write(f"Zero cols removed   : {len(zero_cols)}\n")
    f.write(f"GPU backend         : {GPU}\n")
    f.write(f"Errors              : {len(errors)}\n")
    if errors:
        f.write("\nERRORS:\n")
        for e in errors:
            f.write(f"  {e}\n")
    else:
        f.write("\nAll checks passed.\n")

hard_errors = [e for e in errors if 'zero' not in e.lower()]
if hard_errors:
    print(f"VALIDATION FAILED — {len(hard_errors)} error(s).", file=sys.stderr)
    sys.exit(1)

clean.to_parquet(args.out_features, index=False)
print(f"Proteomics validation passed: {len(clean)} samples, "
      f"{len(active_cols)} features.")
```

if **name** == ‘**main**’:
main()
