# #!/usr/bin/env python3
“””
STEP 1 — VALIDATE_INPUT (core)

Validates and standardises two input files:

1. Sample metadata  (sample_id, condition_label)
1. Type A feature matrix (samples x features, binary/count)

Both are read via cuDF (pandas fallback) and written as Parquet
for GPU-resident processing throughout the pipeline.

The condition_label column must contain exactly two distinct values:

- One representing the CONDITION (parent node / e.g. ICD-10 code)
- One representing the CONTROL (true baseline)

Feature columns must be named v_0, v_1, … v_n.
Higher-order combinations of these features drive edge creation in
EXPAND_NODES; individual feature columns are preserved here as-is.

NOTE: Parent node identity (e.g. ICD-10 code) is carried in the
sample metadata but is not used by the algorithm — it appears in
documentation only.
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

def validate_samples(df, errors):
“””
Validate sample metadata DataFrame.
Requires: sample_id, condition_label
condition_label must have exactly 2 distinct values.
Returns clean DataFrame.
“””
required = {‘sample_id’, ‘condition_label’}
missing = required - set(df.columns)
if missing:
sys.exit(f”ERROR: samples file missing columns: {missing}\n”
f”Found: {list(df.columns)}”)

```
# Null checks
for col in required:
    nulls = int(df[col].isnull().sum())
    if nulls:
        errors.append(f"Samples: {nulls} null value(s) in '{col}'")

# Duplicate sample IDs
n_dupes = len(df) - int(df['sample_id'].nunique())
if n_dupes:
    errors.append(f"Samples: {n_dupes} duplicate sample_id(s)")

# Condition label must have exactly 2 values
n_labels = int(df['condition_label'].nunique())
if n_labels != 2:
    errors.append(
        f"Samples: condition_label must have exactly 2 distinct values "
        f"(condition and control), found {n_labels}: "
        f"{df['condition_label'].unique().tolist()}"
    )

clean = df.dropna(subset=list(required)).copy()
clean['sample_id'] = clean['sample_id'].astype(str)
return clean
```

def validate_features(df, valid_sample_ids, errors):
“””
Validate Type A feature matrix.
Requires: sample_id column + v_0..v_n feature columns.
All feature values cast to float32.
Returns (clean_df, vector_cols).
“””
if ‘sample_id’ not in df.columns:
sys.exit(“ERROR: features file must contain a ‘sample_id’ column.”)

```
vector_cols = sorted(
    [c for c in df.columns if c.startswith('v_')],
    key=lambda c: int(c.split('_')[1])
)
if not vector_cols:
    sys.exit("ERROR: features file must contain columns named v_0, v_1, ...")

# Referential integrity
df['sample_id'] = df['sample_id'].astype(str)
bad = df[~df['sample_id'].isin(valid_sample_ids)]
if len(bad):
    errors.append(
        f"Features: {len(bad)} sample_id(s) not in samples file"
    )

# Null checks per feature
for col in vector_cols:
    nulls = int(df[col].isnull().sum())
    if nulls:
        errors.append(f"Features: {nulls} null(s) in column '{col}'")

clean = df[df['sample_id'].isin(valid_sample_ids)].copy()
clean = clean.dropna(subset=vector_cols)
for col in vector_cols:
    clean[col] = clean[col].astype('float32')

return clean, vector_cols
```

def main():
parser = argparse.ArgumentParser(description=“Validate core pipeline inputs”)
parser.add_argument(’–samples’,      required=True)
parser.add_argument(’–features’,     required=True)
parser.add_argument(’–out_samples’,  required=True)
parser.add_argument(’–out_features’, required=True)
parser.add_argument(’–report’,       required=True)
args = parser.parse_args()

```
all_errors = []

# Read inputs
samples_df  = cudf.read_csv(args.samples,  sep=detect_delimiter(args.samples))
features_df = cudf.read_csv(args.features, sep=detect_delimiter(args.features))
print(f"Read {len(samples_df)} samples, "
      f"{len(features_df)} feature rows from CSV/TSV.")

# Validate
samples_clean = validate_samples(samples_df, all_errors)
valid_ids = samples_clean['sample_id']
features_clean, vector_cols = validate_features(
    features_df, valid_ids, all_errors
)

# Report
labels = samples_clean['condition_label'].unique().tolist()
with open(args.report, 'w') as f:
    f.write("VALIDATION REPORT — CORE (Type A)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Samples          : {len(samples_clean)}\n")
    f.write(f"Condition labels : {labels}\n")
    f.write(f"Feature rows     : {len(features_clean)}\n")
    f.write(f"Feature dims     : {len(vector_cols)}\n")
    f.write(f"GPU backend      : {GPU}\n")
    f.write(f"Errors           : {len(all_errors)}\n")
    if all_errors:
        f.write("\nERRORS:\n")
        for e in all_errors:
            f.write(f"  {e}\n")
    else:
        f.write("\nAll checks passed.\n")

if all_errors:
    print(f"VALIDATION FAILED — {len(all_errors)} error(s).", file=sys.stderr)
    sys.exit(1)

samples_clean.to_parquet(args.out_samples,  index=False)
features_clean.to_parquet(args.out_features, index=False)
print(f"Validation passed. Parquet written: "
      f"{args.out_samples}, {args.out_features}")
```

if **name** == ‘**main**’:
main()
