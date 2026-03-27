# #!/usr/bin/env python3
“””
STEP 2 — BUILD_GRAPH (core)

Constructs the initial anchor graph. Anchor nodes represent parent
nodes (e.g. ICD-10 disease codes — noted here for documentation
purposes only; the algorithm treats them as opaque identifiers).

Each anchor’s feature signature is the mean feature vector across
all CONDITION samples associated with that anchor, after outlier
exclusion (samples beyond 2 SD of the per-feature mean are excluded
from the signature computation, but remain in the dataset).

The graph structure (anchor-anchor edges) comes from the supplied
edge file and is independent of the feature data.

Outputs per anchor node:

- feature signature vector (mean of condition samples, float32)
- degree (from anchor-anchor edge list)
- sample count (number of condition samples contributing)
  “””

import argparse
import numpy as np

try:
import cudf
import cupy as cp
import cugraph
GPU = True
print(”[INFO] RAPIDS detected — running on GPU.”)
except ImportError:
import pandas as cudf
import numpy as cp
import networkx as _nx_fallback
GPU = False
print(”[INFO] RAPIDS not found — falling back to pandas / NetworkX.”)

def detect_delimiter(path):
import csv
with open(path, newline=’’) as f:
sample = f.read(4096)
sniffer = csv.Sniffer()
try:
return sniffer.sniff(sample, delimiters=’,\t’).delimiter
except csv.Error:
return ‘,’

def compute_anchor_signatures(samples_df, features_df, vector_cols,
n_sd_outlier=2.0):
“””
For each anchor (unique condition_label value that is NOT the control),
compute the mean feature vector across its condition samples,
excluding per-feature outliers beyond n_sd_outlier standard deviations.

```
Returns a cuDF DataFrame with columns:
    anchor_id, sample_count, v_0..v_n
"""
# Identify control label — the label shared by the most samples
# or explicitly the one NOT being treated as a disease condition.
# We take the label with fewer samples as condition (heuristic),
# or use 'control' string match if present.
labels = samples_df['condition_label'].unique().to_pandas().tolist()
control_label = next(
    (l for l in labels if 'control' in str(l).lower()), labels[0]
)
condition_label = [l for l in labels if l != control_label][0]

condition_ids = samples_df[
    samples_df['condition_label'] == condition_label
]['sample_id']

condition_features = features_df[
    features_df['sample_id'].isin(condition_ids)
][['sample_id'] + vector_cols]

# Outlier exclusion: for each feature, mask values beyond n_sd_outlier SD
# Uses pandas for the masking step (cuDF masked mean support varies)
cf_pd = condition_features[vector_cols].to_pandas() \
        if GPU else condition_features[vector_cols]

feat_mean = cf_pd.mean()
feat_std  = cf_pd.std().replace(0, 1e-9)
z_scores  = (cf_pd - feat_mean) / feat_std
mask      = z_scores.abs() <= n_sd_outlier
cf_masked = cf_pd.where(mask)
signature = cf_masked.mean().fillna(0).values.astype(np.float32)

sample_count = int(len(condition_ids))

# Build single-row anchor signature DataFrame
# (In production with multiple parent nodes, group by parent_node_id here)
import pandas as pd
sig_dict = {col: [float(signature[i])]
            for i, col in enumerate(vector_cols)}
sig_dict['anchor_id']    = [condition_label]
sig_dict['sample_count'] = [sample_count]
sig_dict['control_label'] = [control_label]

sig_pd = pd.DataFrame(sig_dict)
return cudf.DataFrame.from_pandas(sig_pd) if GPU else sig_pd
```

def build_graph(edges_df):
if GPU:
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source=‘source’, destination=‘target’)
else:
G = _nx_fallback.Graph()
for _, row in edges_df.iterrows():
G.add_edge(str(row[‘source’]), str(row[‘target’]))
return G

def get_degrees(G, anchor_ids):
if GPU:
deg_df  = G.degree()
deg_map = dict(zip(
deg_df[‘vertex’].to_pandas().tolist(),
deg_df[‘degree’].to_pandas().tolist()
))
else:
deg_map = dict(G.degree())
return {aid: deg_map.get(aid, 0) for aid in anchor_ids}

def main():
parser = argparse.ArgumentParser(description=“Build anchor graph”)
parser.add_argument(’–samples’,      required=True)
parser.add_argument(’–features’,     required=True)
parser.add_argument(’–edges’,        required=True)
parser.add_argument(’–out_graph’,    required=True)
parser.add_argument(’–out_degrees’,  required=True)
parser.add_argument(’–out_edges’,    required=True)
parser.add_argument(’–n_sd_outlier’, type=float, default=2.0)
args = parser.parse_args()

```
samples_df  = cudf.read_parquet(args.samples)
features_df = cudf.read_parquet(args.features)
edges_df    = cudf.read_csv(args.edges, sep=detect_delimiter(args.edges))
edges_df['source'] = edges_df['source'].astype(str)
edges_df['target'] = edges_df['target'].astype(str)

vector_cols = sorted(
    [c for c in features_df.columns if c.startswith('v_')],
    key=lambda c: int(c.split('_')[1])
)

# Compute anchor feature signatures
signatures_df = compute_anchor_signatures(
    samples_df, features_df, vector_cols, args.n_sd_outlier
)
anchor_ids = signatures_df['anchor_id'].tolist()

# Build graph
G = build_graph(edges_df)
degrees = get_degrees(G, anchor_ids)

# Attach degrees and kind
import pandas as pd
sig_pd = signatures_df.to_pandas() if GPU else signatures_df
sig_pd['degree'] = sig_pd['anchor_id'].map(degrees).fillna(0).astype(int)
sig_pd['kind']   = 'anchor'

graph_df  = cudf.DataFrame.from_pandas(sig_pd) if GPU else sig_pd
edges_out = cudf.DataFrame.from_pandas(
    edges_df.to_pandas() if GPU else edges_df
) if GPU else edges_df

# Degree table
deg_pd = pd.DataFrame([
    {'node_id': aid, 'degree': deg}
    for aid, deg in sorted(degrees.items(), key=lambda x: -x[1])
])
deg_df = cudf.DataFrame.from_pandas(deg_pd) if GPU else deg_pd

graph_df.to_parquet(args.out_graph,   index=False)
deg_df.to_parquet(args.out_degrees,   index=False)
edges_out.to_parquet(args.out_edges,  index=False)

print(f"Graph built: {len(anchor_ids)} anchor node(s), "
      f"{len(edges_df)} edge(s).")
print(f"Feature dimensions: {len(vector_cols)}")
print(f"Top degrees: {sorted(degrees.items(), key=lambda x:-x[1])[:5]}")
```

if **name** == ‘**main**’:
main()
