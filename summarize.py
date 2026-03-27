# #!/usr/bin/env python3
“””
STEP 5 — SUMMARIZE (core)

Reads all upstream Parquet outputs via cuDF and writes final TSV
deliverables and a human-readable summary report.

New fields vs original:

- association_source : type_a, type_b, or both
- combo_depth        : feature combination depth that drove association
- combo_signatures   : feature indices in winning combination per group
  “””

import argparse
import json
from datetime import datetime

try:
import cudf
GPU = True
print(”[INFO] RAPIDS detected — summarising on GPU.”)
except ImportError:
import pandas as cudf
GPU = False
print(”[INFO] RAPIDS not found — falling back to pandas.”)

def to_tsv(df, path):
df.to_csv(path, sep=’\t’, index=False)

def main():
parser = argparse.ArgumentParser(description=“Summarize results”)
parser.add_argument(’–nodes’,       required=True)
parser.add_argument(’–edges’,       required=True)
parser.add_argument(’–groups’,      required=True)
parser.add_argument(’–scores’,      required=True)
parser.add_argument(’–duplicates’,  required=True)
parser.add_argument(’–log’,         required=True)
parser.add_argument(’–out_nodes’,   required=True)
parser.add_argument(’–out_edges’,   required=True)
parser.add_argument(’–out_groups’,  required=True)
parser.add_argument(’–out_report’,  required=True)
args = parser.parse_args()

```
nodes_df     = cudf.read_parquet(args.nodes)
edges_df     = cudf.read_parquet(args.edges)
groups_df    = cudf.read_parquet(args.groups)
scores_df    = cudf.read_parquet(args.scores)
duplicate_df = cudf.read_parquet(args.duplicates)
log_df       = cudf.read_parquet(args.log)

# Degree computation from edge list
import pandas as pd
edges_pd = edges_df.to_pandas() if GPU else edges_df
if len(edges_pd):
    src_deg = edges_pd.groupby('source').size().rename('deg')
    tgt_deg = edges_pd.groupby('target').size().rename('deg')
    deg_series = pd.concat([src_deg, tgt_deg]).groupby(level=0).sum()
    deg_map = deg_series.to_dict()
else:
    deg_map = {}

nodes_pd = nodes_df.to_pandas() if GPU else nodes_df
nodes_pd['degree'] = nodes_pd['id'].astype(str).map(deg_map).fillna(0).astype(int)

# Final node output
node_cols = ['id', 'kind', 'anchor_id', 'high_dim_edge', 'degree',
             'duplicate_edge', 'association_source',
             'combo_depth', 'combo_features', 'round']
node_cols = [c for c in node_cols if c in nodes_pd.columns]
to_tsv(nodes_pd[node_cols], args.out_nodes)

# Final edge output
to_tsv(edges_df, args.out_edges)

# Final groups output
groups_pd = groups_df.to_pandas() if GPU else groups_df
group_cols = ['anchor_id', 'high_dim_edge', 'edge_holder',
              'synthetic_count', 'entropy', 'duplicate_edge',
              'association_source', 'combo_signatures']
group_cols = [c for c in group_cols if c in groups_pd.columns]
to_tsv(groups_pd[group_cols], args.out_groups)

# Statistics
n_anchors   = int((nodes_pd['kind'] == 'anchor').sum())
n_synthetic = int((nodes_pd['kind'] == 'synthetic').sum())
n_edges     = len(edges_pd)
n_dupes     = len(duplicate_df)

log_pd = log_df.to_pandas() if GPU else log_df
rounds_run = int(log_pd['round'].max()) if len(log_pd) else 0
n_added    = int((log_pd['action'] == 'added').sum()) if len(log_pd) else 0
n_skipped  = int((log_pd['action'] == 'skipped').sum()) if len(log_pd) else 0

# Association source breakdown
type_a_only = int((nodes_pd['association_source'] == 'type_a').sum()) \
              if 'association_source' in nodes_pd.columns else 0
type_b_only = int((nodes_pd['association_source'] == 'type_b').sum()) \
              if 'association_source' in nodes_pd.columns else 0
both        = int((nodes_pd['association_source'] == 'both').sum()) \
              if 'association_source' in nodes_pd.columns else 0

# Combo depth breakdown
combo_depths = {}
if 'combo_depth' in log_pd.columns:
    added_log = log_pd[log_pd['action'] == 'added']
    combo_depths = added_log['combo_depth'].value_counts().to_dict()

with open(args.out_report, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("  SYNTHETIC NODE EXPANSION — SUMMARY REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"  Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  GPU backend : {GPU}\n")
    f.write("\n")
    f.write("GRAPH\n")
    f.write(f"  Anchor nodes      : {n_anchors}\n")
    f.write(f"  Synthetic nodes   : {n_synthetic}\n")
    f.write(f"  Total nodes       : {n_anchors + n_synthetic}\n")
    f.write(f"  Total edges       : {n_edges}\n")
    f.write("\n")
    f.write("EXPANSION\n")
    f.write(f"  Rounds run        : {rounds_run}\n")
    f.write(f"  Nodes added       : {n_added}\n")
    f.write(f"  Copies skipped    : {n_skipped}\n")
    if combo_depths:
        f.write(f"  Combo depth breakdown:\n")
        for depth in sorted(combo_depths):
            f.write(f"    depth {depth}: {combo_depths[depth]} node(s)\n")
    f.write("\n")
    f.write("ASSOCIATION SOURCES\n")
    f.write(f"  Type A only       : {type_a_only}\n")
    f.write(f"  Type B only       : {type_b_only}\n")
    f.write(f"  Both              : {both}\n")
    f.write("\n")
    f.write("EDGE ASSIGNMENT\n")
    f.write(f"  Groups            : {len(groups_pd)}\n")
    f.write(f"  Duplicated edges  : {n_dupes}\n")
    f.write("\n")
    f.write("OUTPUT FILES\n")
    f.write("  final_nodes.tsv  — id, kind, combo_depth, assoc_source\n")
    f.write("  final_edges.tsv  — all edges\n")
    f.write("  final_groups.tsv — groups with entropy + combo signatures\n")
    f.write("=" * 60 + "\n")

print(f"Summary complete: {n_anchors} anchors | {n_synthetic} synthetic | "
      f"{n_edges} edges | {n_dupes} duplicated edges.")
```

if **name** == ‘**main**’:
main()
