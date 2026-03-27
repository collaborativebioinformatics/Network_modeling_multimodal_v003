# #!/usr/bin/env python3
“””
STEP 3 — EXPAND_NODES (core)

Iterative combinatorial expansion loop.

Key design principles:

- Anchor nodes are copied in descending degree order each round
- Similarity is checked combinatorially: single features first,
  then pairs, triples, up to max_combo_depth
- Higher dimensionality increases the likelihood of combination-driven
  edges — a combination fires if the JOINT similarity of the feature
  subset exceeds the threshold
- A synthetic node is created if EITHER:
  (a) Type A (genomic) combinatorial similarity fires, OR
  (b) Type B (proteomic) similarity fires (optional external signal)
- Each synthetic node records which features (and at what combo depth)
  drove its creation — used downstream for entropy scoring
- Termination: a full round produces no new synthetic nodes

Combinatorial similarity:
For a candidate synthetic vector s and anchor vector a, at combo depth k:

- Select the top-k features by absolute value in s
- Compute cosine similarity restricted to those k dimensions
- If sim >= threshold, the combination fires

This naturally captures the observation that higher-dimensional data
is more likely to be driven by feature combinations rather than
individual features.

Type B hook:
If –type_b_scores is provided (a Parquet file with columns
sample_id, anchor_id, similarity_score), a synthetic node is also
created for any anchor whose max Type B similarity >= threshold,
even if Type A did not fire. The association_source column records
‘type_a’, ‘type_b’, or ‘both’.
“””

import argparse
import json
import numpy as np
from itertools import combinations

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
print(”[INFO] RAPIDS not found — falling back to pandas / NumPy / NetworkX.”)

# —————————————————————————

# Graph helpers

# —————————————————————————

def build_graph(edges_df):
if GPU:
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source=‘source’, destination=‘target’)
else:
G = _nx_fallback.Graph()
for _, row in edges_df.iterrows():
G.add_edge(str(row[‘source’]), str(row[‘target’]))
return G

def get_degrees(G, node_ids):
if GPU:
deg_df  = G.degree()
deg_map = dict(zip(
deg_df[‘vertex’].to_pandas().tolist(),
deg_df[‘degree’].to_pandas().tolist()
))
else:
deg_map = dict(G.degree())
return {nid: deg_map.get(nid, 0) for nid in node_ids}

def add_edge(G, u, v, edges_df):
“”“Add an edge and return updated edge DataFrame.”””
import pandas as pd
new_row = pd.DataFrame({‘source’: [str(u)], ‘target’: [str(v)]})
if GPU:
new_cudf = cudf.DataFrame.from_pandas(new_row)
edges_df = cudf.concat([edges_df, new_cudf], ignore_index=True)
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source=‘source’, destination=‘target’)
else:
G.add_edge(str(u), str(v))
edges_df = pd.concat([edges_df, new_row], ignore_index=True)
return G, edges_df

# —————————————————————————

# Combinatorial similarity

# —————————————————————————

def top_k_indices(vec, k):
“”“Return indices of the k largest absolute values in vec.”””
abs_vec = np.abs(vec)
return np.argsort(abs_vec)[-k:]

def cosine_sim_subset(a, b, indices):
“”“Cosine similarity restricted to a subset of dimensions.”””
a_sub = a[indices]
b_sub = b[indices]
na = np.linalg.norm(a_sub)
nb = np.linalg.norm(b_sub)
if na == 0 or nb == 0:
return 0.0
return float(np.dot(a_sub, b_sub) / (na * nb))

def combinatorial_similarity(query_vec, anchor_vec, threshold,
max_combo_depth):
“””
Check if query_vec is similar to anchor_vec at any combination depth
from 1 to max_combo_depth.

```
At depth k, the top-k features (by absolute value in query) are
selected and cosine similarity is computed on that subset.

Returns (fired: bool, combo_depth: int, similarity: float,
         feature_indices: list)
Higher depth fires are only checked if lower depth does not fire,
since single-feature matches are most interpretable.
"""
n_dims = len(query_vec)
max_depth = min(max_combo_depth, n_dims)

for k in range(1, max_depth + 1):
    indices = top_k_indices(query_vec, k)
    sim = cosine_sim_subset(query_vec, anchor_vec, indices)
    if sim >= threshold:
        return True, k, sim, indices.tolist()

return False, 0, 0.0, []
```

def batch_combinatorial_similarity(query_vec, anchor_matrix, anchor_ids,
self_id, threshold, max_combo_depth):
“””
Run combinatorial similarity check of query_vec against all anchors.
Returns list of (anchor_id, combo_depth, similarity, feature_indices)
for all anchors that fire, excluding self.
“””
results = []
for i, anchor_id in enumerate(anchor_ids):
if anchor_id == self_id:
continue
anchor_vec = anchor_matrix[i]
fired, depth, sim, feat_idx = combinatorial_similarity(
query_vec, anchor_vec, threshold, max_combo_depth
)
if fired:
results.append((anchor_id, depth, sim, feat_idx))
return results

# —————————————————————————

# Synthetic vector generation

# —————————————————————————

def generate_synthetic_vector(anchor_vec, low_dim, seed=None):
“””
Returns:
high_dim : perturbed anchor vector (for similarity checking)
low_dim  : random projection embedding
Replace projection with PCA / UMAP in production.
“””
rng = np.random.default_rng(seed)
d = len(anchor_vec)
high_dim  = anchor_vec + rng.standard_normal(d).astype(np.float32) * 0.05
proj      = (rng.standard_normal((low_dim, d)) / np.sqrt(low_dim)).astype(np.float32)
low_dim_v = proj @ anchor_vec
return high_dim, low_dim_v

# —————————————————————————

# Main expansion loop

# —————————————————————————

def expand(graph_df, edges_df, similarity_threshold, low_dim,
max_rounds, max_combo_depth, type_b_df=None):

```
vector_cols = sorted(
    [c for c in graph_df.columns if c.startswith('v_')],
    key=lambda c: int(c.split('_')[1])
)

# Convert to pandas for numpy operations
graph_pd = graph_df.to_pandas() if GPU else graph_df
edges_pd = edges_df.to_pandas() if GPU else edges_df

anchor_ids = graph_pd['anchor_id'].tolist()
anchor_matrix = graph_pd[vector_cols].values.astype(np.float32)

# Build groups
groups = {
    aid: {
        'anchor_id':       aid,
        'high_dim_edge':   aid,   # anchor is its own edge key
        'edge_holder':     aid,
        'members':         [aid],
        'synthetic_count': 0,
        'combo_signatures': []    # list of {depth, features} dicts
    }
    for aid in anchor_ids
}

# Type B similarity lookup: {anchor_id: max_similarity}
type_b_map = {}
if type_b_df is not None:
    tb_pd = type_b_df.to_pandas() if GPU else type_b_df
    type_b_map = dict(zip(
        tb_pd['anchor_id'].tolist(),
        tb_pd['similarity_score'].tolist()
    ))

G = build_graph(
    cudf.DataFrame.from_pandas(edges_pd) if GPU else edges_pd
)

synthetic_rows = []
log_rows       = []

for round_num in range(1, max_rounds + 1):
    degrees = get_degrees(G, anchor_ids)
    ranked  = sorted(anchor_ids, key=lambda a: degrees.get(a, 0),
                     reverse=True)
    new_this_round = []

    for anchor_id in ranked:
        seed = round_num * 10000 + (
            anchor_id if isinstance(anchor_id, int)
            else hash(str(anchor_id)) % 10000
        )
        anchor_idx = anchor_ids.index(anchor_id)
        anchor_vec = anchor_matrix[anchor_idx]

        syn_high, syn_low = generate_synthetic_vector(
            anchor_vec, low_dim=low_dim, seed=seed
        )
        syn_id = f"syn_{anchor_id}_r{round_num}"

        # --- Type A check (combinatorial) ---
        type_a_matches = batch_combinatorial_similarity(
            syn_high, anchor_matrix, anchor_ids,
            anchor_id, similarity_threshold, max_combo_depth
        )
        type_a_fired = len(type_a_matches) > 0

        # --- Type B check (optional) ---
        type_b_sim   = type_b_map.get(anchor_id, 0.0)
        type_b_fired = type_b_sim >= similarity_threshold

        fires = type_a_fired or type_b_fired
        if type_a_fired and type_b_fired:
            assoc_source = 'both'
        elif type_a_fired:
            assoc_source = 'type_a'
        elif type_b_fired:
            assoc_source = 'type_b'
        else:
            assoc_source = 'none'

        # Best combo from Type A (shallowest depth, highest sim)
        best_combo = {}
        if type_a_matches:
            best = min(type_a_matches, key=lambda x: (x[1], -x[2]))
            best_combo = {
                'matched_anchor': best[0],
                'combo_depth':    best[1],
                'similarity':     round(best[2], 6),
                'feature_indices': best[3]
            }

        log_rows.append({
            'round':           round_num,
            'anchor_id':       anchor_id,
            'syn_id':          syn_id,
            'degree':          degrees.get(anchor_id, 0),
            'type_a_fired':    type_a_fired,
            'type_b_fired':    type_b_fired,
            'association_source': assoc_source,
            'combo_depth':     best_combo.get('combo_depth', 0),
            'max_similarity':  best_combo.get('similarity',
                               round(type_b_sim, 6)),
            'action':          'added' if fires else 'skipped'
        })

        if not fires:
            continue

        # Register synthetic node
        syn_row = {
            'anchor_id':          anchor_id,
            'id':                 syn_id,
            'kind':               'synthetic',
            'high_dim_edge':      anchor_id,
            'round':              round_num,
            'duplicate_edge':     False,
            'association_source': assoc_source,
            'combo_depth':        best_combo.get('combo_depth', 0),
            'combo_features':     json.dumps(
                best_combo.get('feature_indices', [])
            ),
            'degree':             1
        }
        for i, col in enumerate(vector_cols):
            syn_row[col] = float(syn_low[i]) if i < len(syn_low) else 0.0

        synthetic_rows.append(syn_row)
        groups[anchor_id]['members'].append(syn_id)
        groups[anchor_id]['synthetic_count'] += 1
        if best_combo:
            groups[anchor_id]['combo_signatures'].append(best_combo)

        G, edges_pd = add_edge(
            G, anchor_id, syn_id,
            cudf.DataFrame.from_pandas(edges_pd) if GPU else edges_pd
        )
        if GPU:
            edges_pd = edges_pd.to_pandas()

        new_this_round.append(syn_id)

    print(f"Round {round_num}: {len(new_this_round)} new synthetic node(s).")
    if not new_this_round:
        print(f"Converged after {round_num} round(s).")
        break
else:
    print(f"WARNING: reached max_rounds={max_rounds} without convergence.")

# -------------------------------------------------------------------
# Assemble output DataFrames
# -------------------------------------------------------------------
import pandas as pd

anchor_pd = graph_pd.copy()
anchor_pd['id']                 = anchor_pd['anchor_id']
anchor_pd['kind']               = 'anchor'
anchor_pd['high_dim_edge']      = anchor_pd['anchor_id']
anchor_pd['round']              = 0
anchor_pd['duplicate_edge']     = False
anchor_pd['association_source'] = 'type_a'
anchor_pd['combo_depth']        = 0
anchor_pd['combo_features']     = '[]'

if synthetic_rows:
    syn_pd = pd.DataFrame(synthetic_rows)
    for col in anchor_pd.columns:
        if col not in syn_pd.columns:
            syn_pd[col] = None
    all_nodes_pd = pd.concat(
        [anchor_pd[syn_pd.columns], syn_pd], ignore_index=True
    )
else:
    all_nodes_pd = anchor_pd

groups_pd = pd.DataFrame([
    {
        'anchor_id':       g['anchor_id'],
        'high_dim_edge':   g['high_dim_edge'],
        'edge_holder':     g['edge_holder'],
        'synthetic_count': g['synthetic_count'],
        'members':         json.dumps(g['members']),
        'combo_signatures': json.dumps(g['combo_signatures'])
    }
    for g in groups.values()
])

log_pd = pd.DataFrame(log_rows)

# Convert back to cuDF if GPU
to_df = lambda pd_df: cudf.DataFrame.from_pandas(pd_df) if GPU else pd_df
edges_out = cudf.DataFrame.from_pandas(edges_pd) if GPU else edges_pd

return (to_df(all_nodes_pd), edges_out,
        to_df(groups_pd), to_df(log_pd))
```

# —————————————————————————

# Entry point

# —————————————————————————

def main():
parser = argparse.ArgumentParser(description=“Expand synthetic nodes”)
parser.add_argument(’–graph’,           required=True)
parser.add_argument(’–edges’,           required=True)
parser.add_argument(’–similarity’,      type=float, required=True)
parser.add_argument(’–low_dim’,         type=int,   required=True)
parser.add_argument(’–max_rounds’,      type=int,   required=True)
parser.add_argument(’–max_combo_depth’, type=int,   default=3)
parser.add_argument(’–type_b_scores’,   default=None)
parser.add_argument(’–out_nodes’,       required=True)
parser.add_argument(’–out_edges’,       required=True)
parser.add_argument(’–out_groups’,      required=True)
parser.add_argument(’–out_log’,         required=True)
args = parser.parse_args()

```
graph_df = cudf.read_parquet(args.graph)
edges_df = cudf.read_parquet(args.edges)

type_b_df = None
if args.type_b_scores and args.type_b_scores != 'NO_FILE':
    type_b_df = cudf.read_parquet(args.type_b_scores)
    print(f"Type B scores loaded: {len(type_b_df)} rows.")

print(f"Loaded {len(graph_df)} anchor nodes, {len(edges_df)} edges.")

nodes_df, edges_out, groups_df, log_df = expand(
    graph_df, edges_df,
    similarity_threshold=args.similarity,
    low_dim=args.low_dim,
    max_rounds=args.max_rounds,
    max_combo_depth=args.max_combo_depth,
    type_b_df=type_b_df
)

nodes_df.to_parquet(args.out_nodes,  index=False)
edges_out.to_parquet(args.out_edges, index=False)
groups_df.to_parquet(args.out_groups, index=False)
log_df.to_parquet(args.out_log,      index=False)

n_syn = int((nodes_df['kind'] == 'synthetic').sum())
print(f"Expansion complete: {n_syn} synthetic nodes created.")
```

if **name** == ‘**main**’:
main()
