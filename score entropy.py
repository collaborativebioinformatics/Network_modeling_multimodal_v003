# #!/usr/bin/env python3
“””
STEP 4 — SCORE_ENTROPY (core)

Shannon entropy scoring and edge assignment.

Key inversion from original pipeline:

- ORIGINAL (Type A genomic): exclusivity is the default.
  Duplication allowed only when H >= threshold (edge genuinely split).
- TYPE B proteomic / combinatorial: duplication is the default.
  Exclusivity enforced only when H < threshold (edge strongly belongs
  to one group). This reflects that non-exclusive associations are
  more biologically common in this context.

The behaviour is controlled per-node by the ‘association_source’ column:

- ‘type_a’ only  → original exclusive logic (duplicate if H >= theta)
- ‘type_b’ only  → inverted logic (exclusive if H < theta)
- ‘both’         → inverted logic (Type B permissiveness takes precedence
  when both signals are present)

Additionally tracks:

- combo_depth: the feature combination depth that drove the association
- combo_features: which feature indices were in the winning combination
- association_source: type_a, type_b, or both

*** INTEGRATION POINT ***
Replace compute_cooccurrence_scores() with a lookup into your actual
co-occurrence frequency data when available.
“””

import argparse
import json
import math

try:
import cudf
GPU = True
print(”[INFO] RAPIDS detected — scoring on GPU.”)
except ImportError:
import pandas as cudf
GPU = False
print(”[INFO] RAPIDS not found — falling back to pandas.”)

def shannon_entropy(counts):
“”“H = -sum(p * log2(p)) for p > 0. Returns float (bits).”””
total = sum(counts)
if total == 0:
return 0.0
h = 0.0
for c in counts:
if c > 0:
p = c / total
h -= p * math.log2(p)
return h

def build_neighbor_index(edges_df):
“”“Build {node_id: set(neighbors)} from cuDF edge DataFrame.”””
fwd = edges_df[[‘source’, ‘target’]].rename(
columns={‘source’: ‘node’, ‘target’: ‘neighbor’}
)
rev = edges_df[[‘target’, ‘source’]].rename(
columns={‘target’: ‘node’, ‘source’: ‘neighbor’}
)
all_edges = cudf.concat([fwd, rev], ignore_index=True)
all_pd    = all_edges.to_pandas() if GPU else all_edges
return all_pd.groupby(‘node’)[‘neighbor’].apply(set).to_dict()

def compute_cooccurrence_scores(members, neighbor_index, high_dim_edge):
“””
Proxy co-occurrence: count how many of each member’s neighbors
are also members of the same anchor group.

```
*** INTEGRATION POINT ***
Replace with a lookup into your actual co-occurrence frequency table.
For Type B data this should incorporate the z-score magnitude of
the matched features, not just presence/absence.
"""
member_set = set(str(m) for m in members)
scores = {}
for member in members:
    nbrs = neighbor_index.get(str(member), set())
    scores[member] = len(nbrs & member_set)
return scores
```

def should_duplicate(entropy, threshold, association_source):
“””
Determine duplication eligibility based on entropy and data type.

```
Type A (genomic): duplicate if H >= threshold  (original logic)
Type B / both:    duplicate if H >= 0           (default duplicate,
                  exclusive only if H < threshold)
"""
if association_source == 'type_a':
    return entropy >= threshold
else:
    # type_b or both: non-exclusive by default
    return entropy >= 0.0    # always duplicate unless explicitly blocked
```

def score_and_reassign(nodes_df, edges_df, groups_df, entropy_threshold):
neighbor_index = build_neighbor_index(edges_df)
groups_pd = groups_df.to_pandas() if GPU else groups_df
nodes_pd  = nodes_df.to_pandas()  if GPU else nodes_df

```
# Build association_source lookup per anchor
assoc_source_map = {}
if 'association_source' in nodes_pd.columns:
    anchor_nodes = nodes_pd[nodes_pd['kind'] == 'anchor']
    assoc_source_map = dict(zip(
        anchor_nodes['anchor_id'].tolist(),
        anchor_nodes['association_source'].tolist()
    ))

score_rows    = []
updated_rows  = []
duplicate_rows = []

for _, group in groups_pd.iterrows():
    anchor_id     = group['anchor_id']
    high_dim_edge = group['high_dim_edge']
    members       = json.loads(group['members'])
    assoc_source  = assoc_source_map.get(anchor_id, 'type_a')

    scores = compute_cooccurrence_scores(
        members, neighbor_index, high_dim_edge
    )
    counts = list(scores.values())
    h = shannon_entropy(counts)

    best_member = max(scores, key=scores.get)
    reassigned  = best_member != group['edge_holder']
    duplicate   = should_duplicate(h, entropy_threshold, assoc_source)

    sorted_members = sorted(scores.items(), key=lambda x: -x[1])
    top_two = [m for m, _ in sorted_members[:2]]

    for member, score in scores.items():
        score_rows.append({
            'anchor_id':          anchor_id,
            'high_dim_edge':      high_dim_edge,
            'member':             member,
            'cooccurrence_score': score,
            'entropy':            round(h, 6),
            'edge_holder':        best_member,
            'reassigned':         reassigned,
            'association_source': assoc_source,
            'duplicate':          duplicate
        })

    if duplicate:
        duplicate_rows.append({
            'anchor_id':          anchor_id,
            'high_dim_edge':      high_dim_edge,
            'holder_1':           top_two[0],
            'holder_2':           top_two[1] if len(top_two) > 1 else '',
            'entropy':            round(h, 6),
            'association_source': assoc_source
        })

    updated_rows.append({
        'anchor_id':          anchor_id,
        'high_dim_edge':      high_dim_edge,
        'edge_holder':        best_member,
        'synthetic_count':    group['synthetic_count'],
        'members':            group['members'],
        'combo_signatures':   group.get('combo_signatures', '[]'),
        'entropy':            round(h, 6),
        'duplicate_edge':     duplicate,
        'association_source': assoc_source
    })

    status = ("DUPLICATE" if duplicate
              else ("REASSIGNED" if reassigned else "unchanged"))
    print(f"  [{anchor_id}] H={h:.4f} holder={best_member} "
          f"src={assoc_source} [{status}]")

import pandas as pd
to_df = lambda df: cudf.DataFrame.from_pandas(df) if GPU else df

scores_df    = to_df(pd.DataFrame(score_rows))
updated_df   = to_df(pd.DataFrame(updated_rows))
duplicate_df = to_df(pd.DataFrame(duplicate_rows))

# Flag duplicate_edge on nodes
if duplicate_rows:
    dup_holders = set(
        [str(r['holder_1']) for r in duplicate_rows] +
        [str(r['holder_2']) for r in duplicate_rows if r['holder_2']]
    )
    nodes_pd = nodes_pd.copy()
    nodes_pd['duplicate_edge'] = nodes_pd['id'].astype(str).isin(dup_holders)
    nodes_df = to_df(nodes_pd)

return scores_df, updated_df, duplicate_df, nodes_df
```

def main():
parser = argparse.ArgumentParser(description=“Shannon entropy scoring”)
parser.add_argument(’–nodes’,          required=True)
parser.add_argument(’–edges’,          required=True)
parser.add_argument(’–groups’,         required=True)
parser.add_argument(’–threshold’,      type=float, required=True)
parser.add_argument(’–out_scores’,     required=True)
parser.add_argument(’–out_groups’,     required=True)
parser.add_argument(’–out_duplicates’, required=True)
parser.add_argument(’–out_nodes’,      required=True)
args = parser.parse_args()

```
nodes_df  = cudf.read_parquet(args.nodes)
edges_df  = cudf.read_parquet(args.edges)
groups_df = cudf.read_parquet(args.groups)

print(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges, "
      f"{len(groups_df)} groups.")

scores_df, updated_df, duplicate_df, nodes_out = \
    score_and_reassign(nodes_df, edges_df, groups_df, args.threshold)

scores_df.to_parquet(args.out_scores,    index=False)
updated_df.to_parquet(args.out_groups,   index=False)
duplicate_df.to_parquet(args.out_duplicates, index=False)
nodes_out.to_parquet(args.out_nodes,     index=False)

print(f"Entropy scoring complete: {len(updated_df)} groups, "
      f"{len(duplicate_df)} duplicated edges.")
```

if **name** == ‘**main**’:
main()
