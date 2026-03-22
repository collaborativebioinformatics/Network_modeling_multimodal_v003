# Network_modeling_multimodal_v003

# Synthetic Node Expansion Pipeline

A Nextflow pipeline for iterative graph expansion using multimodal
high-dimensional data. Starting from anchor nodes, the pipeline
generates synthetic low-dimensional copies until no new high-dimensional
edges are created, using Shannon entropy to guide edge assignment and
duplication decisions.

-----

## Background and Motivation

The pipeline was designed to work with biomedical graph data where
anchor nodes represent disease categories (e.g. ICD-10 codes — noted
here for context only; the algorithm treats them as opaque identifiers).
Each anchor is associated with high-dimensional feature data from one
or two sources:

- **Type A (genomic):** Binary or count features with ~100,000 possible
  dimensions per parent node. Associations are combinatorial — the
  algorithm tests single features, then pairs, triples, etc. up to a
  configurable depth.
- **Type B (proteomic):** Scalar expression values across ~3,000 features,
  expressed relative to a per-feature control baseline. Non-exclusive
  associations are biologically more common here, which is reflected in
  an inverted entropy logic for edge assignment.

Subconditions are an **emergent property** of the algorithm — they are
not provided as input. The only pre-defined grouping is condition vs.
control. Synthetic nodes define subconditions through their pattern of
high-dimensional edge associations across many features.

-----

## Algorithm

### Core Concept

Each anchor node has exactly one connection to the high-dimensional
data at any time (the “high-dim edge”). This edge is mobile — it can
migrate between the anchor and its synthetic copies based on
co-occurrence affinity. When a node’s co-occurrence distribution is
sufficiently split between two groups (measured by Shannon entropy),
the edge can be duplicated rather than forced to choose.

### Expansion Loop

1. Rank anchor nodes by degree (descending) each round
1. For each anchor, generate a synthetic low-dimensional copy
1. Check the copy against all other anchors using combinatorial
   cosine similarity (Type A) and/or z-score cosine similarity (Type B)
1. If either check fires, add the synthetic node to the graph
1. Record which features and combination depth drove the association
1. Repeat until a full round produces no new synthetic nodes

### Combinatorial Similarity (Type A)

Rather than checking all features at once, the algorithm selects the
top-k features by absolute value in the candidate vector and computes
cosine similarity on that subset. It tests depths 1 through
`max_combo_depth` in order, firing on the shallowest combination that
exceeds the threshold. This reflects the observation that higher
dimensionality increases the likelihood of combination-driven rather
than single-feature associations.

### Z-Score Baseline (Type B)

Control samples (identified by `condition_label`) define a per-feature
baseline. The baseline is computed as:

1. Initial mean and SD across control samples
1. Outlier exclusion: control samples beyond `n_sd_outlier` SD per
   feature are removed
1. Recomputed mean and SD on the remaining control samples

Condition samples are then z-scored against this baseline:
`z = (x - control_mean) / control_sd`

The anchor signature is the mean z-score vector across condition
samples (with a second round of outlier exclusion at the sample level).
Cosine similarity between this signature and individual condition
sample z-score vectors gives the Type B similarity score.

Note: a sample that is a uniform outlier in magnitude but consistent
in direction is intentionally retained — it represents a genuine
(if extreme) expression of the same subcondition profile.

### Edge Assignment and Duplication (Shannon Entropy)

After expansion, each anchor group’s high-dim edge is assigned to
the member with the highest co-occurrence affinity. The entropy of the
co-occurrence distribution determines whether the edge is duplicated:

|Data type    |Default  |Condition for duplication       |
|-------------|---------|--------------------------------|
|Type A only  |Exclusive|H ≥ entropy_threshold           |
|Type B / both|Duplicate|Always (exclusive only if H < θ)|

This reflects the biological reality that non-exclusive associations
are more common in proteomic data than genomic data.

-----

## Pipeline Structure

```
main.nf
├── modules/
│   ├── core/
│   │   ├── validate_input.nf     Step 1: validate samples + features
│   │   ├── build_graph.nf        Step 2: build anchor graph
│   │   ├── expand_nodes.nf       Step 3: iterative expansion loop
│   │   ├── score_entropy.nf      Step 4: entropy scoring + edge assignment
│   │   └── summarize.nf          Step 5: final outputs + report
│   └── proteomics/               (optional — activated by --proteomics_features)
│       ├── validate_type_b.nf    Validate raw expression matrix
│       └── preprocess.nf         Baseline subtraction + z-scoring
└── bin/
    ├── core/                     Python scripts for core modules
    └── proteomics/               Python scripts for proteomics modules
```

The proteomics module is **fully optional**. When `--proteomics_features`
is not provided, a sentinel value is passed to `EXPAND_NODES` which
skips the Type B check entirely. All other pipeline behaviour is
unchanged.

-----

## GPU Acceleration

All Python scripts use **RAPIDS** (cuDF, cuPy, cuGraph) when available,
with automatic fallback to pandas, NumPy, and NetworkX on CPU-only
environments. The fallback is transparent — the same algorithm runs on
either backend. The inter-process data format is **Parquet** throughout,
which is natively GPU-resident in cuDF.

GPU backend detection:

```python
try:
    import cudf, cupy, cugraph
    GPU = True
except ImportError:
    import pandas as cudf, numpy as cupy, networkx
    GPU = False
```

-----

## Input Files

### `--samples` (required)

Sample metadata. One row per sample.

|Column           |Description                            |
|-----------------|---------------------------------------|
|`sample_id`      |Unique sample identifier               |
|`condition_label`|Exactly 2 values: condition and control|

### `--features` (required)

Type A feature matrix. One row per sample.

|Column     |Description                                         |
|-----------|----------------------------------------------------|
|`sample_id`|Must match `--samples`                              |
|`v_0..v_n` |Feature values (binary/count, ~100k dims per anchor)|

### `--edges` (required)

Anchor-to-anchor edge list.

|Column  |Description   |
|--------|--------------|
|`source`|Anchor node ID|
|`target`|Anchor node ID|

### `--proteomics_features` (optional)

Raw proteomic expression matrix. One row per sample.

|Column     |Description                                          |
|-----------|-----------------------------------------------------|
|`sample_id`|Must match `--samples`                               |
|`v_0..v_n` |Raw expression values (~3k dims; baseline-subtracted |
|           |automatically using control samples from `--samples`)|

All files may be CSV (`,`) or TSV (`\t`) — the delimiter is detected
automatically.

-----

## Output Files

```
results/
├── validated/
│   ├── samples.parquet           Validated sample metadata
│   ├── features.parquet          Validated Type A feature matrix
│   └── validation_report.txt
├── graph/
│   ├── anchor_graph.parquet      Anchor nodes with feature signatures
│   ├── degree_table.parquet      Node degrees (descending)
│   └── edges.parquet             Validated edge list
├── expansion/
│   ├── expanded_nodes.parquet    All nodes (anchors + synthetics)
│   ├── expanded_edges.parquet    Full edge list after expansion
│   ├── groups.parquet            Anchor groups with member lists
│   └── expansion_log.parquet     Per-round expansion log
├── entropy/
│   ├── entropy_scores.parquet    Per-node entropy + co-occurrence scores
│   ├── reassigned_groups.parquet Groups after edge reassignment
│   ├── duplicated_edges.parquet  Edges assigned to multiple nodes
│   └── scored_nodes.parquet      Nodes with duplicate_edge flags
├── proteomics/                   (only when --proteomics_features provided)
│   ├── validated/
│   │   ├── proteomics_features.parquet
│   │   └── proteomics_report.txt
│   ├── type_b_scores.parquet     Per-anchor Type B similarity scores
│   ├── baseline_stats.parquet    Per-feature control mean/SD/outlier counts
│   └── zscores.parquet           Condition sample z-score matrix
├── final_nodes.tsv               All nodes: kind, combo_depth, assoc_source
├── final_edges.tsv               All edges
├── final_groups.tsv              Groups with entropy + combo signatures
├── summary_report.txt
├── pipeline_report.html          Nextflow execution report
├── pipeline_timeline.html
├── pipeline_trace.tsv
└── pipeline_dag.svg
```

### Key output columns

**`final_nodes.tsv`**

|Column              |Description                                  |
|--------------------|---------------------------------------------|
|`id`                |Node identifier                              |
|`kind`              |`anchor` or `synthetic`                      |
|`anchor_id`         |Parent anchor for synthetic nodes            |
|`high_dim_edge`     |Currently held high-dim edge                 |
|`degree`            |Graph degree                                 |
|`duplicate_edge`    |Whether this node holds a duplicated edge    |
|`association_source`|`type_a`, `type_b`, or `both`                |
|`combo_depth`       |Feature combination depth that fired (Type A)|
|`combo_features`    |Feature indices in winning combination       |
|`round`             |Expansion round in which node was created    |

**`final_groups.tsv`**

|Column              |Description                                   |
|--------------------|----------------------------------------------|
|`anchor_id`         |Anchor node identifier                        |
|`high_dim_edge`     |High-dim edge associated with this group      |
|`edge_holder`       |Current edge holder (anchor or synthetic copy)|
|`synthetic_count`   |Number of synthetic nodes in this group       |
|`entropy`           |Shannon entropy of co-occurrence distribution |
|`duplicate_edge`    |Whether the edge was duplicated               |
|`association_source`|Data type that drove expansion                |
|`combo_signatures`  |JSON list of feature combination records      |

-----

## Parameters

|Parameter            |Default  |Description                                |
|---------------------|---------|-------------------------------------------|
|`--similarity`       |`0.8`    |Cosine similarity threshold                |
|`--entropy_threshold`|`0.95`   |Shannon entropy cutoff for edge duplication|
|`--low_dim`          |`8`      |Synthetic node embedding dimensions        |
|`--max_rounds`       |`50`     |Maximum expansion rounds                   |
|`--max_combo_depth`  |`3`      |Maximum feature combination depth (Type A) |
|`--n_sd_outlier`     |`2.0`    |SD threshold for outlier exclusion         |
|`--outdir`           |`results`|Output directory                           |

-----

## Usage

### Type A only (genomic)

```bash
nextflow run main.nf \
  --samples   samples.csv  \
  --features  features.csv \
  --edges     edges.csv    \
  -profile local
```

### Type A + Type B (genomic + proteomic)

```bash
nextflow run main.nf \
  --samples              samples.csv      \
  --features             features.csv     \
  --edges                edges.csv        \
  --proteomics_features  proteomics.csv   \
  -profile local
```

### HPC (SLURM)

```bash
nextflow run main.nf \
  --samples   samples.csv  \
  --features  features.csv \
  --edges     edges.csv    \
  -profile slurm
```

### AWS Batch

```bash
# Update nextflow.config with your queue, region, and S3 bucket first
nextflow run main.nf \
  --samples   s3://your-bucket/samples.csv  \
  --features  s3://your-bucket/features.csv \
  --edges     s3://your-bucket/edges.csv    \
  --outdir    s3://your-bucket/results      \
  -profile aws
```

### Google Cloud

```bash
# Update nextflow.config with your project and GCS bucket first
nextflow run main.nf \
  --samples   gs://your-bucket/samples.csv  \
  --features  gs://your-bucket/features.csv \
  --edges     gs://your-bucket/edges.csv    \
  --outdir    gs://your-bucket/results      \
  -profile gcp
```

-----

## Integration Points

Two functions are marked as integration points for real data:

**`bin/core/score_entropy.py` — `compute_cooccurrence_scores()`**
Currently uses neighbor-set overlap as a proxy for co-occurrence.
Replace with a lookup into your actual edge co-occurrence frequency
table. For Type B data, consider weighting by z-score magnitude rather
than presence/absence.

**`bin/proteomics/preprocess.py` — `compute_anchor_signature()`**
Currently computes a single anchor signature per condition label.
For multi-condition datasets (multiple parent nodes), extend by
grouping on `condition_label` before computing signatures.

-----

## Requirements

- [Nextflow](https://www.nextflow.io/) >= 23.10
- [Docker](https://www.docker.com/) (or Singularity for HPC)
- Python >= 3.12
- CPU: `networkx`, `numpy`, `pandas`, `pyarrow`
- GPU (optional): RAPIDS `cugraph`, `cudf`, `cupy`
  — Base image: `nvcr.io/nvidia/rapidsai/base:24.10-cuda12.5-py3.12`
  
