#!/usr/bin/env nextflow

/*
================================================================================
  Synthetic Node Expansion Pipeline
================================================================================
  Iteratively copies high-connectivity anchor nodes (e.g. ICD-10 disease codes,
  noted here for documentation only — the algorithm treats them as opaque
  identifiers) into low-dimensional synthetic nodes until no new high-dim
  edges are created.

  Two types of high-dimensional scalar data are supported:

    Type A (genomic)   : binary/count features, ~100k dimensions per parent
                         node. Associations are combinatorial — pairs, triples
                         etc. checked up to max_combo_depth. Exclusivity is
                         the default; duplication allowed when H >= threshold.

    Type B (proteomic) : ~3k dimensions, raw expression values normalised
                         to a per-feature control baseline (mean ± outlier
                         exclusion, variance-weighted z-score). Non-exclusive
                         associations are the default; exclusivity enforced
                         only when H < threshold.

  Both types are kept functionally separate. Either firing is sufficient to
  generate a synthetic node. The association_source column records which
  type drove each synthetic node's creation ('type_a', 'type_b', or 'both').

  Subconditions are an EMERGENT property of the algorithm — they are NOT
  provided as input. The condition/control label is the only pre-defined
  grouping. Synthetic nodes define subconditions through their pattern of
  high-dim edge associations across many features (not just one).

  The proteomics module is OPTIONAL. The pipeline runs without it if
  --proteomics_features is not provided.

  Usage (Type A only):
    nextflow run main.nf \
      --samples   samples.csv  \
      --features  features.csv \
      --edges     edges.csv

  Usage (Type A + Type B):
    nextflow run main.nf \
      --samples              samples.csv      \
      --features             features.csv     \
      --edges                edges.csv        \
      --proteomics_features  proteomics.csv
================================================================================
*/

nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.samples             = null
params.features            = null
params.edges               = null
params.proteomics_features = null    // optional — activates Type B module
params.similarity          = 0.8
params.entropy_threshold   = 0.95
params.low_dim             = 8
params.max_rounds          = 50
params.max_combo_depth     = 3       // max feature combination depth (Type A)
params.n_sd_outlier        = 2.0     // SD threshold for outlier exclusion
params.outdir              = "results"
params.help                = false

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------
if (params.help) {
    log.info """
    ╔══════════════════════════════════════════════════════════╗
    ║         Synthetic Node Expansion Pipeline                ║
    ╚══════════════════════════════════════════════════════════╝

    Required:
      --samples              Sample metadata CSV/TSV
                             Columns: sample_id, condition_label
      --features             Type A feature matrix CSV/TSV
                             Columns: sample_id, v_0..v_n
      --edges                Anchor-anchor edge list CSV/TSV
                             Columns: source, target

    Optional (Type B / proteomics):
      --proteomics_features  Raw proteomic expression matrix CSV/TSV
                             Columns: sample_id, v_0..v_n (~3k features)
                             Raw values — baseline subtraction and z-scoring
                             are performed automatically using control samples
                             identified via condition_label in --samples.

    Algorithm:
      --similarity           Cosine similarity threshold   [${params.similarity}]
      --entropy_threshold    Shannon entropy cutoff        [${params.entropy_threshold}]
      --low_dim              Synthetic embedding dims      [${params.low_dim}]
      --max_rounds           Max expansion rounds          [${params.max_rounds}]
      --max_combo_depth      Max feature combo depth       [${params.max_combo_depth}]
      --n_sd_outlier         Outlier exclusion SD cutoff   [${params.n_sd_outlier}]

    Output:
      --outdir               Output directory              [${params.outdir}]

    Profiles:
      -profile local         CPU (default)
      -profile slurm         HPC + GPU
      -profile aws           AWS Batch
      -profile gcp           Google Cloud
    """.stripIndent()
    exit 0
}

if (!params.samples)  error "ERROR: --samples is required."
if (!params.features) error "ERROR: --features is required."
if (!params.edges)    error "ERROR: --edges is required."

// ---------------------------------------------------------------------------
// Module imports
// ---------------------------------------------------------------------------
include { VALIDATE_INPUT  } from './modules/core/validate_input'
include { BUILD_GRAPH     } from './modules/core/build_graph'
include { EXPAND_NODES    } from './modules/core/expand_nodes'
include { SCORE_ENTROPY   } from './modules/core/score_entropy'
include { SUMMARIZE       } from './modules/core/summarize'

include { VALIDATE_TYPE_B  } from './modules/proteomics/validate_type_b'
include { PREPROCESS_TYPE_B } from './modules/proteomics/preprocess'

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------
workflow {

    log.info """
    ╔══════════════════════════════════════════════════════════╗
    ║         Synthetic Node Expansion Pipeline                ║
    ╚══════════════════════════════════════════════════════════╝
    samples             : ${params.samples}
    features (Type A)   : ${params.features}
    edges               : ${params.edges}
    proteomics (Type B) : ${params.proteomics_features ?: 'not provided'}
    similarity          : ${params.similarity}
    entropy_threshold   : ${params.entropy_threshold}
    max_combo_depth     : ${params.max_combo_depth}
    n_sd_outlier        : ${params.n_sd_outlier}
    low_dim             : ${params.low_dim}
    max_rounds          : ${params.max_rounds}
    outdir              : ${params.outdir}
    profile             : ${workflow.profile}
    """.stripIndent()

    // -------------------------------------------------------------------
    // Core input channels
    // -------------------------------------------------------------------
    samples_ch  = Channel.fromPath(params.samples,  checkIfExists: true)
    features_ch = Channel.fromPath(params.features, checkIfExists: true)
    edges_ch    = Channel.fromPath(params.edges,    checkIfExists: true)

    // -------------------------------------------------------------------
    // Core pipeline
    // -------------------------------------------------------------------
    VALIDATE_INPUT(samples_ch, features_ch)

    BUILD_GRAPH(
        VALIDATE_INPUT.out.samples,
        VALIDATE_INPUT.out.features,
        edges_ch
    )

    // -------------------------------------------------------------------
    // Type B (proteomics) — optional
    // Produces type_b_scores.parquet consumed by EXPAND_NODES.
    // When absent, a sentinel value is passed so EXPAND_NODES skips
    // the Type B check cleanly.
    // -------------------------------------------------------------------
    if (params.proteomics_features) {

        proto_ch = Channel.fromPath(
            params.proteomics_features, checkIfExists: true
        )

        VALIDATE_TYPE_B(
            proto_ch,
            VALIDATE_INPUT.out.samples
        )

        PREPROCESS_TYPE_B(
            VALIDATE_TYPE_B.out.features,
            VALIDATE_INPUT.out.samples
        )

        type_b_ch = PREPROCESS_TYPE_B.out.scores

    } else {
        // Sentinel channel — expand_nodes.py detects 'NO_FILE' and skips
        type_b_ch = Channel.value(file('NO_FILE'))
    }

    // -------------------------------------------------------------------
    // Expansion + scoring
    // -------------------------------------------------------------------
    EXPAND_NODES(
        BUILD_GRAPH.out.graph,
        BUILD_GRAPH.out.edges,
        type_b_ch
    )

    SCORE_ENTROPY(
        EXPAND_NODES.out.nodes,
        EXPAND_NODES.out.edges,
        EXPAND_NODES.out.groups
    )

    SUMMARIZE(
        SCORE_ENTROPY.out.nodes,
        EXPAND_NODES.out.edges,
        SCORE_ENTROPY.out.groups,
        SCORE_ENTROPY.out.scores,
        SCORE_ENTROPY.out.duplicates,
        EXPAND_NODES.out.log
    )
}
