/*
 * EXPAND_NODES
 * Core iterative expansion loop.
 * Copies high-connectivity anchor nodes into synthetic low-dim nodes
 * using combinatorial feature similarity until no new high-dim edges form.
 *
 * The similarity check is combinatorial: single features first, then
 * pairs, triples, etc. Higher dimensionality increases the likelihood
 * of combination-driven edges.
 *
 * An optional type_b_scores channel can be provided by the proteomics
 * module. When present, a synthetic node is created if EITHER the
 * Type A OR Type B similarity check fires.
 *
 * Inputs:
 *   anchor_graph.parquet : anchor nodes with feature signatures
 *   edges.parquet        : anchor edge list
 *   type_b_scores        : (optional) proteomics similarity scores parquet
 *
 * Outputs:
 *   expanded_nodes.parquet : all nodes (anchors + synthetics) with metadata
 *   expanded_edges.parquet : full edge list after expansion
 *   groups.parquet         : anchor groups with member lists
 *   expansion_log.parquet  : per-round per-attempt expansion log
 */

process EXPAND_NODES {
    tag "expand_nodes"
    label "process_high"

    publishDir "${params.outdir}/expansion", mode: 'copy'

    input:
    path anchor_graph
    path edges
    path type_b_scores   // pass 'NO_FILE' sentinel when proteomics absent

    output:
    path "expanded_nodes.parquet",  emit: nodes
    path "expanded_edges.parquet",  emit: edges
    path "groups.parquet",          emit: groups
    path "expansion_log.parquet",   emit: log

    script:
    def type_b_arg = type_b_scores.name != 'NO_FILE' \
        ? "--type_b_scores ${type_b_scores}" : ""
    """
    python3 ${projectDir}/bin/core/expand_nodes.py \
        --graph           ${anchor_graph}           \
        --edges           ${edges}                  \
        --similarity      ${params.similarity}      \
        --low_dim         ${params.low_dim}         \
        --max_rounds      ${params.max_rounds}      \
        --max_combo_depth ${params.max_combo_depth} \
        --out_nodes       expanded_nodes.parquet    \
        --out_edges       expanded_edges.parquet    \
        --out_groups      groups.parquet            \
        --out_log         expansion_log.parquet     \
        ${type_b_arg}
    """
}
