/*
 * SCORE_ENTROPY
 * Shannon entropy scoring and edge assignment.
 *
 * Key difference from original: duplication is the DEFAULT for Type B
 * data. Exclusivity is enforced only when entropy is LOW (the edge
 * strongly belongs to one group). The entropy_threshold parameter
 * here is a LOWER bound for exclusivity, not an upper bound for
 * duplication.
 *
 * Tracks which data type (A, B, or both) drove each association.
 *
 * Inputs:
 *   expanded_nodes.parquet : all nodes with metadata
 *   expanded_edges.parquet : full edge list
 *   groups.parquet         : anchor groups
 *
 * Outputs:
 *   entropy_scores.parquet    : per-node entropy scores + association source
 *   reassigned_groups.parquet : groups after edge assignment
 *   duplicated_edges.parquet  : edges assigned to multiple nodes
 *   scored_nodes.parquet      : nodes with duplicate_edge flags updated
 */

process SCORE_ENTROPY {
    tag "score_entropy"
    label "process_medium"

    publishDir "${params.outdir}/entropy", mode: 'copy'

    input:
    path expanded_nodes
    path expanded_edges
    path groups

    output:
    path "entropy_scores.parquet",    emit: scores
    path "reassigned_groups.parquet", emit: groups
    path "duplicated_edges.parquet",  emit: duplicates
    path "scored_nodes.parquet",      emit: nodes

    script:
    """
    python3 ${projectDir}/bin/core/score_entropy.py \
        --nodes          ${expanded_nodes}           \
        --edges          ${expanded_edges}           \
        --groups         ${groups}                   \
        --threshold      ${params.entropy_threshold} \
        --out_scores     entropy_scores.parquet      \
        --out_groups     reassigned_groups.parquet   \
        --out_duplicates duplicated_edges.parquet    \
        --out_nodes      scored_nodes.parquet
    """
}
