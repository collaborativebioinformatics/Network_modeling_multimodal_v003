/*
 * SUMMARIZE
 * Reads all upstream Parquet outputs and writes final TSV deliverables
 * and a human-readable summary report.
 *
 * Inputs:
 *   scored_nodes.parquet      : final node table
 *   expanded_edges.parquet    : full edge list
 *   reassigned_groups.parquet : final group assignments
 *   entropy_scores.parquet    : entropy scores
 *   duplicated_edges.parquet  : duplicated edge assignments
 *   expansion_log.parquet     : expansion round log
 *
 * Outputs:
 *   final_nodes.tsv   : all nodes with kind, feature combo, edge source
 *   final_edges.tsv   : all edges
 *   final_groups.tsv  : anchor groups with entropy and combo signatures
 *   summary_report.txt
 */

process SUMMARIZE {
    tag "summarize"
    label "process_low"

    publishDir "${params.outdir}", mode: 'copy'

    input:
    path scored_nodes
    path expanded_edges
    path reassigned_groups
    path entropy_scores
    path duplicated_edges
    path expansion_log

    output:
    path "final_nodes.tsv",    emit: nodes
    path "final_edges.tsv",    emit: edges
    path "final_groups.tsv",   emit: groups
    path "summary_report.txt", emit: report

    script:
    """
    python3 ${projectDir}/bin/core/summarize.py \
        --nodes      ${scored_nodes}      \
        --edges      ${expanded_edges}    \
        --groups     ${reassigned_groups} \
        --scores     ${entropy_scores}    \
        --duplicates ${duplicated_edges}  \
        --log        ${expansion_log}     \
        --out_nodes   final_nodes.tsv     \
        --out_edges   final_edges.tsv     \
        --out_groups  final_groups.tsv    \
        --out_report  summary_report.txt
    """
}
