/*
 * BUILD_GRAPH
 * Constructs the initial anchor graph from validated inputs.
 * Anchor nodes represent parent nodes (e.g. ICD-10 codes).
 * Each anchor carries a high-dim feature combination signature
 * derived from the samples associated with it.
 *
 * Inputs:
 *   samples.parquet  : validated sample metadata
 *   features.parquet : validated feature matrix
 *   edges_file       : CSV/TSV of anchor-anchor edges (source, target)
 *
 * Outputs:
 *   anchor_graph.parquet : anchor nodes with feature signatures + degrees
 *   degree_table.parquet : sorted degree table
 *   edges.parquet        : validated edge list
 */

process BUILD_GRAPH {
    tag "build_graph"
    label "process_medium"

    publishDir "${params.outdir}/graph", mode: 'copy'

    input:
    path samples
    path features
    path edges_file

    output:
    path "anchor_graph.parquet",  emit: graph
    path "degree_table.parquet",  emit: degrees
    path "edges.parquet",         emit: edges

    script:
    """
    python3 ${projectDir}/bin/core/build_graph.py \
        --samples     ${samples}      \
        --features    ${features}     \
        --edges       ${edges_file}   \
        --out_graph   anchor_graph.parquet \
        --out_degrees degree_table.parquet \
        --out_edges   edges.parquet
    """
}
