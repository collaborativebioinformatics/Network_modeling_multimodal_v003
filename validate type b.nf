/*
 * VALIDATE_TYPE_B
 * Validates the raw proteomic expression matrix.
 * Confirms sample IDs are present in the core sample table.
 * Outputs clean Parquet for preprocessing.
 *
 * Input format (CSV/TSV):
 *   sample_id : must match IDs in core samples table
 *   v_0..v_n  : raw expression values (~3k features)
 *               Values represent absolute expression levels;
 *               baseline subtraction happens in PREPROCESS_TYPE_B.
 *
 * Note: ~3k feature dimensions is typical for this data type but
 * the module is agnostic to the actual number of feature columns.
 */

process VALIDATE_TYPE_B {
    tag "validate_type_b"
    label "process_low"

    publishDir "${params.outdir}/proteomics/validated", mode: 'copy'

    input:
    path proteomics_features
    path core_samples          // validated core samples.parquet

    output:
    path "proteomics_features.parquet", emit: features
    path "proteomics_report.txt",       emit: report

    script:
    """
    python3 ${projectDir}/bin/proteomics/validate_type_b.py \
        --features     ${proteomics_features}        \
        --core_samples ${core_samples}               \
        --out_features proteomics_features.parquet   \
        --report       proteomics_report.txt
    """
}
