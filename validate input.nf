/*
 * VALIDATE_INPUT
 * Validates sample metadata and Type A (genomic) feature matrix.
 * Outputs clean Parquet files for downstream processes.
 *
 * Inputs:
 *   samples_file : CSV/TSV with columns sample_id, condition_label
 *   features_file: CSV/TSV samples x features matrix (v_0..v_n)
 *
 * Outputs:
 *   samples.parquet  : validated sample metadata
 *   features.parquet : validated feature matrix (float32, outliers flagged)
 */

process VALIDATE_INPUT {
    tag "validate_input"
    label "process_low"

    publishDir "${params.outdir}/validated", mode: 'copy'

    input:
    path samples_file
    path features_file

    output:
    path "samples.parquet",           emit: samples
    path "features.parquet",          emit: features
    path "validation_report.txt",     emit: report

    script:
    """
    python3 ${projectDir}/bin/core/validate_input.py \
        --samples         ${samples_file}  \
        --features        ${features_file} \
        --out_samples     samples.parquet  \
        --out_features    features.parquet \
        --report          validation_report.txt
    """
}
