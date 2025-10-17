# SANER2026
# Replication Package: Release Commit Data and Feature Analysis

This replication package reproduces all datasets, feature extraction pipelines, and performance analyses used in the study.  
The process is fully automated through a sequence of Python scripts organized into modular stages.

---

## Directory Structure Overview

release_commit_data/
│
├── git_release.py
├── git_release_derby.py
├── git_authors_total_lines.py
├── git_authors_total_derby.py
│
├── pickles_creating_product.py
├── pickles_creating_total.py
│
└── comparison_outputs/
├── 1_process_metrics_product.py
├── 2_process_metrics_vectors_product.py
├── 3_traditional_hyper_centralities.py
├── 4_build_csvs.py
├── 5_build_csvs_vectors.py
│
└── csv_outputs/
├── 1_combine_csv.py
├── 2_combine_csv_vector.py
├── 3_pr+pt_predictions.py
├── 4_pr+pv_predictions.py
└── 5_pr+pv+h_predictions.py

Make sure you have **Python 3.9+** installed.

## Data Preparation

These scripts collect and preprocess release commit data from open-source repositories.

- `git_release.py` – Extracts release commits and metadata.
- `git_release_derby.py` – Specialized extraction for the Apache Derby project.
- `git_authors_total_lines.py` – Aggregates author-level commit statistics (total lines added/removed).
- `git_authors_total_derby.py` – Derby-specific author statistics.

All outputs are stored as `.pkl` files for efficient reuse.

## Pickle Creation

The pickle files serve as serialized datasets for subsequent processing.

- `pickles_creating_product.py` – Generates project-level pickle datasets.
- `pickles_creating_total.py` – Combines all project-level pickles into a unified dataset.

**Output directory:** `release_commit_data/pickles/`

## Feature Extraction and Comparison Pipelines

Located in `comparison_outputs/`, these scripts compute and compare feature sets.

| Script | Description |
|---------|-------------|
| `1_process_metrics_product.py` | Computes standard process metrics per project. |
| `2_process_metrics_vectors_product.py` | Generates vectorized process metrics. |
| `3_traditional_hyper_centralities.py` | Computes both traditional and hypergraph-based centralities. |
| `4_build_csvs.py` | Compiles metrics into structured CSV files. |
| `5_build_csvs_vectors.py` | Creates vectorized CSVs for ML experiments. |

**Output directory:** `comparison_outputs/csv_outputs/`

## CSV Combination and Prediction Analysis

The following scripts merge results and compute predictive performance across experiments.

| Script | Description |
|---------|-------------|
| `1_combine_csv.py` | Merges CSV files across projects. |
| `2_combine_csv_vector.py` | Combines vector-based CSV outputs. |
| `3_pr+pt_predictions.py` | Performs prediction using process + traditional metrics. |
| `4_pr+pv_predictions.py` | Prediction using process + vector features. |
| `5_pr+pv+h_predictions.py` | Prediction using process + vector + hypergraph metrics. |

**All results** are saved under:  
`release_commit_data/comparison_outputs/csv_outputs/`

## Reproducing the Results

To reproduce the full pipeline as in the paper:

```bash
python git_release.py
python pickles_creating_total.py
python comparison_outputs/1_process_metrics_product.py
python comparison_outputs/2_process_metrics_vectors_product.py
python comparison_outputs/3_traditional_hyper_centralities.py
python comparison_outputs/4_build_csvs.py
python comparison_outputs/5_build_csvs_vectors_product.py
python comparison_outputs/csv_outputs/5_pr+pv+h_predictions.py

