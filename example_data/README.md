# Example data

Mock data in the formats the pipeline expects. Identifiers, labels and features are
synthetic. The five patients are enough to run every step of modules 1 and 2 end to end, but
not to produce meaningful metrics.

## Contents

| Path | Format |
| --- | --- |
| `features/<slide_id>.pt` | float tensor `(n_tiles, 768)`, standing in for CHIEF tile features |
| `demographic_classifier/labels/TCGA-BRCA-{Age,Race,Sex}.json` | slide id to one-hot demographic label |
| `pathology_task_prediction/clinical.csv` | one row per patient: `case_submitter_id`, `age`, `race`, `sex`, `primary_diagnosis`, `kfold_partition` |
| `pathology_task_prediction/mutations/{PIK3CA,TP53}.csv` | `Patient ID`, `Altered` (0 wild type, 1 mutated) |
| `pathology_task_prediction/models/TCGA_BRCA_PIK3CA_CHIEF_fold0.pt` | attention-MIL state dict of a PIK3CA model trained on TCGA-BRCA CHIEF features |

Slide ids follow the TCGA barcode layout, so the patient id is the first 12 characters and
the sample type follows it.

## Running the examples

From the repository root:

```bash
cd 01_demographic_classifier && ./run.sh configs/example_train.json && cd ..

python 02_pathology_task_prediction/build_task_manifest.py \
    --cohort TCGA --cancer BRCA --task PIK3CA --features-dir example_data/features \
    --mutation-table example_data/pathology_task_prediction/mutations/PIK3CA.csv \
    --clinical example_data/pathology_task_prediction/clinical.csv --fold-column kfold_partition \
    --output output/example/manifest.csv
python 02_pathology_task_prediction/train.py --manifest output/example/manifest.csv \
    --output-dir output/example/baseline --epochs 2 --num-workers 0
python 02_pathology_task_prediction/predict.py --manifest output/example/manifest.csv \
    --checkpoint example_data/pathology_task_prediction/models/TCGA_BRCA_PIK3CA_CHIEF_fold0.pt \
    --num-workers 0 --output output/example/released_model_predictions.csv
```

`--task subtype_brca --clinical-labels` in place of `--mutation-table` builds the ductal
versus lobular subtype task from `primary_diagnosis`.

## Analysis tables (modules 3 and 4)

`analysis/` holds synthetic patient tables in the schema of `configs/analysis.yaml`: two
cohorts (TCGA, 100 patients; DFCI, 60), one cancer, 200 slides, two tasks. Values are random
with a few planted effects (age on nuclear size, race on nuclear intensity, a race disparity
in the baseline predictions) so that every step returns something.

| Path | Format |
| --- | --- |
| `analysis/clinical.csv` | one row per patient: `cohort`, `cancer`, `patient_id`, `age`, `sex`, `race`, `technical_cov1-4`, `biological_cov1-2` |
| `analysis/predictions.csv`, `analysis/predictions_mitigated.csv` | slide predictions of a baseline and a mitigated model: `cohort`, `cancer`, `task`, `patient_id`, `slide_id`, `probability`, `label` |
| `analysis/morphology.csv` | one row per patient: `all_n_nuclei` and mean nuclear features per cell type (`all_*`, `class_0_*` ... `class_4_*`) |
| `analysis/feature_annotations.csv` | `feature`, `cell_type`, `feature_group`, `morphometric` for the enrichment test |
| `analysis/composition.csv` | one row per slide: `n_tiles` and nucleus counts per cell type |
| `analysis/attention_tiles.csv` | one row per tile and task: `attention` and nucleus counts per cell type |
| `analysis/analysis_example.yaml` | `configs/analysis.yaml` with `min_level_count`, `min_group_size` and `min_residual_df` lowered to fit 160 patients |

The study thresholds in `configs/analysis.yaml` would discard most of this data, so every
command passes `--spec example_data/analysis/analysis_example.yaml`. From the repository root:

```bash
E=example_data/analysis; S="--spec $E/analysis_example.yaml"; O=output/example/analysis; mkdir -p $O
D=04_morphologic_correlates_of_demographic_signal; C="Neoplastic Inflammatory Connective Dead Epithelial"

# module 3: disparities of the baseline, then the resolution rule against the mitigated model
python 03_fairness_evaluation/evaluate_fairness.py $S --predictions $E/predictions.csv \
    --clinical $E/clinical.csv --output $O/fairness.csv --group-output $O/group_performance.csv
python 03_fairness_evaluation/compare_models.py $S --baseline $E/predictions.csv \
    --mitigated $E/predictions_mitigated.csv --clinical $E/clinical.csv --output $O/comparison.csv

# 4a nuclear morphology: associations, cohort pooling, enrichment
python $D/4a_nuclear_morphology/1_test_associations.py $S --features $E/morphology.csv \
    --clinical $E/clinical.csv --output $O/associations.csv
python $D/4a_nuclear_morphology/2_pool_cohorts.py --associations $O/associations.csv --output $O/pooled.csv
python $D/4a_nuclear_morphology/3_test_enrichment.py --results $O/pooled.csv \
    --annotations $E/feature_annotations.csv --output $O/enrichment.csv

# 4b cell composition
python $D/4b_cell_composition/test_composition.py $S --slides $E/composition.csv --clinical $E/clinical.csv \
    --components $C --n-tiles-column n_tiles --output $O/composition_effects.csv --omnibus-output $O/composition_omnibus.csv

# 4c model attention
python $D/4c_model_attention/1_build_attention_composition.py --tiles $E/attention_tiles.csv --components $C \
    --predictions $E/predictions.csv --output $O/attention_composition.csv
python $D/4c_model_attention/2_test_attention_composition.py $S --attention $O/attention_composition.csv \
    --clinical $E/clinical.csv --components $C --output $O/attention_effects.csv --omnibus-output $O/attention_omnibus.csv

# 4d morphology-error decomposition (the bootstrap is the slow step: about 3 minutes at --n-boot 200)
python $D/4d_morphology_error_decomposition/1_build_patient_outcomes.py --predictions $E/predictions.csv --output $O/outcomes.csv
python $D/4d_morphology_error_decomposition/2_decompose_disparity.py $S --outcomes-table $O/outcomes.csv \
    --mediators $E/morphology.csv --clinical $E/clinical.csv --n-boot 200 --output $O/decomposition.csv
python $D/4d_morphology_error_decomposition/3_test_and_pool.py --decomposition $O/decomposition.csv \
    --output $O/decomposition_tested.csv --pooled-output $O/decomposition_pooled.csv
```

Module 5 runs end to end on synthetic data with
`python 05_demographic_signal_agentic_modulation/examples/own_pipeline.py`; `unveil-tilefilter
signal` additionally needs a demographic classifier, see *Model Weights* in the main README.
