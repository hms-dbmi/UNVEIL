# UNVEIL: Uncovering Non-apparent Visual Encodings in Latent Space

Kao TW et al. _Demographic Differences in Tissue Architecture Drive Disparities in Pathology AI Performance._

This repository implements UNVEIL, a framework for identifying, quantifying, attributing and
mitigating demographic-associated signals in pathology foundation model representations. By
integrating demographic classification, fairness evaluation, morphologic attribution and
demographic signal-aware agentic modulation, UNVEIL addresses performance disparities in
computational pathology tasks.

## Overview

Pathology foundation models encode demographic-linked morphological variations during
training, which can contribute to performance disparities across population groups. UNVEIL
provides a systematic approach to:

- **Detect** demographic signals in learned representations through demographic attribute
  classification
- **Quantify** performance disparities of downstream pathology tasks across demographic
  groups, with six fairness metrics and bootstrap bias tests
- **Attribute** demographic signal and disparities to nuclear morphology, cell composition
  and model attention
- **Mitigate** disparities with a demographic signal-aware tile filter, discovered by an
  agentic program search, that removes the tiles carrying the most demographic signal from
  training slides

## Framework Workflow

```
WSI Tiles → Foundation Model Encoding → Attention-Based MIL Aggregation
                                                      ↓
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ↓                                                             ↓
           1. Demographic Classifier                                  2. Pathology Task Model
            (age, race, sex prediction)                      (mutation, subtype, grade, stage, ...)
                       │                                                             ↓
                       │                                               3. Fairness Evaluation
                       │                                                             ↓
                       ├──────────────→ 4. Morphologic Correlates of Demographic Signal
                       │                 (nuclear morphology, cell composition, attention,
                       │                  morphology-error decomposition)
                       ↓
           5. Demographic Signal-Aware Agentic Modulation
              (training tiles with the strongest demographic signal are removed;
               the task model is retrained and re-evaluated with module 3)
```

The framework consists of five modules:

1. **Demographic signal detection.** Train classifiers to predict demographic attributes (age,
   race, sex) from slide representations, quantifying the extent of demographic information
   encoded in foundation model features.

2. **Pathology task prediction.** Train attention-based MIL models for mutation, subtype,
   grade, stage and tumor detection tasks with patient-level cross-validation.

3. **Fairness evaluation.** Measure performance disparities across demographic groups with
   six fairness metrics, bootstrap bias tests and confidence intervals, and compare baseline
   and mitigated models on the same patients.

4. **Morphologic correlates of demographic signal.** Link demographic groups and prediction
   disparities to nuclear morphometric features, cell-type composition and the attention of
   task models, with covariate-adjusted models and false discovery rate control.

5. **Demographic signal-aware agentic modulation.** For each training slide, score every tile
   by how strongly it carries evidence for the slide's demographic group and how much the
   demographic classifier attends to it, and remove the highest-scoring tiles before the task
   model is trained. The scoring rule was found by an LLM-driven evolutionary search and then
   locked. Validation, test and external slides are never filtered.

## Repository Structure

```
.
├── 01_demographic_classifier/                     # Demographic attribute prediction
│   ├── configs/                                   # Configuration examples
│   ├── run.py                                     # Main entry point
│   ├── run.sh                                     # Execution wrapper
│   ├── experiment_runner.py                       # Training orchestration
│   ├── dataset.py                                 # Feature dataset loaders
│   └── model.py                                   # Attention-MIL and MLP classifiers
│
├── 02_pathology_task_prediction/                  # Downstream task models
│   ├── configs/tasks.yaml                         # Clinical label mappings (grade, stage, subtype, ...)
│   ├── build_task_manifest.py                     # Slide manifest with labels and patient-level folds
│   ├── mil.py                                     # Gated attention-MIL model and slide bags
│   ├── train.py                                   # Cross-validated training, optional tile masks
│   ├── predict.py                                 # Scoring of external cohorts or released checkpoints
│   └── run_cross_validation.sh                    # Job-scheduler wrapper
│
├── 03_fairness_evaluation/                        # Disparities across demographic groups
│   ├── evaluate_fairness.py                       # Six fairness metrics, bias tests, intervals
│   └── compare_models.py                          # Paired baseline-versus-mitigated comparison
│
├── 04_morphologic_correlates_of_demographic_signal/
│   ├── 4a_nuclear_morphology/                     # Associations, cohort pooling, enrichment
│   ├── 4b_cell_composition/                       # Compositional mixed models and permutation tests
│   ├── 4c_model_attention/                        # Attention-weighted composition of task models
│   └── 4d_morphology_error_decomposition/         # Mediation of prediction disparities by morphology
│
├── 05_demographic_signal_agentic_modulation/      # unveil_tilefilter package
│   ├── src/unveil_tilefilter/                     # Signal, masks, training adapters, resolution rule
│   ├── reference/                                 # Locked filter file used in the paper
│   ├── examples/                                  # Synthetic end-to-end example
│   ├── tests/                                     # Package tests
│   └── verification/                              # Checks against the paper's files
│
├── unveil/                                        # Shared statistics (design, effects, mixed models,
│                                                  #   permutation, meta-analysis, multiplicity, fairness)
├── configs/analysis.yaml                          # Exposures and covariates for modules 3 and 4
├── tests/                                         # Statistical tests against reference implementations
├── example_data/                                  # Mock data demonstrating required formats
├── requirements.txt                               # Python dependencies
├── pyproject.toml                                 # Installs the unveil package
└── LICENSE                                        # GNU AFFERO GENERAL PUBLIC LICENSE v3.0
```

## Data Requirements

### Input Data Format

- **WSI tile features** (`.pt` files): pre-extracted tile-level features from a foundation
  model, one file per slide named `<slide_id>.pt`, of shape `(n_tiles, feature_dim)`.
  Supported dimensions include CHIEF (768), UNI (1024), GIGAPATH (1536) and VIRCHOW2 (2560).
- **Demographic labels** (module 1): JSON files mapping slide ids to demographic categories.
- **Task labels** (module 2): a 0/1 mutation table, a clinical column mapped through
  `02_pathology_task_prediction/configs/tasks.yaml`, or the TCGA sample type.
- **Clinical table** (modules 3 and 4): one row per patient with the exposures and
  covariates named in `configs/analysis.yaml`.

Every table is keyed by `cohort`, `cancer` and `patient_id`. Slide tables add `slide_id` and
prediction tables add `task`.

### Covariates

Covariates are grouped into three blocks, published here under proxy names:

| Block | Columns | Examples of what they stand for |
| --- | --- | --- |
| technical | `technical_cov1` to `technical_cov4` | tissue source site, scanner, stain batch, slide type |
| biological | `biological_cov1`, `biological_cov2` | histologic subtype, tumor purity |
| demographic | `age`, `sex`, `race` | |

Rename your columns or edit the spec to match. Columns listed under `categorical_covariates`
are encoded as indicators; rare levels are pooled into `other` and missing values become
their own level. Columns listed under `random_intercept` are modelled as random intercepts
where a script supports it.

### Example Data

Mock example data is provided in `example_data/` to demonstrate the required formats:

- 5 mock WSI feature files with CHIEF-sized embeddings (768-dim)
- Clinical metadata and mutation status files for BRCA
- Demographic label files (age, race, sex)
- A trained PIK3CA attention-MIL checkpoint
- Synthetic patient tables for modules 3 and 4 (`example_data/analysis/`), with a spec whose
  thresholds fit 160 patients

See `example_data/README.md` for detailed specifications and the commands that run every
module on it.

## Installation

### Prerequisites

- Python 3.8 (the study's version; `requirements.txt` pins its packages and installs on 3.8 to 3.10)
- CUDA-capable GPU (recommended)
- SLURM cluster environment (optional, for batch jobs)

### Setup

```bash
# Clone repository
git clone https://github.com/hms-dbmi/UNVEIL.git
cd UNVEIL

# Create environment
conda create -n unveil python=3.8
conda activate unveil

# Install dependencies and the shared statistics package
pip install -r requirements.txt
pip install -e .

# Install the tile filter used in module 5
pip install -e 05_demographic_signal_agentic_modulation

# Run the tests
pytest tests
```

## Usage

The commands below run from the repository root.

### I: Train Demographic Classifier

Train classifiers to detect demographic signals in foundation model representations:

```bash
cd 01_demographic_classifier && ./run.sh configs/example_train.json
```

Key configuration parameters:

- `train_targets_file_path_list`: paths to demographic label JSON files
- `features_dir_path_list`: directories containing WSI feature `.pt` files
- `model_init_args.input_dim`: feature dimension matching the foundation model
- `model_init_args.output_dim`: number of demographic categories

Output: trained classifiers and performance metrics for each cross-validation fold in the
specified `save_dir`.

### II: Train Pathology Task Models

Build a slide manifest with labels and patient-level folds, then train attention-based MIL
models:

```bash
python 02_pathology_task_prediction/build_task_manifest.py \
    --cohort TCGA --cancer BRCA --task PIK3CA \
    --features-dir example_data/features \
    --mutation-table example_data/pathology_task_prediction/mutations/PIK3CA.csv \
    --clinical example_data/pathology_task_prediction/clinical.csv --fold-column kfold_partition \
    --output output/manifest.csv

python 02_pathology_task_prediction/train.py --manifest output/manifest.csv --output-dir output/baseline
```

Labels come from a mutation table, a clinical column mapped through `configs/tasks.yaml`
(grade, stage, subtype, receptor status), or the TCGA sample type (tumor detection). Folds
come from a released split, a fixed clinical column, or a patient-grouped stratified split.

For each test fold k, the patients of fold k+1 form the validation set and the remaining
folds form the training set. Training follows the study recipe:

- class-weighted cross-entropy, Adam at 1e-4 with step decay 0.955, batch 16 with two
  accumulation steps
- at most 2000 randomly drawn tiles per training bag
- up to 100 epochs, stopping after 10 epochs without improvement in patient-level
  validation AUC

`predict.py` scores an external cohort or a released checkpoint, and
`run_cross_validation.sh` wraps the manifest, baseline and masked training steps for a job
scheduler.

Output: per-fold models and the out-of-fold `predictions.csv` used by modules 3 and 4.

### III: Assess Fairness

Evaluate performance disparities across demographic groups, and compare a mitigated model
with its baseline on the same patients:

```bash
python 03_fairness_evaluation/evaluate_fairness.py --predictions output/baseline/predictions.csv \
    --clinical clinical.csv --output fairness.csv --group-output group_performance.csv

python 03_fairness_evaluation/compare_models.py --baseline output/baseline/predictions.csv \
    --mitigated output/signal_aware/predictions.csv --clinical clinical.csv --output comparison.csv
```

AUC, TPR, TNR, balanced accuracy and accuracy are computed in every demographic group. Each disparity
is the best group minus the worst group, or with `--privileged-reference` the reference
level minus the worst other group:

| Metric | Disparity in |
| --- | --- |
| `EOpp0` | TNR |
| `EOpp1` | TPR |
| `EBAcc` | balanced accuracy |
| `EOddAbs` | mean of the absolute TPR and TNR disparities |
| `ACCDiff` | accuracy |
| `AUCDiff` | AUC |

- **Groups:** continuous exposures are split at the patient median. Categorical exposures
  compare all levels with at least `min_group_size` patients, and `--pairwise` adds each
  level against the reference.
- **Bias test:** each patient keeps its group while outcome and prediction pairs are redrawn
  into the group slots within outcome classes, without replacement (`--null permutation
  --stratify-by label`, the study default: 2,000 permutations of demographic labels within
  outcome classes) or with replacement (`--null bootstrap`). The one-sided p-value is the
  share of null disparities at least as large as the observed one.
- **Intervals:** a patient bootstrap within groups, 2,000 resamples.
- **Comparison:** each disparity under both models, the paired bootstrap interval of the
  change, and a p-value from swapping the two models' predictions within patients.
- **Multiplicity:** Benjamini-Hochberg q-values per cohort, exposure, contrast and metric,
  across cancers and tasks.

Output: one row per task, exposure, contrast and metric, and optionally per-group
performance with intervals.

### IV: Attribute Demographic Signal to Morphology

Three defaults apply to every analysis in this module:

- Analyses are adjusted for all three covariate blocks. The exposure being tested is always
  removed from its own adjustment set. `--adjust technical biological` restricts the blocks
  and `--adjust none` gives the unadjusted analysis.
- Slide-level inputs are averaged to the patient before testing, except where a patient
  random intercept models the slides directly (4b).
- Benjamini-Hochberg q-values are computed within broad families, such as all features of
  one exposure contrast in one cohort. The family is printed in every output and can be
  changed with `--fdr-family`.

Categorical exposures are tested as pairwise contrasts against the reference level, for
example `male_vs_female` or `black_vs_white`. Continuous exposures enter linearly; in 4d they
are contrasted between their 10th and 90th percentiles.

**4a. Nuclear morphology**

```bash
python 04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology/1_test_associations.py \
    --features morphology.csv --clinical clinical.csv --output associations.csv
python 04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology/2_pool_cohorts.py \
    --associations associations.csv --output pooled.csv
python 04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology/3_test_enrichment.py \
    --results pooled.csv --annotations feature_annotations.csv --output enrichment.csv
```

The adjusted effect is a partial correlation on normal scores; `--unadjusted` adds Cliff's
delta or Spearman's rho. Cohorts are pooled on the Fisher z scale with DerSimonian-Laird
random effects, with Hartung-Knapp intervals, prediction intervals and leave-one-cohort-out
robustness. Enrichment of significant features by cell type and feature group is tested by
permuting the category labels.

**4b. Cell composition**

```bash
python 04_morphologic_correlates_of_demographic_signal/4b_cell_composition/test_composition.py \
    --slides composition.csv --clinical clinical.csv \
    --components tumor lymphocyte stroma necrosis --n-tiles-column n_tiles --output composition_effects.csv
```

Counts are closed and centred log-ratio transformed with multiplicative zero replacement.
Each component is modelled at slide level with a patient random intercept and checked at
patient level with a Freedman-Lane permutation test. `--omnibus-output` adds a multivariate
PERMANOVA and a dispersion test.

**4c. Model attention**

```bash
python 04_morphologic_correlates_of_demographic_signal/4c_model_attention/1_build_attention_composition.py \
    --tiles attention_tiles.csv --components tumor lymphocyte stroma necrosis \
    --predictions output/baseline/predictions.csv --output attention_composition.csv
python 04_morphologic_correlates_of_demographic_signal/4c_model_attention/2_test_attention_composition.py \
    --attention attention_composition.csv --clinical clinical.csv \
    --components tumor lymphocyte stroma necrosis --output attention_effects.csv --omnibus-output attention_omnibus.csv
```

The attention-weighted composition of task models is tested with the 4b model, per task by
default. Passing `--predictions` adds `prediction_correct`, so attention can also be compared
between correct and incorrect predictions with `--exposures prediction_correct`.

**4d. Morphology-error decomposition**

```bash
python 04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition/1_build_patient_outcomes.py \
    --predictions output/baseline/predictions.csv --output outcomes.csv
python 04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition/2_decompose_disparity.py \
    --outcomes-table outcomes.csv --mediators morphology.csv --clinical clinical.csv --output decomposition.csv
python 04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition/3_test_and_pool.py \
    --decomposition decomposition.csv --output decomposition_tested.csv --pooled-output decomposition_pooled.csv
```

For each morphology feature, the exposure effect on the patient score and on the prediction
error is split by interventional g-computation into an indirect effect through the feature
and a direct effect, with the controlled direct effect, interaction components and
label-dependent indirect effect also reported. Inference uses a patient bootstrap, and
indirect effects are tested against the composite null with the HDMT procedure before
Benjamini-Hochberg correction. `--chunk` and `--n-chunks` split the features across a job
array.

### V: Apply Demographic Signal-Aware Agentic Modulation

Mitigate performance disparities by removing the tiles with the strongest demographic signal
from training slides:

```bash
unveil-tilefilter signal --features-dir example_data/features --checkpoint demographic_classifier.pt \
    --model-config demographic_classifier.json --model-code 01_demographic_classifier --out signals/
unveil-tilefilter masks --signals signals/ --out masks/adaptive.npz
python 02_pathology_task_prediction/train.py --manifest output/manifest.csv \
    --output-dir output/signal_aware --tile-masks masks/adaptive.npz
```

The demographic classifier from module 1 provides, for every tile, its attention and its
probability of the slide's demographic group. The released classifiers are plain
state_dicts with a JSON config naming the model class and its arguments (`--model-config`);
a checkpoint pickled by module 1 loads without it. Each tile is scored by this evidence weighted
by its attention rank, and the 33% highest-scoring tiles of each training slide are removed,
always keeping at least 64 tiles. Validation, test and external slides are never filtered,
so the model is evaluated on whole slides. The scoring rule was discovered by an LLM-driven
evolutionary search (OpenEvolve) on TCGA development tasks and then locked.

- `--filter random_matched` and `--filter attention_matched` build the two controls, which
  remove the same number of tiles from each slide at random or by attention alone.
- A biased task is resolved when the mitigated model no longer shows a significant EOddAbs
  disparity and its AUROC is not significantly lower than the baseline's by a paired
  permutation test: the `resolved` column of `compare_models.py` (module 3).
  `unveil-tilefilter resolve` scores `fairtune_pipeline` run folders with the same fairness
  criterion and an AUROC margin.

See `05_demographic_signal_agentic_modulation/README.md` for the exact filter, the
`fairtune_pipeline` adapter and the verification against the paper's files.

## Output Structure

```
output/
├── demographic_classifier_<attribute>/          # Module 1 (save_dir)
│   ├── configs.json                             # Configuration
│   ├── summary_metrics.csv                      # Metrics across folds
│   └── cv_fold_<k>/
│       ├── data_split.json                      # Train, validation and test split
│       ├── models/                              # Best checkpoints per selection metric
│       ├── train_log.json
│       └── metrics.csv
│
├── manifest.csv                                 # Module 2 slide manifest
├── baseline/                                    # Module 2 baseline models
│   ├── config.json
│   ├── predictions.csv                          # Out-of-fold slide probabilities
│   └── fold_<k>/
│       ├── model.pt                             # Best model weights
│       ├── history.csv                          # Training curve
│       └── summary.json                         # Fold performance
│
├── signals/<slide_id>.npz                       # Module 5 per-tile demographic signal
├── masks/adaptive.npz (+ .json)                 # Module 5 training masks and their record
└── signal_aware/                                # Module 5 models, same layout as baseline/
```

## Model Weights

The trained models are released on Hugging Face: https://huggingface.co/yulab/UNVEIL (weights under CC BY-NC
4.0; `MODEL_CARD.md` in this repository is the same model card).

| Release | Files | Used by |
| --- | --- | --- |
| Demographic classifiers | one gated attention-MIL classifier per cancer, attribute and slide type, as a state_dict with a JSON config | `unveil-tilefilter signal` (module 5) |
| UNVEIL task models | the attention-MIL task models trained with the module 5 tile filter, 136 tasks with four fold models each | `02_pathology_task_prediction/predict.py`; module 3 |

Module 2 trains unmitigated models from the recipe above.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('yulab/UNVEIL', local_dir='weights')"
python 02_pathology_task_prediction/predict.py --manifest output/manifest.csv \
    --checkpoint weights/task_models/<task>/fold0/model.pt --output predictions.csv
```

## License

GNU Affero General Public License v3.0. See `LICENSE`.
