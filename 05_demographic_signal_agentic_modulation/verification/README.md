# Verification against the paper's own files

The four scripts here check that the package does what the paper's code did, on the
paper's data. They were run on 2026-10-10 on the cluster where those files live. The
commands are below. Each script takes its paths as
arguments; nothing in the package refers to these locations.

The locations used:
- `REPRO`: the reproduction package (`UNVEIL_repro_package_v9_20261010`), which holds the
  mask files, the run folders, `MANIFEST.csv` and the paper's analysis code;
- `SIGNALS`: the paper's stored demographic signals;
- `DUMP`: the collaborators' demographic classifiers;
- `NCKU`: the tile features;
- `UPSTREAM`: the collaborators' `fairtune_pipeline`, with its model code in
  `$UPSTREAM/../demographic_classifier`;
- `FT`: the Python of the environment the signals were computed and the models trained in.

## 1. Masks: `eq_masks.py`

```bash
python verification/eq_masks.py --mask-root $REPRO/masks --signal-store $SIGNALS
```

The reproduction package ships 107 mask files:
- 12 for the primary set and 26 for the secondary and tertiary sets, built with the
  paper's filter;
- 34 `attention_matched` and 35 `random_matched`, the controls.

Each file is rebuilt with this package's defaults from the same signal folder and compared
slide by slide. Every signal file is also checked on loading, as `load_signal` does.

## 2. Signals: `eq_signals.py`

```bash
$FT verification/eq_signals.py --features-dir "$NCKU/TCGA-BRCA-FS/VIRCHOW2/20X/pt_files(stain_norm)" \
    --checkpoint $DUMP/age/experiments/04_23/cv_fold_5/models/target_1_best_BalancedAccuracy.pt \
    --model-code $UPSTREAM/../demographic_classifier \
    --signal-dir $SIGNALS/Age/BRCA/VIRCHOW2/FS --n-slides 10
# likewise GBM/Age/UNI/FS (age/experiments/08_23/cv_fold_10/.../target_1_best_AUC.pt)
# and LGG/Race/GIGAPATH/FS (race/experiments/09_20/cv_fold_4/.../target_1_best_BCE.pt)
```

- These are the classifiers `classifier_selection.csv` names for those cells.
- There are 10 slides per cell, the largest one always included.
- Each slide is compared exactly: float16 attention and tile scores, float32 slide
  prediction.
- The script was run in `FT`, with one thread.

## 3. Resolution: `eq_resolve.py`

```bash
python verification/eq_resolve.py --runs-root $REPRO/runs \
    --old-analysis $REPRO/code/primary/analysis --manifest $REPRO/MANIFEST.csv
```

- Every run folder with a bootstrap is read by `resolve.parse_run` and by the paper's
  `collect_confirmation.parse_result_dir`.
- The two must agree on whether the run is scored, on all 14 values, and on the fairness
  half of the rule.
- With `--manifest`, each folder that both refuse is read again at its source folder
  (`MANIFEST.csv` `source_dir`) with `--ignore-error-keys`.
  - That rule compares file times. The source folders in the collaborators' tree keep the
    original times.
  - Copies on a file system that resets the times of old files would pass it trivially.

## 4. The adapter on the real pipeline: `eq_fairtune.py`

```bash
$FT verification/eq_fairtune.py --upstream $UPSTREAM \
    --reference-config $REPRO/code/s16/trees/V10_abc418b8/agentic/overlay/configs/train_configs.yaml
```

- **What runs.** In a temporary directory, which it removes afterwards, the script builds
  the real pipeline's `CancerDataset` for the training, validation, testing and "all"
  partitions, with the injection installed. It also builds `ReweightDataset` for the
  training partition. No training is run.
- **What it compares.** The reference config is the `train_configs.yaml` the paper's
  runs read.

## Tests

`python tests/run_all.py` runs seven test files that need no data.
- `test_filters.py` compares the filters with `reference/filters_locked_c844fcf6.py` on 96
  synthetic slides (576 masks).
- It also checks non-default parameters against the formula in the README.
