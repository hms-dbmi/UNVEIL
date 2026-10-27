# UNVEIL adaptive tile filter: implementation package

This package applies the UNVEIL bias mitigation to your own data: you extract the tile
features and prepare the tasks; the package computes the demographic signal, builds the
masks, filters the training bags, and scores the result. Every path, seed and threshold
is an argument. The only built-in values are defaults, and those are the values used in
the paper.

**What the method does.** For each slide, a demographic classifier (an attention-MIL model
trained to predict the sensitive attribute from the slide) says which tiles carry the most
evidence for the slide's demographic group. When a task model is trained, those tiles are
removed from the **training** slides: by default the 33% of each slide's tiles that score
highest. Validation, test and external-cohort slides are never filtered. The model learns
from slides with less demographic signal; it is evaluated on whole slides.

With default parameters the filter gives exactly the masks the paper's runs trained on.
This is checked on all 107 mask files of the reproduction package (see *Verification*).

Version 1.0.0 (2026-10-10).

---

## What you need

| | |
|---|---|
| Tile features | One file per slide (`.pt`, `.npy` or `.h5`), an (n_tiles, dim) matrix. The file name without its extension is the slide's id (`folder_id`). The row order is the tile order that the signal, the masks and training all use. |
| A binary sensitive attribute | Group 1 vs group 0, e.g. White vs other, female vs male, older vs younger. |
| A demographic classifier | An attention-MIL model trained to predict that attribute from features of the same tiles: the same tile grid in the same order, so its masks line up with every feature file of the slide. It need not use the same foundation model as the task model; in the paper, one classifier per (cancer, attribute, slide type) supplied the masks for the task models of all four foundation models. The default adapter fits the gated-attention ABMIL of the paper; any other model plugs in through `--adapter` (step 1). |
| A training pipeline | Either the collaborators' `fairtune_pipeline`, for which an adapter is included (step 3b), or your own, for which a dataset wrapper is included (step 3a). |
| Python ≥ 3.8 | `numpy` and `pandas`; `torch` for steps 1 and 3 (1.13 or later for step 1, which loads features with `weights_only=True`); `pyyaml` for step 3b; `h5py` for `.h5` features. The tests need all of them. Step 3b uses symbolic links, which Windows allows only in Developer Mode or with administrator rights. |

Install with `pip install -e .`, which gives the `unveil-tilefilter` command. Without
installing, use `PYTHONPATH=src python -m unveil_tilefilter`.

---

## The four steps

### 1. Signal: what the demographic classifier sees in each tile

```bash
unveil-tilefilter signal \
    --features-dir /path/to/features/TCGA-BRCA-FS/UNI \
    --checkpoint   /path/to/demographic_classifier.pt \
    --model-code   /path/to/classifier_code \
    --out          /path/to/signals/BRCA_Race_FS
```

For every slide this writes `<out>/<folder_id>.npz` with:
- `attention`: the classifier's softmax weights over the slide's tiles;
- `tile_score`: the probability of group 1 that the classifier gives each tile on its own;
- `slide_pred`: the slide's probability of group 1;
- `n_tiles`.

**One directory holds one classifier's output.**
- `signal_manifest.json` records the checkpoint (path and md5) and the settings. It is
  written before the first slide, so even a killed run names what wrote its files.
- A rerun with the same classifier and settings keeps the slides already done and computes
  the rest.
- A directory written with another classifier or other settings is refused, and so is one
  that has signal files but no manifest. `--overwrite` recomputes such a directory, but only
  if the run covers every slide in it.

**Options:**
- **Model.** `--model-code` is put on `sys.path` so the model's class can be found. The
  released classifiers are plain state_dicts: `--model-config` gives the JSON with
  `model_type` (a class of the `model` module in `--model-code`, or `module:Class`) and
  `model_init_args`, and the weights load with `weights_only=True`. A pickled checkpoint
  loads without it; `--checkpoint-key` then names the entry that holds the model (default
  `model`; `''` for a bare object).
- **Default adapter (`gated_abmil`).** It expects `deep_attention.theta1/theta2/W/temperature`,
  an `mlp` ending in a sigmoid, and `input_dim`.
- **Any other model:** pass `--adapter mymodule:myfunction`, where
  `myfunction(model, x) -> (attention, tile_score, slide_pred)` and `x` is the slide's (n, d)
  float32 tensor. Add `--activation sigmoid` if it returns logits.
- **Your own signal files.** You can also write the signal files yourself
  (`unveil_tilefilter.signal.save_signal`); step 2 reads nothing else. It checks each file
  on loading: probabilities in [0, 1], non-negative attention, finite values.
- **Defaults (as in the paper):** `--chunk 2048`, `--threads 1`, float16 storage. With the
  same torch build, they reproduce the paper's signal files bit for bit.

### 2. Masks: which tiles each training slide keeps

```bash
unveil-tilefilter masks --signals /path/to/signals/BRCA_Race_FS \
    --out /path/to/masks/BRCA_Race_FS_adaptive.npz
```

This writes one `.npz` (slide id → boolean keep-mask) and a `.json` sidecar. The sidecar
holds the filter, every parameter, the seed, `mask_id` (a hash of the rule) and keep-rate
statistics. Add `--check-features-dir` to check each slide's feature rows against its
signal length.

Masks do not depend on the fold or the training seed: every split and every seed of a task
reads the same file.

### 3a. Training in your own pipeline

```python
from unveil_tilefilter.dataset import MaskedBagDataset
from unveil_tilefilter.masks import MaskSet

train_ds = MaskedBagDataset(
    my_train_ds,                                 # returns (bag, ...) per slide
    MaskSet(["/path/to/masks/BRCA_Race_FS_adaptive.npz"]),
    folder_ids=train_folder_ids,                 # one per item, same order
    bag_index=0, length_index=None, max_tiles=2000)
```

- **Wrap only the training dataset.**
- **Turn off your dataset's own tile subsampling.** Set `max_tiles` here instead: the filter
  then sees the whole bag, and the sample is drawn from the tiles it keeps, as in the paper.
- **Every training slide must have a mask.** This is checked when the wrapper is built.
- **Example.** `examples/own_pipeline.py` runs all of this end to end on synthetic data.

### 3b. Training with `fairtune_pipeline`

```bash
unveil-tilefilter fairtune --upstream /path/to/fairtune_pipeline \
    --masks /path/to/masks/BRCA_Race_FS_adaptive.npz --entrypoint mutation -- \
    --cancer brca --train_method agenticFilter --seed 0 --partition 2 \
    --model_path /abs/path/results/... <the pipeline's other arguments>
```

Everything after `--` goes to the pipeline unchanged, including the seed, the split and
every path. The pipeline's code is not edited. The adapter:

1. **Builds an overlay directory** (`--overlay`, default `./unveil_overlay`).
   - It holds symlinks to the pipeline's code, and a `configs/` that points to a copy of
     the pipeline's configs with three train methods appended.
   - `agenticFilter` trains on `adaptive` masks, `agenticRandomFilter` on `random_matched`
     masks, `agenticAttentionOnlyFilter` on `attention_matched` masks.
   - Each method is the pipeline's own `baseline` block, so model, optimiser and model
     selection are the baseline's.
   - Several jobs may share one overlay. A copy that has not changed is left alone, and a
     changed one is swapped in atomically, so a running job never loses its configs.
2. **Checks the pipeline's code.** It compares the five pipeline functions it depends on with
   the version it was tested on, and stops if one differs. Once you have read the
   difference, `--allow-upstream-drift` lets it run.
3. **Filters the training data only.** It replaces `dataset.CancerDataset` and
   `dataset.ReweightDataset` with subclasses that filter the training partition, and gives
   the three methods a results folder of their own, so they never write into the
   baseline's.
4. **Runs the entrypoint.** `--entrypoint` is `mutation` (`main_genetic.py`), or `stage`,
   `grade`, `classification` or `detection` (`main.py`).
5. **Checks what was filtered and writes a record.** A census of every dataset the pipeline
   built must show that only training data was filtered, and that some was.
   - The record goes to `--record`, by default
     `<model_path>/unveil_tilefilter_run_<time>_<pid>.json`.
   - There is one record per run, so runs that share a `--model_path` do not overwrite each
     other's.

**Paths.** The pipeline runs from the overlay directory, so a relative path in its
arguments resolves there. Give absolute paths. The adapter refuses a relative
`--model_path`, and notes it when `--model_path` is missing (the pipeline's default is
relative).

**Platform.** The adapter's dataset classes are made at run time and cannot be pickled. A
filtered run with DataLoader workers (the pipeline's `--n_workers` defaults to 1) therefore
needs the `fork` start method, the default on Linux up to Python 3.13. On macOS, on
Windows, or with Python 3.14 or later (whose default on Linux is `forkserver`), give the
pipeline `--n_workers 0` (and see `--seed-numpy` under *Reproducibility*).

**Baseline.** Run without `--masks` under the pipeline's own method, e.g. `baseline`. The
census then asserts that nothing was filtered.

**Scoring a trained model.** This covers an external cohort, or a TCGA model's own
partitions.
- Add `--inference` to the adapter and the pipeline's `--inference_only` after `--`.
- To re-run the bias test on a finished run's saved results, pass the pipeline's
  `--sig_test_only` instead of `--inference_only`, also with `--inference`.
- **Both write into the run's own folder by default.** The pipeline's
  `--inference_output_path` defaults to `--model_path`, so `--inference_only` replaces the
  run's inference CSVs, and both modes replace its bias-test files. The bootstrap is not
  seeded, so the p-values change and the earlier ones are gone. To keep them, give
  `--inference_only` a separate `--inference_output_path` (its inference and bias-test files
  then go there), or copy the run folder before `--sig_test_only`.
- Keep the `--train_method` the model was trained under, so the pipeline finds that
  model's folder.
- `--inference` takes no masks, and the census asserts that no partition was filtered,
  the training partition included.
```bash
unveil-tilefilter fairtune --upstream /path/to/fairtune_pipeline --inference \
    --entrypoint mutation -- --cancer brca --train_method agenticFilter --seed 0 \
    --model_path /abs/path/results/... --inference_only <the pipeline's arguments for the cohort>
```

**The adapter refuses runs that would be filed or scored wrongly:**
- a mask file built with a different filter from the train method's;
- masks given while `--train_method` is not one of the three;
- one of the three methods given without masks, outside `--inference`;
- `--inference` with masks, or without the pipeline's `--inference_only` or
  `--sig_test_only`;
- `--inference_only` without `--inference`, which would score the training partition on
  filtered bags;
- `--sig_test_only` without `--inference`: it builds no dataset, so with masks the census
  would fail only after the pipeline had rewritten the run's bias-test files;
- `--inference_only` and `--sig_test_only` together: the pipeline would only re-run the
  bias test, on the saved inference CSVs;
- a relative `--model_path`;
- an abbreviated or repeated flag that the adapter reads;
- a two-cancer task passed as one quoted token.

`--check-only` builds the overlay and checks the pipeline without running anything.

### 4. Resolve: is the bias resolved?

```bash
unveil-tilefilter resolve --run-dir /path/to/filtered/task_folder \
    --baseline-dir /path/to/baseline/task_folder          # or --baseline-auroc 0.81
unveil-tilefilter resolve --table runs.csv --out resolved.csv   # many runs:
    # runs.csv columns: name, run_dir, baseline_dir or baseline_auroc
```

The study's rule: a biased task is **resolved** when the mitigated model no longer shows a
significant EOddAbs disparity and its AUROC is not significantly lower than the baseline's
by a paired permutation test. On patient-level predictions this is
`03_fairness_evaluation/compare_models.py` (column `resolved`).

This tool scores `fairtune_pipeline` task folders instead: the latest complete run's
`<k>_bootstrapTest_concatenate_*.csv` files and `<k>_run_status.json`. Those files hold no
paired test, so it applies the fairness half exactly and a margin for the AUROC half:

- **fair:** the EOddAbs bias-test p-value is ≥ `--alpha` (0.05); `--metrics` adds the
  pipeline's other metrics (EOpp0, EOpp1, EBAcc, AUCDiff, ACC_diff), all of which must then
  be non-significant and present; `--p-rule gt` asks for p > alpha instead;
- **keeps AUROC:** AUROC − baseline AUROC ≥ −`--auroc-margin` (0.02), computed in that
  order in floating point.
  - The baseline is the task's original unmitigated model, the one the biased tasks were
    selected on.
  - The margin is a threshold, not a significance test; use `compare_models.py` for the
    paired test.

**Failed folds.** A run whose status record shows a failed fold is not scored. The pipeline
catches a fold's training error, goes on to the next fold and still writes a complete
bootstrap, so the files alone look like a finished run.

**`--ignore-error-keys`** applies the rule the pipeline's authors use for their own runs.
- It scores a "success" run despite `error_class`, `error_msg` and `traceback` when both
  bootstrap files were written no earlier than the run's `ended_at`.
- It cannot tell keys left by an earlier attempt that was rerun cleanly from a fold that
  failed in the same attempt: both look the same.
- The paper applied it only to the collaborators' runs, after checking their job logs. Use
  it only for runs whose logs you have checked.

**The p-values are the pipeline's.** Each tests the run's own group gap against a
shuffled-label null; none compares the run with the baseline. The pipeline's bootstrap is
not seeded, so p-values change slightly if it is rerun.

---

## The filter, exactly

For a slide of `n` tiles (all defaults are the paper's values):

```
if n <= min_tiles (64):  keep every tile
evidence_i = tile_score_i        if slide_pred >= decision_threshold (0.5)
             1 - tile_score_i    otherwise
rank_i     = tile i's attention rank, 0..1 (equal attention values share a rank)
score_i    = evidence_i * (evidence_base + attention_gain * rank_i) + attention_bonus * rank_i
                          (0.25)            (0.75)                    (0.02)
drop       = min(floor(drop_frac * n), n - max(min_tiles, ceil(min_keep_frac * n)))
                       (0.33)                    (64)            (0.05)
remove the `drop` highest-scoring tiles (equal scores: lower tile index first)
```

Every number is an option of `unveil-tilefilter masks`: `--drop-frac`, `--evidence-base`,
`--attention-gain`, `--attention-bonus`, `--decision-threshold`, `--min-tiles` and
`--min-keep-frac`.

**Origin of the scoring rule.** An LLM-driven evolutionary search (OpenEvolve) found it on
31 TCGA development tasks, and it was then locked. The search is not part of this package
and is not needed to apply the method.

`reference/filters_locked_c844fcf6.py` is the locked file the paper's runs used, kept
verbatim. The tests compare this package against it. Its `evolved_v0` docstring still
describes an earlier candidate (0.8·evidence + 0.2·rank, dropping 30%). The code under that
docstring is the rule above, and the code is what ran.

**Controls** (`--filter`):
- `random_matched` drops the same number of tiles from each slide, at random. It uses one
  fixed draw per slide for a given `--seed`, default 0.
- `attention_matched` drops the same number of tiles: those with the highest attention.
- `identity` drops nothing.

The two matched controls differ from the method only in which tiles they remove, not in
how many.

---

## Choices that are yours

- **Split and training seed.** These are the pipeline's arguments. In `fairtune_pipeline`,
  `--seed` with `--partition 2` fixes both the 4-fold patient split and the weight
  initialisation.
- **The demographic classifier.** In the paper, each (cancer, attribute, slide type) used
  one classifier: a fold model from the collaborators' 10-fold demographic experiments,
  the one with the best AUC at 20x. It was applied to every slide, so for many slides the
  signal is in-sample. The filter touches only training slides, but you may prefer a
  classifier trained on your training patients only.
- **Which slides get masks.** By default, every slide with a signal (`--slides` narrows
  it). Only the training slides' masks are used.
- **The baseline for `resolve`.** Give the unmitigated run that defines the biased task.

## Reproducibility

- **Masks** are a deterministic function of the signals and the parameters.
  `random_matched` draws with numpy's `Generator`; its sidecar records the numpy version.
- **Signals** are bit-identical with the same torch build, one thread and chunk 2048.
- **The `max_train_tiles` subsample.** In `fairtune_pipeline` it is drawn with numpy
  inside the DataLoader.
  - **With `--n_workers` 1 or more** (the pipeline's default): torch seeds numpy in each
    worker from the run's torch seed, so the draw follows `--seed` and differs from epoch
    to epoch.
  - **With `--n_workers 0`:** the draw uses the main process's numpy RNG, which the
    pipeline does not seed. `--seed-numpy` seeds it, and has no effect when there are
    workers.
- **Training is not bit-reproducible.** In our runs, two trainings with the same seed did
  not give identical results; GPU kernels, for one, are not deterministic by default.

---

## Verification

These were run on 2026-10-10, on the paper's own files.

| Check | Result |
|---|---|
| `tests/run_all.py`: unit and integration tests, with no data needed | 7 files, 44 tests: all pass |
| `verification/eq_masks.py`: every mask file in the reproduction package, rebuilt from the paper's signals with default parameters | 107 of 107 files identical: 102,851 slides |
| `verification/eq_signals.py`: stored signals recomputed with the paper's classifiers, in the environment that made them | 30 of 30 slides identical: 3 classifiers, 10 slides each |
| `verification/eq_resolve.py`: every run folder of the reproduction package, parsed by this package and by the paper's own code (the pre-revision six-metric rule, passed explicitly) | 1,629 folders: 1,601 scored by both, 28 refused by both, 0 disagreements |
| `verification/eq_fairtune.py`: the adapter on the real `fairtune_pipeline` | 22 of 22 checks pass |

The `eq_fairtune` checks:
- the overlay's config reads back identical to the paper's, and `configs/` is the
  overlay's own copy;
- the pipeline's code matches the tested version;
- only training data is filtered;
- `max_train_tiles` is drawn from the kept tiles;
- `ReweightDataset` is filtered too;
- the census assertion fails when tampered with.

The folders that both parsers refuse are the collaborators' original adversarial-debiasing
runs with error keys left by an earlier attempt. `eq_resolve.py --manifest` reads each one
at its source folder in the collaborators' tree, which keeps the original file times, and
`--ignore-error-keys` admits all 28 there.

`verification/README.md` gives the commands.

## What is not included

- **The collaborators' code**: `fairtune_pipeline` and the demographic-classifier training
  code. Use your own copies.
- **The evolutionary search** that designed the filter.
- **The paper's analyses**: tables, head-to-head comparisons, paired tests. They are in
  the reproduction package.

## Contents

```
src/unveil_tilefilter/
  filters.py    the filter, the two matched controls, their invariants
  signal.py     step 1: signal from the demographic classifier
  masks.py      step 2: mask files and their sidecars
  dataset.py    step 3a: wrapper for your own training dataset
  fairtune.py   step 3b: fairtune_pipeline adapter (+ fairtune_tested_pins.json)
  resolve.py    step 4: the resolution rule
  cli.py        the unveil-tilefilter command
tests/          run_all.py and seven test files (no data needed)
verification/   eq_*.py checks against the paper's files, README.md
reference/      filters_locked_c844fcf6.py, the paper's locked filter file
examples/       own_pipeline.py and toy_models.py, a synthetic end-to-end example
```
