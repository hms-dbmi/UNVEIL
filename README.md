# UNVEIL: Uncovering Non-apparent Visual Encodings in Latent space

This repository implements UNVEIL, a framework for identifying, quantifying, and mitigating demographic-associated signals in pathology foundation model representations. By integrating demographic classification, nuclear morphometric analysis, and demographic signal-aware agentic scheduling, UNVEIL addresses performance disparities in computational pathology tasks (demonstrated here using mutation prediction as an exemplar).

## Overview

Pathology foundation models encode demographic-linked morphological variations during training, which can contribute to performance disparities across population groups. UNVEIL provides a systematic approach to:

1. Detect demographic signals in learned representations through demographic attribute classification
2. Quantify the contribution of demographic-associated features to downstream task performance disparities
3. Mitigate disparities using demographic signal-aware agentic scheduling that adaptively modulates patch contributions

### Framework Workflow

```
WSI Patches → Foundation Model Encoding → Attention-Based MIL Aggregation
                                                          ↓
                                    ┌─────────────────────┴──────────────────────┐
                                    ↓                                             ↓
                        Demographic Classifier                    Downstream Task Model
                     (Age, Race, Sex prediction)              (Mutation prediction, etc.)
                                    |                                             ↓
                                    |                                   Fairness Evaluation
                                    |                                             ↓
                                    └────────────────→  Demographic Signal-Aware Agentic Scheduling
                                                        (Adaptive patch contribution modulation)
```

The framework operates in three stages:

1. **Demographic Signal Detection**: Train classifiers to predict demographic attributes (age, race, sex) from WSI representations, quantifying the extent of demographic information encoded in foundation model features.

2. **Fairness Assessment**: Train downstream task models (mutation prediction as exemplar) and evaluate performance disparities across demographic groups, linking disparities to demographic-predictive signals through attention mapping and nuclear morphometric analysis.

3. **Disparity Mitigation**: Apply demographic signal-aware agentic scheduling that uses multi-factor routing to selectively modulate patches most strongly associated with demographic signals, reducing performance gaps while maintaining overall predictive accuracy.

## Repository Structure

```
.
├── demographic_classifier/          # Demographic attribute prediction
│   ├── configs/                     # Configuration examples
│   ├── run.py                       # Main entry point
│   ├── run.sh                       # Execution wrapper
│   ├── experiment_runner.py         # Training orchestration
│   ├── dataset.py                   # Feature dataset loaders
│   └── model.py                     # MLP classifier architectures
│
├── mutation_prediction/             # Downstream task pipeline (mutation prediction exemplar)
│   ├── configs/                     # Dataset and attribute mapping configurations
│   ├── main_genetic.py              # Main training script
│   ├── dataset.py                   # Dataset loading and preprocessing
│   ├── network.py                   # Attention-based MIL architectures
│   ├── demographic_agent.py         # Base demographic-aware filtering agent
│   ├── unified_demographic_agent.py # Multi-factor routing implementation
│   ├── run_baseline.sh              # Standard training without demographic awareness
│   └── run_agentic.sh               # Training with demographic signal-aware scheduling
│
├── example_data/                    # Mock data demonstrating required formats
│   ├── demographic_classifier/      # Demographic classification data
│   ├── mutation_prediction/         # Exemplar downstream task data
│   └── README.md                    # Detailed data format specifications
│
├── requirements.txt                 # Python dependencies for all components
└── LICENSE                          # GNU AFFERO GENERAL PUBLIC LICENSE v3.0
```

## Data Requirements

### Input Data Format

Both components require:

1. **WSI Feature Embeddings** (`.pt` files): Pre-extracted patch-level features from foundation models
   - Format: `{'features': torch.Tensor}` with shape `(N_patches, feature_dim)`
   - Supported dimensions: CHIEF (768), UNI (1024), GIGAPATH (1536), VIRCHOW2 (2560)

2. **Metadata Files**:
   - **Demographic Classifier**: JSON files mapping slide IDs to demographic categories
   - **Mutation Prediction**: TCGA Pan-Cancer Atlas structure with clinical metadata and mutation status

### Example Data

Mock example data is provided in `example_data/` to demonstrate required formats:
- 5 mock WSI feature files with CHIEF embeddings (768-dim)
- Clinical metadata and mutation status files for BRCA
- Demographic label files (age, race, sex)
- Example configuration file for demographic classifier

See `example_data/README.md` for detailed specifications and instructions for generating mock data.


## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- SLURM cluster environment (for batch jobs)

### Setup

```bash
# Clone repository
git clone <repository_url>
cd HiddenFeature_Bao_new_github

# Create environment
conda create -n unveil python=3.8
conda activate unveil

# Install dependencies
pip install -r requirements.txt

# Create output directories
mkdir -p output/{mutation_models,mutation_models_agentic,demographic_classifier}
mkdir -p logs/{train_TCGA_mutation,demographic_agentic}
```

## Usage

### I: Train Demographic Classifier

Train classifiers to detect demographic signals in foundation model representations:

```bash
cd demographic_classifier
./run.sh configs/example_train.json
```

**Key configuration parameters**:
- `train_targets_file_path_list`: Paths to demographic label JSON files
- `features_dir_path_list`: Directories containing WSI feature `.pt` files
- `model_init_args.input_dim`: Feature dimension matching foundation model
- `model_init_args.output_dim`: Number of demographic categories

Output: Trained classifiers saved in specified `save_dir` with performance metrics.

### II: Train Downstream Task Models and Assess Fairness

Train mutation prediction models (or other pathology tasks) and evaluate performance across demographic groups:

#### Baseline (No Demographic Awareness)

```bash
cd mutation_prediction
sbatch run_baseline.sh
```

This runs standard attention-based MIL training across all 31 cancer types without demographic-aware filtering.

**Key parameters** (edit in `run_baseline.sh`):
- `FOUNDATION_MODEL`: Feature extractor (CHIEF, UNI, GIGAPATH, VIRCHOW2)
- `SENSITIVE`: Demographic attribute to track (age, race, sex)
- `cancer`: Cancer type (lowercase abbreviation)

Output: Models and performance metrics saved in `./output/mutation_models/{ATTRIBUTE}/{DATA_SOURCE}/{FOUNDATION_MODEL}/{SLIDE_TYPE}/`

### III: Apply Demographic Signal-Aware Agentic Scheduling

Mitigate performance disparities using adaptive patch contribution modulation:

```bash
cd mutation_prediction
sbatch run_agentic.sh
```

This applies multi-factor routing that selectively modulates patches based on:
1. **Demographic model accuracy**: Use random filtering if accuracy < 0.60 (unreliable predictions)
2. **Group imbalance**: Conservative filtering if imbalance > 5.0 in early training (< 150 slides)
3. **Training progress**: Random filtering in early phase (< 200 slides), signal-leveraging in late phase (≥ 200 slides)

**Routing decision logic**:
```
if demographic_accuracy < 0.60:
    → Random filtering (model unreliable)
elif imbalance_ratio > 5.0 AND total_slides < 150:
    → Conservative random (preserve minority)
elif total_slides < 200:
    → Random filtering (early training)
else:
    → Signal-leveraging filtering (late training)
```

This approach ensures zero data leakage: all decisions use only training data statistics (group sizes, demographic model accuracy computed on training set, training progress metrics).

Output: Models saved in `./output/mutation_models_agentic/{ATTRIBUTE}/TCGA/{FOUNDATION_MODEL}/FS/agent_demographic_agentic/` with agent decision logs.


## Output Structure

```
output/
├── mutation_models/                          # Baseline models
│   └── {ATTRIBUTE}/{DATA_SOURCE}/{FOUNDATION_MODEL}/{SLIDE_TYPE}/
│       └── {CANCER}_{GENE}/
│           ├── checkpoint_best.pt            # Best model weights
│           ├── results.csv                   # Performance metrics
│           └── inference_results_fold{N}.csv # Per-fold predictions
│
├── mutation_models_agentic/                  # Agentic scheduling models
│   └── {ATTRIBUTE}/TCGA/{FOUNDATION_MODEL}/FS/agent_demographic_agentic/
│       └── {CANCER}_{GENE}/
│           ├── checkpoint_best.pt
│           ├── results.csv
│           ├── inference_results_fold{N}.csv
│           └── agent_logs/                   # Routing decision logs
│
└── demographic_classifier/
    └── {save_dir}/
        ├── configs.json                      # Configuration
        ├── best_model.pt                     # Trained classifier
        └── metrics/                          # Performance metrics
```


## Adapting to Other Downstream Tasks

While this repository demonstrates UNVEIL using mutation prediction as an exemplar, the framework is task-agnostic. To adapt to other pathology tasks:

1. Modify `main_genetic.py` to load task-specific labels and loss functions
2. Update `dataset.py` to handle task-specific data formats
3. Adjust `network.py` output dimensions for task requirements
4. Maintain the same demographic classifier and agentic scheduling pipeline

The core UNVEIL methodology (demographic signal detection → fairness assessment → agentic mitigation) remains applicable across pathology prediction tasks.

## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.0. See the [LICENSE](LICENSE) file for details.


