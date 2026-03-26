# Example Data Structure

This directory contains mock example data demonstrating the required format for running UNVEIL.

## Directory Structure

```
example_data/
├── mutation_prediction/          # Data for mutation prediction pipeline
│   ├── tcga_pan_cancer/          # TCGA Pan-Cancer Atlas data
│   ├── clinical_information/     # Clinical metadata
│   ├── features/                 # Pre-extracted WSI features
│   └── models/                   # Pre-trained example models
│       ├── baseline/             # Baseline models (no demographic agent)
│       └── agentic/              # Agentic models (with demographic agent)
└── demographic_classifier/       # Data for demographic classifier
    ├── features/                 # Pre-extracted WSI features
    └── labels/                   # Demographic labels (JSON format)
```

## Mutation Prediction Data Requirements

### 1. TCGA Pan-Cancer Atlas Structure

```
tcga_pan_cancer/
└── {cancer}_tcga_pan_can_atlas_2018/
    ├── clinical_data.tsv         # Clinical metadata (TSV format)
    └── Common Genes/             # Gene mutation data
        └── Common Genes_{GENE}_/
            └── data.csv          # Mutation status per patient
```

**Note**: The directory naming format `Common Genes_{GENE}_` (with trailing underscore) is required by the data loading code. The gene name should be the standard gene symbol (e.g., TP53, PIK3CA, KRAS).

**Note**: The directory naming format `Common Genes_{GENE}_` (with trailing underscore) is required by the data loading code. The gene name should be the standard gene symbol (e.g., TP53, PIK3CA, KRAS).

**clinical_data.tsv** columns (key fields):
- `Patient ID`: Unique patient identifier
- `Sex`: Male/Female
- `Race Category`: Demographic race category
- `Diagnosis Age`: Age at diagnosis (years)
- Additional clinical fields...

**data.csv** columns (mutation status):
- `Patient ID`: Matches clinical_data.tsv
- `Altered`: Binary mutation status (0=wild-type, 1=mutated)

### 2. Clinical Information Files

```
clinical_information/TCGA/
└── {CANCER}_clinical_information.csv
```

CSV format with columns:
- `case_submitter_id`: Patient identifier
- `age`: Age group (old/young) or numeric age
- `race`: Race category
- `sex`: Male/Female
- Additional clinical fields...

### 3. WSI Feature Files

```
features/
└── {SLIDE_ID}.pt
```

PyTorch tensor format:
```python
torch.Tensor  # Shape: (N_patches, feature_dim)
# feature_dim: CHIEF=768, UNI=1024, GIGAPATH=1536, VIRCHOW2=2560
# Example: torch.Size([3873, 768]) for a slide with 3873 patches using CHIEF
```

### 4. Pre-trained Models (for inference-only mode)

```
models/
├── baseline/                     # Baseline models (no demographic agent)
│   └── brca/
│       └── 4_brca_Common Genes_{GENE}-Percentage_{FREQ}_2/
│           ├── 1_{cancer}{GENE}__fold_0/
│           │   └── model.pt      # Model weights
│           └── 1_result.csv      # Training results
└── agentic/                      # Agentic models (with demographic agent)
    └── brca/
        └── 4_brca_Common Genes_{GENE}-Percentage_{FREQ}_2/
            ├── 1_{cancer}_{GENE}_fold_0/
            │   └── model.pt      # Model weights
            └── 2_result.csv      # Training results
```

**Directory Structure Requirements**:
- Task prefix: `4` indicates mutation prediction task (task=4)
- Cancer folder: `{cancer}` in lowercase (e.g., `brca`)
- Gene folder: Format `4_{cancer}_Common Genes_{GENE}-Percentage_{FREQ}_2`
- Model folder: Format `{index}_{cancer}{GENE}[_]{fold}` where:
  - `index`: Model iteration/version number (e.g., 1, 2)
  - Baseline: Double underscore before fold (e.g., `1_brcaPIK3CA__fold_0`)
  - Agentic: Single underscore before fold (e.g., `1_brca_PIK3CA_fold_0`)
  - `fold`: Cross-validation fold number (0-3 for 4-fold CV)

**Using Pre-trained Models**:

To run inference with example baseline model:
```bash
cd mutation_prediction
python main_genetic.py \
    --cancer brca \
    --model_path ../example_data/mutation_prediction/models/baseline/ \
    --genes PIK3CA \
    --partition 2 \
    --inference_only \
    --inference_mode test
```

To run inference with example agentic model:
```bash
cd mutation_prediction
python main_genetic.py \
    --cancer brca \
    --model_path ../example_data/mutation_prediction/models/agentic/ \
    --genes PIK3CA \
    --partition 2 \
    --inference_only \
    --inference_mode test \
    --use_demographic_agent \
    --demographic_strategy unified
```

## Demographic Classifier Data Requirements

### 1. Feature Files

Same format as mutation prediction (`.pt` files with extracted features).

### 2. Label Files

```
labels/
└── {DATASET}-{CANCER}-{ATTRIBUTE}.json
```

JSON format:
```json
{
    "SLIDE_ID_1": "category1",
    "SLIDE_ID_2": "category2",
    ...
}
```

Example attributes:
- `age`: "old" or "young"
- `race`: "White" or "Black or African American"
- `sex`: "Male" or "Female"


## Notes

- All patient/slide IDs in example data are mock identifiers (MOCK-SAMPLE-001 through 005)
- Clinical and label data are mock/synthetic for demonstration purposes
- Real data requires appropriate IRB approval and data use agreements
- Features should be pre-extracted using foundation models (CHIEF, UNI, GIGAPATH, VIRCHOW2)
- TCGA data: https://portal.gdc.cancer.gov/
- TCGA Pan-Cancer Atlas: https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html

