# Example Data Structure

This directory contains mock example data demonstrating the required format for running UNVEIL.

## Directory Structure

```
example_data/
├── mutation_prediction/          # Data for mutation prediction pipeline
│   ├── tcga_pan_cancer/          # TCGA Pan-Cancer Atlas data
│   ├── clinical_information/     # Clinical metadata
│   └── features/                 # Pre-extracted WSI features
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

