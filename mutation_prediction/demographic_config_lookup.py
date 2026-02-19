"""
Demographic model configuration lookup from summary_ensemble.csv.
Finds the best demographic classifier for a given (cancer, attribute, foundation_model).
"""

import pandas as pd
from pathlib import Path

CANCER_CODE_MAP = {
    'BRCA': '04',  # Breast cancer
    'LGG': '09',   # Low grade glioma
    'GBM': '08',   # Glioblastoma
    'KIRC': '03'   # Kidney renal clear cell carcinoma
}

def get_best_demographic_config(cancer, attribute, foundation_model, 
                                summary_csv_path='./demographic_classifier/results/summary_ensemble.csv'):
    """
    Query summary_ensemble.csv to find the best demographic model configuration.
    
    Args:
        cancer: Cancer type ('BRCA', 'LGG', 'GBM', 'KIRC')
        attribute: Demographic attribute ('Age', 'Race', 'Sex')
        foundation_model: Foundation model ('CHIEF', 'UNI', 'GIGAPATH', 'VIRCHOW2')
        summary_csv_path: Path to summary_ensemble.csv
    
    Returns:
        dict: {
            'exp_id': int,
            'fold_id': int,
            'metric': str,
            'cancer_code': str
        }
    
    Raises:
        ValueError: If combination not found in summary CSV
    """
    df = pd.read_csv(summary_csv_path, skiprows=1)  # Skip the first header row
    
    # Filter by Cancer, Task (attribute), Foundation Model
    mask = (df['Cancer'].str.upper() == cancer.upper()) & \
           (df['Task'] == attribute) & \
           (df['Foundation Model'].str.upper() == foundation_model.upper())
    
    matching_rows = df[mask]
    
    if len(matching_rows) == 0:
        raise ValueError(f"No demographic model found for Cancer={cancer}, Attribute={attribute}, Model={foundation_model}")
    
    row = matching_rows.iloc[0]
    
    cancer_code = CANCER_CODE_MAP.get(cancer.upper())
    if cancer_code is None:
        raise ValueError(f"Unknown cancer type: {cancer}. Must be one of {list(CANCER_CODE_MAP.keys())}")
    
    return {
        'exp_id': int(row['EXP']),
        'fold_id': int(row['FOLD']),
        'metric': str(row['METRIC']),
        'cancer_code': cancer_code
    }


if __name__ == '__main__':
    # Test
    config = get_best_demographic_config('BRCA', 'Age', 'CHIEF')
    print(f"BRCA-Age-CHIEF best config: {config}")
    
    config = get_best_demographic_config('LGG', 'Sex', 'UNI')
    print(f"LGG-Sex-UNI best config: {config}")

