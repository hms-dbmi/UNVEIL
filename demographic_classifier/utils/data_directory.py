# Foundation model feature dimensions
# Used for configuring model input dimensions

FOUNDATION_DIM = {
    "CHIEF": 768,
    "GIGA": 1536,
    "GIGAPATH": 1536,  
    "VIRCHOW2": 2560,
}

# Regular expressions for dataset filtering
# Used for filtering TCGA samples by tissue type
TCGA_REGEX_FILTERS = {
    'cancer': r'^(?!.*-11[AB]-).*',      # Exclude normal tissue samples (11A/11B)
    'normal': r'^.*-11[AB]-.*$',         # Include only normal tissue samples
    'all': r'.*'                         # Include all samples
}
