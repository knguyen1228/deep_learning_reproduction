
```markdown
# MIMIC-SBDH Reproduction Study

## Description
This repository contains a reproduction study of "MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health" ([Ahsan et al., 2021](https://github.com/hiba008/MIMIC-SBDH)). We implement and compare three approaches: Random Forest, XGBoost, and Bio-ClinicalBERT.
```
## Directory Structure
```
├── data/
│   ├── processed/           # Preprocessed data files
│   └── raw/                # Original MIMIC-SBDH data
├── models/
│   ├── random_forest.py    # Random Forest implementation
│   ├── xgboost_model.py    # XGBoost implementation
│   └── bert_model.py       # Bio-ClinicalBERT implementation
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── scripts/
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
├── results/
│   ├── figures/           # Generated figures
│   └── metrics/          # Performance metrics
├── requirements.txt
├── environment.yml
└── README.md
```

## Installation

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate sbdh_env

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements
- CPU: 16GB RAM minimum, 32GB recommended
- GPU: NVIDIA GPU with 8GB VRAM (for Bio-ClinicalBERT)
- Tested on: NVIDIA T4 GPU

### Dependencies
```
python>=3.7
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
xgboost==1.6.2
torch==1.10.0
transformers==4.18.0
imblearn==0.9.1
```

## Data Preparation
1. Download MIMIC-SBDH dataset
2. Place files in `data/raw/`:
   - filtered_social_history_data.csv
   - MIMIC-SBDH.csv
3. Run preprocessing:
```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
```

## Training

### Random Forest
```bash
python scripts/train.py --model rf --data data/processed/train.csv --output models/rf
```

### XGBoost
```bash
python scripts/train.py --model xgb --data data/processed/train.csv --output models/xgb
```

### Bio-ClinicalBERT
```bash
python scripts/train.py --model bert --data data/processed/train.csv --output models/bert
```

## Evaluation
```bash
python scripts/evaluate.py --model [rf|xgb|bert] --checkpoint [path_to_model] --test data/processed/test.csv
```

## Reproducibility

### Random Seeds
All random seeds are set to 42:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### Hyperparameters
Random Forest:
```python
params = {
    'n_estimators': 100,
    'max_features': 'sqrt',
    'min_samples_split': 2
}
```

XGBoost:
```python
params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'reg_lambda': 1.0
}
```

Bio-ClinicalBERT:
```python
params = {
    'batch_size': 8,
    'learning_rate': 2e-5,
    'epochs': 5,
    'max_length': 128
}
```

## Results

### Performance Metrics
Detailed results are available in `results/metrics/`

### Reproducing Figures
```bash
python scripts/generate_figures.py --results results/metrics --output results/figures
```

## Extension Studies
1. Oversampling Comparison
```bash
python scripts/train.py --model rf --sampling [random|smote] --data data/processed/train.csv
```

2. Ablation Studies
```bash
python scripts/ablation.py --model [rf|xgb] --component [feature_selection|preprocessing]
```

## Citations
```bibtex
@article{ahsan2021mimic,
    author = {Ahsan, Hiba and Ohnuki, Emmie and Mitra, Avijit and Yu, Hong},
    title = {MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health},
    journal = {Proceedings of Machine Learning Research},
    year = {2021}
}
```

## License
This project is licensed under the MIT License.

## Contact
[Your Name] - [Your Email]

## Acknowledgments
- Original MIMIC-SBDH implementation by Ahsan et al.
- Computing resources provided by [Institution]
```
