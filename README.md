
```markdown
# MIMIC-SBDH Reproduction Study

This repository contains the implementation of our reproduction study of "MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health" ([Ahsan et al., 2021](https://github.com/hiba008/MIMIC-SBDH)).

## Overview

We reproduce and extend three machine learning approaches for SBDH classification:
- Random Forest
- XGBoost
- Bio-ClinicalBERT

## Requirements

### Environment
- Python 3.7+
- CUDA 11.3 (for GPU support)
- 16GB RAM minimum, 32GB recommended
- GPU with 8GB VRAM (for Bio-ClinicalBERT)

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas==1.3.5
- numpy==1.21.6
- scikit-learn==1.0.2
- xgboost==1.6.2
- torch==1.10.0
- transformers==4.18.0
- imblearn==0.9.1

## Data

The implementation uses two main data files:
- filtered_social_history_data.csv
- MIMIC-SBDH.csv

## Models

### Random Forest
```python
from models.random_forest import SBDHClassifier
```
- Implementation without oversampling
- Implementation with RandomOverSampler

### XGBoost
```python
from models.xgboost import SBDHClassifier
```
- Base implementation
- Implementation with class balancing

### Bio-ClinicalBERT
```python
from models.bert import BioClinicalBERTClassifier
```
- Fine-tuned implementation for SBDH classification

## Results

Performance comparison across models:

| SBDH Category | RF | RF+OS | XGB | XGB+OS | BERT |
|---------------|-------|--------|------|---------|-------|
| Community-Present | 0.9000 | 0.9184 | 0.9310 | 0.9257 | 0.9568 |
| Community-Absent | 0.8644 | 0.6026 | 0.9055 | 0.8903 | 0.9272 |
| Education | 0.7923 | 0.6097 | 0.8648 | 0.8782 | 0.8995 |
| Economics | 0.8502 | 0.8458 | 0.8608 | 0.8586 | 0.8731 |
| Environment | 0.7928 | 0.6936 | 0.9030 | 0.8994 | 0.8744 |
| Alcohol Use | 0.6091 | 0.6585 | 0.6882 | 0.6932 | 0.8037 |
| Tobacco Use | 0.7174 | 0.7403 | 0.7667 | 0.7679 | 0.8122 |
| Drug Use | 0.4705 | 0.4836 | 0.6349 | 0.6455 | 0.6871 |

## Usage

```python
# Example code for running models
from models import random_forest, xgboost, bert

# Load data
social_history_df = pd.read_csv('filtered_social_history_data.csv')
sbdh_df = pd.read_csv('MIMIC-SBDH.csv')

# Train and evaluate models
# See individual model documentation for specific usage
```

## Extensions

We implemented several extensions to the original paper:
1. Comparison of oversampling techniques (SMOTE vs RandomOverSampler)
2. Ablation studies on feature importance
3. Memory optimization for Bio-ClinicalBERT

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Khang Nguyen khang2@illinois.edu
- Duy Nguyen duyn2@illinois.edu

## Acknowledgments

This work builds upon the MIMIC-SBDH dataset and implementation by Ahsan et al. (2021).
```
