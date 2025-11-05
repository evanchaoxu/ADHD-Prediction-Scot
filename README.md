# Predicting ADHD in 415,168 unselected schoolchildren using machine learning on Scotland-wide linked education and health datasets

This repository provides a modular framework for training, evaluating, and interpreting machine learning classifiers on structured tabular data. The design supports flexibility in preprocessing, model training, ensemble learning, and post hoc analysis, making it suitable for research and review.
> **This repository accompanies the article *Predicting ADHD in 415,168 unselected schoolchildren using machine learning on Scotland-wide linked education and health dataset*.**


---

## Repository Layout

```
.
├── main.py                 # Orchestrates end-to-end pipeline
├── preprocessing.py        # Cleaning, imputation, feature scaling/encoding
├── model_training.py       # Core training logic for individual models, including model selection, fitting, and saving
├── metrics.py              # Statistical information and competence score related functions
├── ensemble.py             # Majority/Competence-based voting ensembling
├── breakdown.py            # Subgroup/score-band performance analysis
├── plot_auc.py             # Visualisation module for generating and saving ROC-AUC plots
├── feature_importance.py   # Model-based feature importance
└── test_new_data.py        # Inference on unseen data
```

---

## Requirements

- Python **3.8+** (recommended 3.10 or newer)
- Packages(core pipeline):
  ```bash
  pip install numpy pandas scikit-learn scipy tensorflow xgboost imblearn matplotlib seaborn joblib
  ```

---

## Usage Instructions

### 1) Run the full pipeline

Execute the following to run the complete pipeline:

```bash
python main.py
```

This typically includes: data loading, preprocessing, model training, evaluation, visualisation/exports (optional).


### 2) Inference on new data

To evaluate new data using a previously trained model:

```bash
python test_new_data.py
```

> Ensure the script is pointed to the saved model artefacts produced during training (e.g., `.pkl` files) and that the preprocessing applied to inference data mirrors training-time transformations.


### 3) Optional Modules

Each module is standalone and can be executed independently for debugging or advanced customisation. For example:

```bash
python preprocessing.py
python model_training.py
python metrics.py
python feature_importance.py
python ensemble.py
python plot_auc.py
python breakdown.py
```


---

## Typical Pipeline Flow

1. `preprocessing.py` → data cleaning, encoding, scaling  
2. `model_training.py` → train base learners  
3. `metrics.py` → compute performance metrics  
4. `ensemble.py` → combine model outputs (if used)
5. `main.py` → Main exection
5. `feature_importance.py` → interpret model contributions  
6. `plot_auc.py` → generate ROC/AUC figures  
7. `test_new_data.py` → apply to new/unseen data  
8. `breakdown.py` → post hoc performance analysis

---

## Notes

- **Paths & parameters**: update file paths, model lists, and hyperparameters inside each script.
- **Additional models**: any scikit-learn compatible estimator should slot into `model_training.py` with minimal changes.
- **Logging**: consider adding experiment tracking (e.g., logs) for large ablations.


---

## Citation

If you need refer this codebase in academic work, please cite the associated paper or contact the author for citation details.

