# Test new data
# This file inculdes load updated_models.pkl and use it for testing new data.

import numpy as np
import pandas as pd
import pickle
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from preprocessing import load_and_prepare_data
from ensemble import save_results_to_csv, majority_vote, competence_weighted_predictions, full_competence_weighted_predictions, display_vote_result
from breakdown import compute_and_save_competence_breakdown, compute_and_save_full_competence_breakdown
import model_training
import matplotlib.pyplot as plt

raw_data_path = "dataset_3_ordinal(no_feature_selection).csv"

# Create a new log folder
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, 'testing_stage.log'), level=logging.INFO)

target = 'adhd'
nominal_vars = ["pupil_ethnicity", "MODE_DEL"]
ordinal_vars = ['AGECAT', 'Dep_quintile', 'MATERNAL_AGECAT', 'Parity', 'bweight_centile', 'fiveminapgar_cat', 'previous_exclusion1']
continuous_vars = ['Gest_age','previous_absence']

df = load_and_prepare_data(raw_data_path, ordinal_vars, continuous_vars)

binary_vars = [c for c in df.columns[:-1] if c not in ordinal_vars + nominal_vars + continuous_vars]

hyperSMUMs_data_2, majority_vote_data_2 = train_test_split(df, test_size=0.2, stratify=df[target], random_state=123)

X_vote_data = majority_vote_data_2.drop(columns=[target])
y_vote_data = majority_vote_data_2[target]


# algorithms = ['GLM', 'RandomForest', 'DecisionTree', 'XGBoost', 'SVM', 'ShallowNN', 'DeepNN']
# algorithms = ['ShallowNN', 'DeepNN']

# for al in algorithms:
#     with open(f'updated_models_{al}_50_lasso.pkl', 'rb') as f:
#        models = pickle.load(f)
    
#     all_probs = []
#     for mdl in models:
#         feats = mdl.selected_features
#         Xm = X_vote_data[feats]
#         p = mdl.predict_proba(Xm)[:,1]
#         all_probs.append(p)

#     probs_arr = np.vstack(all_probs)
#     print(probs_arr)
#     ensemble_prob = probs_arr.mean(axis=0)

#     majority_predictions, agreement_percentages = majority_vote(models, X_vote_data)
    
#     fpr1, tpr1, _ = roc_curve(y_vote_data, ensemble_prob)
#     fpr2, tpr2, _ = roc_curve(y_vote_data, majority_predictions)

#     plt.figure(figsize=(7,6))
#     plt.plot(fpr1, tpr1, lw=2, label=f"Majority-vote ensemble (Probability)\nAUC = {auc(fpr1, tpr1):.4f}")
#     plt.plot(fpr2, tpr2, lw=2, label=f"Majority-vote ensemble (Predictions)\nAUC = {auc(fpr2, tpr2):.4f}")
#     plt.plot([0,1], [0,1], color='gray', linestyle='--', lw=1)
#     plt.xlim(0,1); plt.ylim(0,1.05)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(f'test_{al}.pdf')

#     print(f'{al} AUC plot is done')

# print('All done')


    
model_files = [
    ('updated_models_GLM_50_lasso.pkl', 'GLM'),
    ('updated_models_RandomForest_50_lasso.pkl', 'Random Forest'),
    ('updated_models_DecisionTree_50_lasso.pkl', 'Decision Tree'),
    ('updated_models_XGBoost_50_lasso.pkl', 'XGBoost'),
    ('updated_models_SVM_50_lasso.pkl', 'SVM'),
    ('updated_models_ShallowNN_50_lasso.pkl', 'Shallow Neural Network'),
    ('updated_models_DeepNN_50_lasso.pkl', 'Deep Neural Network')
]

plt.figure(figsize=(6,6))
colors = ['#3288bd','#d53e4f','#fc8d59','#99d594','#e78ac3','#e6f598','#fee08b']

for (fname,label),c in zip(model_files, colors):
    with open(fname, 'rb') as f:
        models = pickle.load(f)

    all_probs = []
    for mdl in models:
        feats = mdl.selected_features
        Xm = X_vote_data[feats]
        p = mdl.predict_proba(Xm)[:,1]
        all_probs.append(p)

    probs_arr = np.vstack(all_probs)
    ensemble_prob = probs_arr.mean(axis=0)

    fpr, tpr, _ = roc_curve(y_vote_data, ensemble_prob)
    plt.plot(fpr, tpr, color=c, lw=2, label=f"{label}")
    print(f'{label} AUC done')

plt.plot([0,1], [0,1], 'k--', lw=1, label="Chance (AUC=0.5)")
plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right", fontsize='small')
plt.tight_layout()
plt.savefig('auc7_update.pdf')

print('All done')

