# Main
# Main process: reading data, training models, validation, model saving, prediction, and outputting competence brekdowns

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from joblib import Parallel, delayed
import multiprocessing
import time
import logging
import os
import matplotlib.pyplot as plt
import pickle
from itertools import product
from preprocessing import load_and_prepare_data
from model_training import hyperSMUMs, validate_ensemble_models
from ensemble import save_results_to_csv, majority_vote, competence_weighted_predictions, full_competence_weighted_predictions, display_vote_result
from breakdown import compute_and_save_competence_breakdown, compute_and_save_full_competence_breakdown

raw_data_path = "dataset_3_ordinal(no_feature_selection).csv"

# Create a new log folder
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, 'all_model_log.log'), level=logging.INFO)


# Main exection

# --------------- Training stage ---------------- #
# obtain ensemble model based on balanced data    #
target = 'adhd'
nominal_vars = ["pupil_ethnicity", "MODE_DEL"]
ordinal_vars = ['AGECAT', 'Dep_quintile', 'MATERNAL_AGECAT', 'Parity', 'bweight_centile', 'fiveminapgar_cat', 'previous_exclusion1']
continuous_vars = ['Gest_age','previous_absence']

df = load_and_prepare_data(raw_data_path, ordinal_vars, continuous_vars)

binary_vars = [c for c in df.columns[:-1] if c not in ordinal_vars + nominal_vars + continuous_vars]
print('Binary Columns in Data:', binary_vars)
print('Continuous Columns in Data:', continuous_vars)
print('Nominal:', nominal_vars)
print('Ordinal:', ordinal_vars)

np.random.seed(123)
hyperSMUMs_data_2, majority_vote_data_2 = train_test_split(df, test_size=0.2, stratify=df[target], random_state=123)
hyperSMUMs_train_data, val_data = train_test_split(hyperSMUMs_data_2, test_size=0.125, stratify=hyperSMUMs_data_2[target], random_state=123)
print("Train-Val data shape (hyperSMUMs_data): ", hyperSMUMs_data_2.shape)
print("Test data shape (majority_vote_data): ", majority_vote_data_2.shape)
print("Train data shape (hyperSMUMs_train_data): ", hyperSMUMs_train_data.shape)
print("Valid data shape (val_data): ", val_data.shape)

# Test different n partition settings (20 to 50)
partition_settings = [20, 30, 40, 50]

# algorithm = 'GLM'
algorithm = 'RandomForest'
# algorithm = 'ShallowNN'
# algorithm = 'XGBoost'
# algorithm = 'DecisionTree'
# algorithm = 'DeepNN'
# algorithm = 'SVM'


# Save all models, size info, and metric info for different setting
all_models = {}
all_size_info = {}
all_metrics_info = {} 

feature_selectors = ['lasso', 'mi', 'rf']

for fs in feature_selectors:
    print(f"\n===  FEATURE SELECTOR: {fs} ===")
    for n_partitions in partition_settings:
        print(f'\n>>>[{algorithm}] Partitions={n_partitions} Feature_selector={fs}')

        start_time = time.time()
        models, size_info_df, metrics_info_df = hyperSMUMs(hyperSMUMs_train_data, target, n_partitions, binary_vars, nominal_vars, ordinal_vars, continuous_vars, algorithm=algorithm, feature_selector=fs)
        end_time = time.time()
        elapsed_time = (end_time - start_time)/3600
        
        print(f'Time taken for {n_partitions} partitions: {elapsed_time:.2f} hours')

        # Store models, size information, and metrics seperately for each partition size
        all_models[(n_partitions, algorithm, fs)] = models
        all_size_info[(n_partitions, algorithm, fs)] = size_info_df
        all_metrics_info[(n_partitions, algorithm, fs)] = metrics_info_df

        print(f'\nSize Information for {n_partitions} partitions using {algorithm}:')
        print(size_info_df)

        print(f'\nMetrics Information for {n_partitions} partitions using {algorithm}:')
        print(metrics_info_df)

        root = os.path.join(os.getcwd(), algorithm)
        os.makedirs(root, exist_ok=True)

        # Save the size info and metrics info to separte CSV files
        csv_folder = os.path.join(root, fs, str(n_partitions))
        os.makedirs(csv_folder, exist_ok=True)
        size_output_filename = os.path.join(csv_folder, f'size_{algorithm}_{n_partitions}_{fs}.csv')
        metrics_output_filename = os.path.join(csv_folder, f'metrics_{algorithm}_{n_partitions}_{fs}.csv')
        save_results_to_csv(size_info_df, size_output_filename)
        save_results_to_csv(metrics_info_df, metrics_output_filename)
        print(f'\nSize information saved to {size_output_filename}')
        print(f'\nMetrics saved to {metrics_output_filename}')

        print(f'\nStored {len(models)} models in a variable for majority vote.')


        # --------------------------- Validation stage ---------------------------------- #
        # read the validation data that maintains the original imbalance distribution,    #
        # and perform hyper-parameter tuning on the ensemble model obtained from training #
        if algorithm == 'GLM':
            candidate_grid = {
                'C': np.logspace(-2, 2, 5),
                'l1_ratio': np.arange(0, 1.1, 0.1)
            }
        elif algorithm == 'RandomForest':
            candidate_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif algorithm == 'DecisionTree':
            candidate_grid = {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif algorithm == 'ShallowNN':
            candidate_grid = {
                'hidden_layer_sizes':[(64,), (128,), (256,)],
                'epochs': [100, 200],
                'batch_size': [16, 32]
            }
        elif algorithm == 'DeepNN':
            candidate_grid = {
                'hidden_layer_sizes':[(256,128), (128,64), (256,128,64)],
                'epochs': [100, 200],
                'batch_size': [16, 32]
            }
        elif algorithm == 'XGBoost':
            candidate_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10]
            }
        elif algorithm == 'SVM':
            candidate_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        else:
            raise ValueError("Unsupported algorithm")
        
        print(f'\nValidating ensemble for {n_partitions} partitions using {algorithm} on imbalanced validation data ...')
        updated_models = validate_ensemble_models(models, val_data, target, candidate_grid, algorithm)
        all_models[(n_partitions, algorithm, fs)] = updated_models

        model_filename = os.path.join(root, fs, str(n_partitions), f"updated_models_{algorithm}_{n_partitions}_{fs}.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(updated_models, f)
        print(f"Updated models have been saved to {model_filename}")

    
        # ---------------- Testing stage ----------------- #
        # Majority vote and competence-based weighted vote #

        X_vote_data = majority_vote_data_2.drop(columns=[target])
        y_vote_data = majority_vote_data_2[target]

        majority_predictions, agreement_percentages = majority_vote(updated_models, X_vote_data)
        print(f'\n---Majority Vote ({algorithm}, {n_partitions}, {fs}) ---')
        display_vote_result(majority_predictions, y_vote_data, len(updated_models))

        # Competence predictions (selected)
        competence_predictions = competence_weighted_predictions(X_vote_data, updated_models)
        print(f'\n---Competence-based SELECTED ({algorithm}, {n_partitions}, {fs}) ---')
        display_vote_result(competence_predictions, y_vote_data, len(updated_models))

        # Competence predictions (Full)
        competence_predictions_full = full_competence_weighted_predictions(X_vote_data, updated_models)
        print(f'\n---Competence-based FULL ({algorithm}, {n_partitions}, {fs}) ---')
        display_vote_result(competence_predictions_full, y_vote_data, len(updated_models))

        # # Breakdown saving
        # cb_folder = os.path.join(root, fs, str(n_partitions), "competence_breakdown")
        # os.makedirs(cb_folder, exist_ok=True)
        # compute_and_save_competence_breakdown(X_vote_data, updated_models, output_folder=cb_folder)

        # fcb_folder = os.path.join(root, fs, str(n_partitions), "competence_breakdown_full")
        # os.makedirs(fcb_folder, exist_ok=True)
        # compute_and_save_full_competence_breakdown(X_vote_data[binary_vars + continuous_vars], updated_models, output_folder=fcb_folder)

print("=== All Done, results are under the folder:", algorithm, "===")    

