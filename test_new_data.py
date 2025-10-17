# Test new data
# This file inculdes load updated_models.pkl and use it for testing new data.

import numpy as np
import pandas as pd
import pickle
import logging
import os
from sklearn.model_selection import train_test_split
from preprocessing import load_and_prepare_data
from ensemble import save_results_to_csv, majority_vote, competence_weighted_predictions, full_competence_weighted_predictions, display_vote_result
from breakdown import compute_and_save_competence_breakdown, compute_and_save_full_competence_breakdown


# Create a new log folder
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, 'test_new_data_log_cc.log'), level=logging.INFO)
# logging.basicConfig(filename=os.path.join(log_folder, 'test_new_data_log_matched.log'), level=logging.INFO)

def main():
    
    with open('updated_models_RandomForest_50_lasso.pkl', 'rb') as f:
       all_models = pickle.load(f)
    
    new_test_data = pd.read_csv('case_control_dataset_ordinal.csv')
    # new_test_data = pd.read_csv('matched_full_random_ordinal.csv')
    target = 'adhd'
    X_vote_data = new_test_data.drop(columns=[target])
    y_vote_data = new_test_data[target]

    models = all_models
    fs = 'lasso'
    n_partitions = 50
    algorithm = 'RandomForest'

    majority_predictions, agreement_percentages = majority_vote(models, X_vote_data)
    
    print(f'\nResults for majority vote ({fs}) with {n_partitions} partitions using {algorithm}: ')
    display_vote_result(majority_predictions, y_vote_data, n_partitions)

    # Print agreement and diversity info
    print(f'\nAgreement across models: {np.mean(agreement_percentages):.2f}%')


    # Competence predictions (selected)
    competence_predictions = competence_weighted_predictions(X_vote_data, models)
    print(f'\nResults for competence-based SELECTED ({fs}) with {n_partitions} partitions using {algorithm}: ')
    display_vote_result(competence_predictions, y_vote_data, n_partitions)

    # Competence predictions (Full)
    competence_predictions_full = full_competence_weighted_predictions(X_vote_data, models)
    print(f'\nResults for competence-based FULL ({fs}) with {n_partitions} partitions using {algorithm}: ')
    display_vote_result(competence_predictions_full, y_vote_data, n_partitions)


    # partition_breakdown = {}
    # for n_partitions in partition_settings:
    #     models = all_models[(n_partitions, algorithm, fs)]
    #     breakdown_dict = compute_and_save_competence_breakdown(X_vote_data, models, output_folder=f'competence_breakdown/cc_new_data/{fs}/selected')
    #     partition_breakdown[n_partitions] = breakdown_dict
    
    # binary_vars = new_test_data.columns[:-3].tolist()
    # continuous_vars = new_test_data.columns[-3:-1].tolist()   #['Gest_age','previous_absence']
    
    # full_features = binary_vars + continuous_vars
    # X_vote_data_full = X_vote_data[full_features]
    # partition_breakdown_full = {}
    # for n_partitions in partition_settings:
    #     models = all_models[(n_partitions, algorithm, fs)]
    #     breakdown_dict_full = compute_and_save_full_competence_breakdown(X_vote_data_full, models, output_folder=f'full_competence_breakdown/cc_new_data/{fs}/full')
    #     partition_breakdown_full[n_partitions] = breakdown_dict_full

if __name__ == "__main__":
    main()
