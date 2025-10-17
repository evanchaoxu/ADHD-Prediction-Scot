# Ensemble
# This file including functions for vote prediction, result display, etc.

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import mode
import logging
from metrics import calculate_competence


# Get competence_weighted_predictions
def competence_weighted_predictions(X_test, models):
    num_samples = X_test.shape[0]
    num_models = len(models)
    competence_scores = np.zeros((num_samples, num_models))
    prediction_probs = np.zeros((num_samples, num_models))

    for model_index, model in enumerate(models):
        features = model.selected_features
        new_binary_vars = model.new_binary_vars
        new_continuous_vars = model.new_continuous_vars
        new_nominal_vars = model.new_nominal_vars
        new_ordinal_vars = model.new_ordinal_vars
        X_test_model = X_test[features]

        print(f"[Model {model_index}] X_test_model.shape = {X_test_model.shape}")
        logging.info(f"[Model {model_index}] X_test_model.shape = {X_test_model.shape}")

        for i in range(num_samples):
            competence_scores[i, model_index] = calculate_competence(
                X_test_model.iloc[i], model.model_data, new_binary_vars, new_nominal_vars, new_ordinal_vars, new_continuous_vars
            )
        
        prob_array = model.predict_proba(X_test_model)
        print(f"[Model {model_index}] predict_proba shape = {prob_array.shape}")
        logging.info(f"[Model {model_index}] predict_proba shape = {prob_array.shape}")

        # if shape is (n_samples, 1), the prob_array[:,1] will show error
        if prob_array.shape[1] == 2:
            prediction_probs[:, model_index] = prob_array[:, 1]
        else:
            logging.warning(f"[Model {model_index}] Only one class detected! Using prob_array[:,0] as fallback = {prob_array[:, 0]}")
            prediction_probs[:, model_index] = prob_array[:, 0]
        
    print(f"competence_scores shape = {competence_scores.shape}")
    print(f"prediction_probs shape = {prediction_probs.shape}")
    
    # Compute weighted predictions for each sample
    weighted_sums = np.sum(competence_scores * prediction_probs, axis=1)
    total_weights = np.sum(competence_scores, axis=1)
    weighted_avg = weighted_sums / total_weights

    final_predictions = (weighted_avg > 0.5).astype(int)

    logging.info(f'Final predictions: {final_predictions}')

    return final_predictions


# Get competence_weighted_predictions
def full_competence_weighted_predictions(X_test, models):
    num_samples = X_test.shape[0]
    num_models = len(models)
    full_competence_scores = np.zeros((num_samples, num_models))
    prediction_probs = np.zeros((num_samples, num_models))

    for model_index, model in enumerate(models):
        full_features = model.full_binary_vars + model.full_nominal_vars + model.full_ordinal_vars + model.full_continuous_vars
        X_test_full = X_test[full_features]

        print(f"[Model {model_index}] X_test_full.shape = {X_test_full.shape}")
        logging.info(f"[Model {model_index}] X_test_full.shape = {X_test_full.shape}")

        for i in range(num_samples):
            full_competence_scores[i, model_index] = calculate_competence(
                X_test_full.iloc[i], model.full_model_data, model.full_binary_vars, model.full_nominal_vars, model.full_ordinal_vars, model.full_continuous_vars
            )
        
        # Model predicitons still use data after feature selection
        features = model.selected_features
        X_test_model = X_test[features]
        prob_array = model.predict_proba(X_test_model)
        print(f"[Model {model_index}] predict_proba shape = {prob_array.shape}")
        logging.info(f"[Model {model_index}] predict_proba shape = {prob_array.shape}")

        # if shape is (n_samples, 1), the prob_array[:,1] will show error
        if prob_array.shape[1] == 2:
            prediction_probs[:, model_index] = prob_array[:, 1]
        else:
            logging.warning(f"[Model {model_index}] Only one class detected! Using prob_array[:,0] as fallback = {prob_array[:, 0]}")
            prediction_probs[:, model_index] = prob_array[:, 0]
        
    print(f"competence_scores shape = {full_competence_scores.shape}")
    print(f"prediction_probs shape = {prediction_probs.shape}")
    
    # Compute weighted predictions for each sample
    weighted_sums_full = np.sum(full_competence_scores * prediction_probs, axis=1)
    total_weights_full = np.sum(full_competence_scores, axis=1)
    weighted_avg_full = weighted_sums_full / total_weights_full

    final_predictions_full = (weighted_avg_full > 0.5).astype(int)

    logging.info(f'Final predictions: {final_predictions_full}')

    return final_predictions_full


# Function to save the results to CSV files
def save_results_to_csv(results, filename):
    results.to_csv(filename, index=False)

# Function to get predictions from a single model
def get_prediction(model, X_vote):
    features = model.selected_features
    X_vote_model = X_vote[features]
    return model.predict(X_vote_model)

# Function for Majority vote
def majority_vote(models, X_vote, n_jobs=None):

    logging.info(f"majority_vote: X_vote shape = {X_vote.shape}")

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1

    predictions = Parallel(n_jobs=n_jobs)(
        delayed(get_prediction)(model, X_vote) for model in models
    )

    predictions = np.vstack(predictions)

    majority_predictions, _ = mode(predictions, axis=0, keepdims=True)
    majority_predictions = majority_predictions.flatten()

    # Check agreement across models
    agreement_percentages = np.mean(predictions == majority_predictions, axis=0) * 100

    return majority_predictions, agreement_percentages

# Function to calculate and display result
def display_vote_result(majority_predictions, y_vote_data, num_classifiers):

    # Debug
    print('Type of y_vote_data: ', type(y_vote_data))
    print('Type of majority_predictions: ', type(majority_predictions))

    conf_matrix = confusion_matrix(y_vote_data, majority_predictions)

    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = accuracy_score(y_vote_data, majority_predictions)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
    f1 = f1_score(y_vote_data, majority_predictions)
    auc_value = roc_auc_score(y_vote_data, majority_predictions)

    print(f'\nResults for {num_classifiers} classifiers:')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('\nMetrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Sensitivity (Recall): {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'AUC: {auc_value:.4f}')

    logging.info(f'\nResults for {num_classifiers} classifiers:')
    logging.info('Confusion Matrix:')
    logging.info(conf_matrix)
    logging.info('\nMetrics:')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Sensitivity (Recall): {sensitivity:.4f}')
    logging.info(f'Specificity: {specificity:.4f}')
    logging.info(f'F1: {f1:.4f}')
    logging.info(f'AUC: {auc_value:.4f}')
