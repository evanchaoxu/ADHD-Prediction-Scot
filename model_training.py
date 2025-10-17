# Model Training
# This file includes functions for model training, partitioning, and validation tuning

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.utils import resample
from preprocessing import smotenc_data, feature_selection, mutual_info_selection, rf_importance_selection, make_preprocessor
from metrics import calculate_cohort_variable_distribution
from joblib import Parallel, delayed
import multiprocessing
from itertools import product


def create_shallow_nn(hidden_layer_sizes=(64,), input_shape=None):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))  
    # model.add(Dense(hidden_layer_sizes[0], activation='relu'))
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_deep_nn(hidden_layer_sizes=(256,128), input_shape=None):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    # model.add(Dense(hidden_layer_sizes[0], activation='relu'))
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Training process
# Define a function to train a model using CV and GS
def train_model(train_data, target, partition_idx, ordinal_vars, nominal_vars, continuous_vars, binary_vars, algorithm):
    """
    Train a using cross-validation and grid search for hyperparameter tuning on balanced data.
    
    Parameters:
    - train_data: The training dataset including features and target.
    - target: The response variable name (str).
    - partition_idx: The index of different partitions
    - algorithm: This including 'RandomForest', 'DecisionTree', 'ShallowNN', 'DeepNN', 'XGBoost', 'SVM'
    
    Return:
    - model: The trained model after grid search.
    """
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]

    pre = make_preprocessor(algorithm, ordinal_vars, nominal_vars, continuous_vars, binary_vars)

    X_train_pre = pre.fit_transform(X_train)
    n_input = X_train_pre.shape[1]

    # Define the parameter grids and models for grid search
    if algorithm == 'GLM':
        param_grid = {
            'model__C': np.logspace(-2, 2, 5),            # Regularization strength from 0.01 to 100 (inverse of lambda: 100 to 0.01)  
            'model__l1_ratio': np.arange(0, 1.1, 0.1)     # Mixing parameter - balances L1 and L2 penalties
        }
        model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000, random_state=partition_idx + 123)

    elif algorithm == 'RandomForest':
        param_grid = {
            'model__n_estimators': [100],
            'model__max_features': ['sqrt'],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
        model = RandomForestClassifier(random_state=partition_idx + 123)
    
    elif algorithm == 'DecisionTree':
        param_grid = {
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
        model = DecisionTreeClassifier(random_state=partition_idx + 123) 
    
    elif algorithm == 'ShallowNN':
        model = KerasClassifier(model=create_shallow_nn, hidden_layer_sizes=(64,), input_shape=n_input, epochs=100, batch_size=16, verbose=0)
        param_grid = {
            'model__hidden_layer_sizes':[(64,), (128,), (256,)],
            'model__epochs': [50],
            'model__batch_size': [16, 32]
        }
        
    elif algorithm == 'DeepNN':
        model = KerasClassifier(model=create_deep_nn, hidden_layer_sizes=(256,128), input_shape=n_input, epochs=100, batch_size=16, verbose=0)
        param_grid = {
            'model__hidden_layer_sizes':[(256, 128), (128, 64), (256, 128, 64)],
            'model__epochs': [50, 100],
            'model__batch_size': [16, 32]
        }
        
    elif algorithm == 'XGBoost':
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 6, 10]
        }
        model = XGBClassifier(eval_metric='logloss', random_state=partition_idx + 123)
    
    elif algorithm == 'SVM':
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        }
        model = SVC(probability=True, random_state=partition_idx + 123)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    

    # Create a pipeline with preprocessor and model
    pipeline = Pipeline([
        ('pre', pre),
        ('model', model)
    ])

    # Define the CV strategy (5-fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=partition_idx + 123)
    
    # Using GridSearchCV for hyperparameter tunig with CV    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                               cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Save the training data of this partition for subsequent validation and tuning
    grid_search.train_data = train_data.copy()
    
    return grid_search

def process_partition(i, split_factor, minority_data, target, ordinal_vars, nominal_vars, binary_vars, continuous_vars, n_partitions, algorithm='RandomForest', feature_selector='lasso', save_smote_csv=False):

    # Get current partition
    # Select the first partition
    current_partition = split_factor[i]

    # Perform undersampling on the majority data using the rand_undersample function
    # Separate the features from the majority data (excluding the adhd column)
    X_majority = current_partition.drop(columns=[target])
    y_majority = current_partition[target]

    # print(f"[Partition {i}] Before undersampling: majority class distribution = {y_majority.value_counts().to_dict()}")

    # Randomised Majority Undersampling
    # Instead of always aiming for the same number of majority samples, we will randomly pick the number
    # Randomise the proportion of majority samples to be undersampled, between 50%-90%
    undersample_ratio = np.random.uniform(0.5, 0.9)
    n_samples_partition = int(undersample_ratio * len(X_majority))
    n_samples_partition = min(len(X_majority), n_samples_partition)
    X_majority_undersampled, y_majority_undersampled = resample(
        X_majority, y_majority, n_samples=n_samples_partition, random_state=np.random.randint(1000)
    )

    # print(f"[Partition {i}] After undersampling: majority class distribution = {y_majority_undersampled.value_counts().to_dict()}")

    # Combine the undersampled majority data with minority data
    X_combined = pd.concat([X_majority_undersampled, minority_data.drop(columns=[target])], ignore_index=True)
    y_combined = pd.concat([y_majority_undersampled, minority_data[target]], ignore_index=True)

    # print(f"[Partition {i}] Combined distribution (before SMOTE): {y_combined.value_counts().to_dict()}")

    # Apply SMOTE to the combined data
    categorical_vars = binary_vars + ordinal_vars + nominal_vars
    smote_resampled = smotenc_data(X_combined, y_combined, target, categorical_vars=categorical_vars)
    X_resampled = smote_resampled.drop(columns=[target])
    y_resampled = smote_resampled[target]

    # print(f"[Partition {i}] After SMOTE: {y_resampled.value_counts().to_dict()}")

    # Save SMOTE resampled data only for the 1st partition --for data inspection after SMOTE
    if save_smote_csv and i == 0:
        smote_filename = f'smote_resampled_P1_{n_partitions}.csv'
        smote_resampled.to_csv(smote_filename, index=False)


    # Split the data into training and test sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled,
                                                        y_resampled,
                                                        test_size=0.2,
                                                        stratify=y_resampled,
                                                        random_state=123)
    train_data = X_train.copy()
    train_data[target] = y_train

    # Multi-Strategy Feature Selection
    if feature_selector == 'lasso':
        selected_features = feature_selection(X_train, y_train, continuous_vars)
    elif feature_selector == 'mi':
        selected_features = mutual_info_selection(X_train, y_train, continuous_vars, k_values=[15,20,25,30,35,40], cv=5, scoring='accuracy', return_scores=False)
    elif feature_selector == 'rf':
        selected_features = rf_importance_selection(X_train, y_train, continuous_vars, thresholds=[0.005, 0.01, 0.02, 0.05, 0.1], max_features_list=[10,15,20,25,30,35,40], cv=5, scoring='accuracy', return_scores=False)
    else:
        raise ValueError("Unsupported feature_selector")


    new_continuous_vars = [col for col in continuous_vars if col in selected_features]
    new_ordinal_vars = [col for col in ordinal_vars if col in selected_features]
    new_nominal_vars = [col for col in nominal_vars if col in selected_features]
    new_binary_vars = [col for col in binary_vars if col in selected_features]

    train_data = train_data[selected_features + [target]]
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    print(f"[Partition {i}] X_train shape after feature selection: {X_train.shape}")
    print(f"[Partition {i}] X_test shape after feature selection: {X_test.shape}")
    logging.info(f"[Partition {i}] X_train shape after feature selection: {X_train.shape}")
    logging.info(f"[Partition {i}] X_test shape after feature selection: {X_test.shape}")

    # Calculate the statistical information used by the model for competence calculations
    # Using the original data (without SMOTE and undersampling)
    original_data = pd.concat([current_partition[selected_features], minority_data[selected_features]], ignore_index=True)
    model_data_stats = calculate_cohort_variable_distribution(original_data, new_continuous_vars, new_binary_vars, new_ordinal_vars, new_nominal_vars)

    # Save the full priginal training data distribution (no oversampling/undersampling, no feature selection)
    # Using all original variables
    full_vars = binary_vars + nominal_vars + ordinal_vars + continuous_vars 
    full_original_data = pd.concat([current_partition[full_vars], minority_data[full_vars]], ignore_index=True)
    full_model_data = calculate_cohort_variable_distribution(full_original_data, continuous_vars, binary_vars, ordinal_vars, nominal_vars)

    # Train the model
    model = train_model(train_data, target, partition_idx=i, ordinal_vars=new_ordinal_vars, nominal_vars=new_nominal_vars, continuous_vars=new_continuous_vars, binary_vars=new_binary_vars, algorithm=algorithm)
    model.train_data = train_data.copy()  # save training data for validation tuning
    model.selected_features = selected_features
    model.new_continuous_vars = new_continuous_vars
    model.new_binary_vars = new_binary_vars
    model.new_nominal_vars = new_nominal_vars
    model.new_ordinal_vars = new_ordinal_vars
    model.model_data = model_data_stats

    # Save complete training data distribution information to the model
    model.full_model_data = full_model_data
    model.full_binary_vars = binary_vars.copy()
    model.full_continuous_vars = continuous_vars.copy()
    model.full_nominal_vars = nominal_vars.copy()
    model.full_ordinal_vars = ordinal_vars.copy()

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_prob)

    # Record sizes
    size_info = {
        'Partition': i + 1,
        'Majority_before': len(X_majority),
        'Majority_after': len(X_majority_undersampled),
        'Minority_before': len(minority_data),
        'Minority_after': len(y_resampled[y_resampled == 1]),
        'Combined': len(X_resampled)
    }

    # Record metrics
    metrics_info = {
        'Partition': i + 1,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1_Score': f1,
        'AUC': auc_value
    }


    # Return results for this partition
    return model, size_info, metrics_info


def hyperSMUMs(data, target, n_partitions, binary_vars, nominal_vars, ordinal_vars, continuous_vars, algorithm='RandomForest', feature_selector='lasso'):
    """
    Main hyper function that processes all partitions and returns results for each.
    
    Parameters:
    - data: The entire dataset including features and target.
    - target: The response variable/target column name (str).
    - n_partitions: Number of partitions to split the majority class into.
    
    Returns:
    - results: A dictionary containing results from all partitions.
    """  
    #Split the whole dataset into 'majority' and 'minority'
    majority_data = data[data[target] == 0]
    minority_data = data[data[target] == 1]

    # Shuffle the majority data
    majority_data_shuffle = majority_data.sample(frac=1, random_state=123).reset_index(drop=True)

    # Split the majority data into n_partitions
    split_factor = np.array_split(majority_data_shuffle, n_partitions)
    
    # Use parallel processing to process partitions
    num_cores = multiprocessing.cpu_count() - 1   # Leave one core free
    
    # Process partitions in parallel
    results = Parallel(n_jobs=num_cores)(
        delayed(process_partition)(i, split_factor, minority_data, target, ordinal_vars, nominal_vars, binary_vars, continuous_vars, n_partitions, algorithm=algorithm, feature_selector=feature_selector, save_smote_csv=True) for i in range(n_partitions)
    )

    # Unzip the results into models, size info, and metric info
    models, size_info_list, metrics_info_list = zip(*results)
    return list(models), pd.DataFrame(size_info_list), pd.DataFrame(metrics_info_list)


# Hyperparametric tuning of each algorithmic model using imbalanced validation data
def tune_model_hyperparameters(train_data, validation_data, target, new_ordinal_vars, new_nominal_vars, new_continuous_vars, new_binary_vars, candidate_grid, partition_idx, algorithm, selected_features):
    """
    For models with a signle partition, tune the hyperparameters on balanced train_data using
    imbalanced validation data. Return the best combination of parameters.
    """

    X_train = train_data[selected_features]
    y_train = train_data[target]
    X_val = validation_data[selected_features]
    y_val = validation_data[target]

    pre = make_preprocessor(algorithm, new_ordinal_vars, new_nominal_vars, new_continuous_vars, new_binary_vars)
    X_train_pre = pre.fit_transform(X_train)
    n_input = X_train_pre.shape[1]

    print(f"[Partition {partition_idx}] tune_model_hyperparameters: y_train distribution = {y_train.value_counts().to_dict()}")
    print(f"[Partition {partition_idx}] tune_model_hyperparameters: y_val distribution = {y_val.value_counts().to_dict()}")
    print(f"[Partition {partition_idx}] X_train shape = {X_train_pre.shape}, X_val shape = {X_val.shape}")
    logging.info(f"[Partition {partition_idx}] tune_model_hyperparameters: y_train distribution = {y_train.value_counts().to_dict()}")
    logging.info(f"[Partition {partition_idx}] tune_model_hyperparameters: y_val distribution = {y_val.value_counts().to_dict()}")
    logging.info(f"[Partition {partition_idx}] X_train shape = {X_train_pre.shape}, X_val shape = {X_val.shape}")


    best_score = -np.inf
    best_pipeline = None
    keys = list(candidate_grid.keys())

    for values in product(*[candidate_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        if algorithm == 'GLM':
            model_instance = LogisticRegression(
                C = params['C'],
                l1_ratio = params['l1_ratio'],
                penalty='elasticnet', 
                solver='saga', 
                max_iter=5000, 
                random_state=partition_idx + 123
            )
        elif algorithm == 'RandomForest':
            model_instance = RandomForestClassifier(
                n_estimators = params['n_estimators'],
                max_features = 'sqrt',
                max_depth = params['max_depth'],
                min_samples_split = params['min_samples_split'],
                min_samples_leaf = params['min_samples_leaf'],
                random_state = partition_idx + 123
            )
        elif algorithm == 'DecisionTree':
            model_instance = DecisionTreeClassifier(
                max_depth = params['max_depth'],
                min_samples_split = params['min_samples_split'],
                min_samples_leaf = params['min_samples_leaf'],
                random_state = partition_idx + 123
            )
        elif algorithm == 'ShallowNN':
            model_instance = KerasClassifier(model=create_shallow_nn, hidden_layer_sizes=params['hidden_layer_sizes'], input_shape=n_input, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        elif algorithm == 'DeepNN':
            model_instance = KerasClassifier(model=create_deep_nn, hidden_layer_sizes=params['hidden_layer_sizes'], input_shape=n_input, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        elif algorithm == 'XGBoost':
            model_instance = XGBClassifier(
                n_estimators = params['n_estimators'],
                learning_rate = params['learning_rate'],
                max_depth = params['max_depth'],
                eval_metric = 'logloss',
                random_state = partition_idx + 123
            )
        elif algorithm == 'SVM':
            model_instance = SVC(
                C = params['C'],
                kernel = params['kernel'],
                probability = True,
                random_state = partition_idx + 123
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        

        pipeline = Pipeline([
            ('pre', pre),
            ('model', model_instance)
        ])

        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        cm = confusion_matrix(y_val, y_val_pred)
        tn,fp,fn,tp = cm.ravel()
        current_accuracy = accuracy_score(y_val, y_val_pred)
        current_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        current_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        current_score = f1_score(y_val, y_val_pred)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        current_auc = roc_auc_score(y_val, y_val_prob)

        print(f"Partition {partition_idx}, Params: {params}, "
              f"F1 {current_score:.4f}, Accuracy: {current_accuracy:.4f}, AUC: {current_auc:.4f}, "
              f"Sensitivity {current_sensitivity:.4f}, Specificity: {current_specificity:.4f}. ")

        if current_score > best_score:
            best_score = current_score
            best_pipeline = pipeline
    print(f"Partition {partition_idx}: Best {algorithm} model on validation with F1: {best_score:.4f}")
    return best_pipeline


def validate_ensemble_models(models_info, validation_data, target, candidate_grid, algorithm):
    """
    Hyper-parameter tuning is performed for each partition's model using the saved train_data and 
    imbalance validation data, and the model is retained with the optimal parameters, 
    returning an updated list of models.
    """
    updated_models = []
    for partition_idx, model in enumerate(models_info):
        train_data = model.train_data
        selected_features = model.selected_features
        new_continuous_vars = model.new_continuous_vars
        new_binary_vars = model.new_binary_vars
        new_nominal_vars = model.new_nominal_vars
        new_ordinal_vars = model.new_ordinal_vars

        # Debug: Print a comparison of the training and validation data columns for the current partition
        train_cols = set(selected_features)
        val_cols = set(validation_data.columns)
        missing_in_val = train_cols - val_cols
        if missing_in_val:
            logging.info(f"[DEBUG] Partition {partition_idx}: Columns in train_data but missing in validation data: {missing_in_val}")
        
        best_pipeline = tune_model_hyperparameters(train_data, validation_data, target, new_ordinal_vars, new_nominal_vars, new_continuous_vars, new_binary_vars, candidate_grid, partition_idx, algorithm, selected_features)
        best_pipeline.model_data = model.model_data
        best_pipeline.selected_features = model.selected_features
        best_pipeline.new_continuous_vars = model.new_continuous_vars
        best_pipeline.new_binary_vars = model.new_binary_vars
        best_pipeline.new_nominal_vars = model.new_nominal_vars
        best_pipeline.new_ordinal_vars = model.new_ordinal_vars

        best_pipeline.full_model_data = model.full_model_data
        best_pipeline.full_continuous_vars = model.full_continuous_vars
        best_pipeline.full_binary_vars = model.full_binary_vars
        best_pipeline.full_nominal_vars = model.full_nominal_vars
        best_pipeline.full_ordinal_vars = model.full_ordinal_vars
        updated_models.append(best_pipeline)
    return updated_models


