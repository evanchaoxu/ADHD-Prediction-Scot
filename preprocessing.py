# Preprocessing
# This file contains functions related to data preprocessing, e.g. feature selection, SMOTE oversampling

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline

ordinal_order = {
    'AGECAT': [1,2,3],
    'Dep_quintile': [5,4,3,2,1],
    'MATERNAL_AGECAT': [1,2,3,4],
    'Parity': [0,1,2],
    'bweight_centile': [1,2,3,4,5,6,7],
    'fiveminapgar_cat': [1,2,3],
    'previous_exclusion1': [0,1,2]
}


# Input data and dtype settings
def load_and_prepare_data(path, ordinal_vars, continuous_vars):
    df = pd.read_csv(path)
    df = df.apply(lambda x: x.astype('category') if x.name not in continuous_vars else x)

    for col in ordinal_vars:
        order = ordinal_order.get(col)
        if order is None:
            raise KeyError(f"Please define {col} in ordinal_order.")
        df[col] = df[col].cat.set_categories(order, ordered = True)
    return df  


def smotenc_data(X, y, target, categorical_vars, k_neighbors=np.random.randint(3, 10), random_state=np.random.randint(1000)): 
    """
    SMOTE method: oversampling the minority data. SMOTE cannot operate with a single target class. 
    
    Parameters:
    - X: Feature matrix (should be combined majority and minority features).
    - y: Target vector (should be combined majority and minority target).
    - k_neighbors: Number of nearest neighbors for SMOTE.
    - random_state: Seed for reproducibility.
    
    Returns:
    - update_smote_data: DataFrame containing the resampled feature matrix and target.
    """  
    cat_idx = [X.columns.get_loc(col) for col in categorical_vars]
    smote_nc = SMOTENC(categorical_features=cat_idx, k_neighbors = k_neighbors, random_state = random_state) 
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    update_smote_data = pd.DataFrame(X_resampled, columns=X.columns)
    update_smote_data[target] = y_resampled
    return update_smote_data


# Feature Selection based on LASSO
def feature_selection(X, y, continuous_vars):
    """
    Feature selection was performed on each subset using LassoCV.
    Only continuous variables are normalised, binary variables are left unchanged.
    Returns a list of selected features.
    """
    feature_order = [col for col in X.columns if col not in continuous_vars] + continuous_vars
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), continuous_vars)],
        remainder='passthrough'
    )
    lasso = LassoCV(cv=5, random_state=123)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('lasso', lasso)])
    pipeline.fit(X[feature_order], y)
    coefs = pipeline.named_steps['lasso'].coef_
    selected_features = [feature for feature, coef in zip(feature_order, coefs) if abs(coef) > 1e-4]
    if not selected_features:
        selected_features = feature_order
    return selected_features

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

def mutual_info_selection(X, y, continuous_vars, k_values=None, cv=5, scoring='accuracy', return_scores=False):
    """
    SelectKBest + mutual_info_classif and automatically search for the optimal k with cross-validation.

    Parameters:
    - X, y: Features and labels
    - k_values: list of k to search, e.g. [30,35,40,45,50,55,60]; if None, just do k='all'.
    - cv: cross validation folds
    - scoring: evaluation metrics
    - return_scores: if or not to return scores for each each feature information gain DataFrame
    
    Returns:
    - selected features or (selected features, scores_df).
    """
    features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), continuous_vars)],
        remainder='passthrough'
    )
    X_pre = preprocessor.fit_transform(X)
    X_pre = pd.DataFrame(X_pre, columns=features, index=X.index)

    if k_values is None:
        k_values = ['all']
    
    best_score = -np.inf
    best_k = None

    for k in k_values:
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_sel = selector.fit_transform(X_pre, y)
        clf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
        scores = cross_val_score(clf, X_sel, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score, best_k = mean_score, k
    
    final_sel = SelectKBest(score_func=mutual_info_classif, k=best_k)
    final_sel.fit(X_pre, y)
    support = final_sel.get_support()
    selected = [feature for feature, keep in zip(features, support) if keep]

    # Score Table (in descending order of MI)
    scores = mutual_info_classif(X_pre, y)
    scores_df = pd.DataFrame({'feature':features, 'score':scores}).sort_values('score', ascending=False).reset_index(drop=True)
    if return_scores:
        return selected, scores_df
    return selected

def rf_importance_selection(X, y, continuous_vars, thresholds=None, max_features_list=None, cv=5, scoring='accuracy', return_scores=False):
    """
    Based on random forest feature_importances_ and automatically searches for threshold or top-N.

    Parameters:
    - thresholds: list of importance threshold to try.
    - max_features_list: list of tried top-Ns
    - cv, scoring: CV folds and scoring metrics
    - return_scores: whether to return feature-importance DataFrame
    """
    features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), continuous_vars)],
        remainder='passthrough'
    )
    X_pre = preprocessor.fit_transform(X)
    X_pre = pd.DataFrame(X_pre, columns=features, index=X.index)

    rf_full = RandomForestClassifier(n_estimators=200, random_state=123, n_jobs=-1)
    rf_full.fit(X_pre, y)
    imp = rf_full.feature_importances_
    
    scores_df = pd.DataFrame({'feature': features, 'importance': imp}).sort_values('importance', ascending=False).reset_index(drop=True)

    if thresholds is None and max_features_list is None:
        selected = scores_df.loc[scores_df['importance'] > 0.01, 'feature'].tolist()
    else:
        best_score, best_sel = -np.inf, None
        if thresholds is not None:
            for thr in thresholds:
                sel = scores_df.loc[scores_df['importance'] > thr, 'feature'].tolist()
                if not sel: continue 
                clf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
                scores = cross_val_score(clf, X_pre[sel], y, cv=cv, scoring=scoring, n_jobs=-1)
                if scores.mean() > best_score:
                    best_score, best_sel = scores.mean(), sel
        if max_features_list is not None:
            for N in max_features_list:
                sel = scores_df['feature'].iloc[:N].tolist()
                clf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
                scores = cross_val_score(clf, X_pre[sel], y, cv=cv, scoring=scoring, n_jobs=-1)
                if scores.mean() > best_score:
                    best_score, best_sel = scores.mean(), sel
        selected = best_sel or scores_df['feature'].iloc[:30].tolist

    if return_scores:
        return selected, scores_df
    return selected

# Setting ColumnTransformer
def make_preprocessor(model_type, ordinal_vars, nominal_vars, continuous_vars, binary_vars):
    """
    Automatically construct preprocessing flow:
        - GLM/DT/RF/XGB/SVM: Ordinal-OneHot(nominal)-Scale(continuous)-passthrough(binary)
        - ShallowNN/DeepNN: OneHot(ordinal+nominal)-Scale(continuous)-passthrough(binary)
    """
    steps = []
    is_nn = model_type.upper() in ("SHALLOWNN", "DEEPNN")

    # Ordinal
    if ordinal_vars:
        if is_nn:
            steps.append(("oh_ord", OneHotEncoder(handle_unknown='ignore'), ordinal_vars))
        else:
            steps.append(("ord", OrdinalEncoder(), ordinal_vars))
        # steps.append(("ord", OrdinalEncoder(), ordinal_vars))

    # Nominal
    if nominal_vars:
        steps.append(("oh_nom", OneHotEncoder(handle_unknown='ignore'), nominal_vars))
    
    # Continuous
    if continuous_vars:
        steps.append(("scale", StandardScaler(), continuous_vars))
    
    pre = ColumnTransformer(transformers=steps, remainder='passthrough')

    return pre
