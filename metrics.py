# Metrics
# This file contains statistical information and competence score related functions used to calcualte the model.

import numpy as np
import pandas as pd
from scipy.stats import mode

def calculate_cohort_variable_distribution(X_data, continuous_vars, binary_vars, ordinal_vars, nominal_vars):
    """
    Compute distribution statistics for four types variables:
        - binary: mode + p0/p1
        - nominal: mode + probability per category
        - ordinal: median + 25%/75% (treated as continuous)
        - continuous: median + 25%/75%
    """

    cohort_distribution = {
        'binary_modes': {},   # Store modes for binary variables
        'binary_info': {},    # Store probabilities for binary variables (for information gain calculation)
        'nominal_modes': {},
        'nominal_info': {},
        'ordinal_medians': {},
        'ordinal_quartiles': {}, 
        'continuous_medians': {},
        'continuous_quartiles': {}   
    }

    for var in binary_vars:
        # Calculate the mode for binary varibales
        cohort_distribution['binary_modes'][var] = X_data[var].mode()[0]

        count_0 = (X_data[var] == 0).sum()
        count_1 = (X_data[var] == 1).sum()
        total_count = len(X_data[var])

        p0 = count_0 / total_count
        p1 = count_1 / total_count

        cohort_distribution['binary_info'][var] = {
            'p0': p0,
            'p1': p1
        }
    
    for var in nominal_vars:
        cohort_distribution['nominal_modes'][var] = X_data[var].mode()[0]
        cohort_distribution['nominal_info'][var] = X_data[var].value_counts(normalize=True).to_dict()
    
    for var in ordinal_vars:
        arr = X_data[var].cat.codes.values
        cohort_distribution['ordinal_medians'][var] = np.median(arr)
        cohort_distribution['ordinal_quartiles'][var] = {
            'lower_quartile': np.percentile(arr, 25),
            'upper_quartile': np.percentile(arr, 75)
        }

    for var in continuous_vars:
        arr = X_data[var].values
        cohort_distribution['continuous_medians'][var] = np.median(arr)
        cohort_distribution['continuous_quartiles'][var] = {
            'lower_quartile': np.percentile(arr, 25),
            'upper_quartile': np.percentile(arr, 75)
        }
    
    return cohort_distribution


def information_gain(value, info_dict):
    p = info_dict.get(value, 0)
    return -np.log2(p) if p>0 else 0


def calculate_competence(sample, model_data_stats, binary_vars, nominal_vars, ordinal_vars, continuous_vars):
    """
    Calculate similarity + info_gain for each of the four types of variables, then average:
        - binary: similarity(0/1 with mode) + info_gain
        - nominal: similarity(==mode) + info_gain
        - ordinal: 1 if within q1-q3 else (1-distance)
        - continuous: 1 if within q1-q3 else (1-distance)
    """
    vals = []

    # Binary
    for var in binary_vars:
        x = sample[var]
        mode = model_data_stats['binary_modes'][var]
        sim = 1 if x==mode else 0
        ig = information_gain(x, model_data_stats['binary_info'][var])
        vals.append((sim+ig)/2)
    
    # Nominal
    for var in nominal_vars:
        x = sample[var]
        mode = model_data_stats['nominal_modes'][var]
        sim = 1 if x==mode else 0
        ig = information_gain(x, model_data_stats['nominal_info'][var])
        vals.append((sim+ig)/2)
    
    # Ordinal
    for var in ordinal_vars:
        raw = sample[var]
        if hasattr(raw, 'cat'):
            x = raw.cat.codes
        else:
            x = raw
        med = model_data_stats['ordinal_medians'][var]
        q1 = model_data_stats['ordinal_quartiles'][var]['lower_quartile']
        q3 = model_data_stats['ordinal_quartiles'][var]['upper_quartile']
        iqr = q3 - q1
        if iqr <= 0:
            vals.append(1)
        elif q1 <= x <= q3:
            vals.append(1)
        else:
            distance = min(abs(x - med) / (q3 - q1), 1)
            vals.append(1-distance)

    # Continuous
    for var in continuous_vars:
        x = sample[var]
        med = model_data_stats['continuous_medians'][var]
        q1 = model_data_stats['continuous_quartiles'][var]['lower_quartile']
        q3 = model_data_stats['continuous_quartiles'][var]['upper_quartile']
        if q1 <= x <= q3:
            vals.append(1)
        else:
            distance = min(abs(x - med) / (q3 - q1), 1)
            vals.append(1-distance)

    return float(np.mean(vals))
