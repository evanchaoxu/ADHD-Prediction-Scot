# Breakdown
# This file contains a function that calculates each model's contribution to each feature in the test data (breakdown)
# as well as the overall score.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import logging
import os
from metrics import calculate_competence


# Calculate the competence breakdown of each model for each feature in the test data 
# and also calculate the overall competence score

def compute_competence_breakdown_for_model(model, X_test):
    """
    For a given model and its test data X_test (keeping only the model's selected features),
    compute the competence score for each sample on each of the features used by the model separately,
    as well as the overall competence score.
    """
    stats = model.model_data
    
    cols = model.selected_features + ['Overall_Competence']
    breakdown_df = pd.DataFrame(index=X_test.index, columns=cols)
    
    for feature in model.selected_features:
        if feature in model.new_binary_vars:
            mode_val = stats['binary_modes'][feature]
            info_val = stats['binary_info'][feature]
            def fn(row):
                x = row[feature]
                sim = 1 if x == mode_val else 0
                ig = -np.log2(info_val['p0']) if x == 0 and info_val['p0'] > 0 else (-np.log2(info_val['p1']) if x == 1 and info_val['p1'] > 0 else 0)
                ig = min(ig, 1)
                return (sim + ig) / 2
        elif feature in model.new_nominal_vars:
            mode_val = stats['nominal_modes'][feature]
            info_val = stats['nominal_info'][feature]
            def fn(row):
                x = row[feature]
                sim = 1 if x == mode_val else 0
                p = info.get(x, 0)
                ig = -np.log2(p) if p>0 else 0
                ig = min(ig, 1)
                return (sim + ig) / 2
        elif feature in model.new_ordinal_vars:
            med = stats['ordinal_medians'][feature]
            q1 = stats['ordinal_quartiles'][feature]['lower_quartile']
            q3 = stats['ordinal_quartiles'][feature]['upper_quartile']
            def fn(row):
                x = row[feature].cat.codes
                if q1 <= x <= q3:
                    return 1
                else:
                    return 1-min(abs(x - med)/(q3-q1), 1)
        elif feature in model.new_continuous_vars:
            med = stats['continuous_medians'][feature]
            q1 = stats['continuous_quartiles'][feature]['lower_quartile']
            q3 = stats['continuous_quartiles'][feature]['upper_quartile']
            def fn(row):
                x = row[feature]
                if q1 <= x <= q3:
                    return 1
                else:
                    return 1 - min(abs(val - med) / (q3 - q1), 1)
        else:
            def fn(row): return np.nan
        
        breakdown_df[feature] = X_test.apply(fn, axis=1)
    
    def overall_fn(row):
        return calculate_competence(row, stats, model.new_binary_vars, model.new_nominal_vars, model.new_ordinal_vars, model.new_continuous_vars)
    breakdown_df['Overall_Competence'] = X_test.apply(overall_fn, axis = 1)

    return breakdown_df

def compute_and_save_competence_breakdown(X_test, models, output_folder='competence_breakdown'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    breakdown_dict = {}
    for i, model in enumerate(models):
        X_test_model = X_test[model.selected_features]
        breakdown_df = compute_competence_breakdown_for_model(model, X_test_model)
        breakdown_dict[i] = breakdown_df
        filename = os.path.join(output_folder, f"competence_breakdown_partition_{i}.csv")
        breakdown_df.to_csv(filename, index=True)
        print(f"Competence breakdown for partition {i} saved to {filename}")
    return breakdown_dict


def compute_full_competence_breakdown_for_model(model, X_test_full):
    """
    For a given model and its test data X_test_full (contains full original features),
    compute the full competence score for each sample on each of the features used by the model separately,
    as well as the overall full competence score.
    """
    stats = model.full_model_data
    full_features = model.full_binary_vars + model.full_nominal_vars + model.full_ordinal_vars + model.full_continuous_vars
    cols = full_features + ['Overall_Full_Competence']
    breakdown_df = pd.DataFrame(index=X_test_full.index, columns=cols)
    
    for feature in full_features:
        if feature in model.full_binary_vars:
            mode_val = stats['binary_modes'][feature]
            info_val = stats['binary_info'][feature]
            def fn(row):
                x = row[feature]
                sim = 1 if x == mode_val else 0
                ig = -np.log2(info_val['p0']) if x == 0 and info_val['p0'] > 0 else (-np.log2(info_val['p1']) if x == 1 and info_val['p1'] > 0 else 0)
                ig = min(ig, 1)
                return (sim + ig) / 2
        elif feature in model.full_nominal_vars:
            mode_val = stats['nominal_modes'][feature]
            info_val = stats['nominal_info'][feature]
            def fn(row):
                x = row[feature]
                sim = 1 if x == mode_val else 0
                p = info.get(x, 0)
                ig = -np.log2(p) if p>0 else 0
                ig = min(ig, 1)
                return (sim + ig) / 2
        elif feature in model.full_ordinal_vars:
            med = stats['ordinal_medians'][feature]
            q1 = stats['ordinal_quartiles'][feature]['lower_quartile']
            q3 = stats['ordinal_quartiles'][feature]['upper_quartile']
            def fn(row):
                x = row[feature].cat.codes
                if q1 <= x <= q3:
                    return 1
                else:
                    return 1-min(abs(x - med)/(q3-q1), 1)    
        elif feature in model.full_continuous_vars:
            med = stats['continuous_medians'][feature]
            q1 = stats['continuous_quartiles'][feature]['lower_quartile']
            q3 = stats['continuous_quartiles'][feature]['upper_quartile']
            def fn(row):
                x = row[feature]
                if q1 <= x <= q3:
                    return 1
                else:
                    return 1 - min(abs(val - med) / (q3 - q1), 1)
        else:
            def fn(row): return np.nan
        
        breakdown_df[feature] = X_test_full.apply(fn, axis=1)
    
    def overall_full_fn(row):
        return calculate_competence(row, stats, model.full_binary_vars, model.full_nominal_vars, model.full_ordinal_vars, model.full_continuous_vars)
    
    breakdown_df['Overall_Full_Competence'] = X_test_full.apply(overall_full_fn, axis = 1)
    return breakdown_df

def compute_and_save_full_competence_breakdown(X_test_full, models, output_folder='full_competence_breakdown'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    breakdown_dict = {}
    for i, model in enumerate(models):
        full_features = model.full_binary_vars + model.full_nominal_vars + model.full_ordinal_vars + model.full_continuous_vars
        X_test_model = X_test_full[full_features]
        breakdown_df = compute_full_competence_breakdown_for_model(model, X_test_model)
        breakdown_dict[i] = breakdown_df
        filename = os.path.join(output_folder, f"full_competence_breakdown_partition_{i}.csv")
        breakdown_df.to_csv(filename, index=True)
        print(f"Full Competence breakdown for partition {i} saved to {filename}")
    return breakdown_dict


# def compute_competence_by_group_full(X_data, models, group_type = 'sex', sex_column = 'sex_1', output_folder='cs_group_full'):
#     """
#     Calculate the average competence score of each model for each subclassification under the specified grouping feature and visualise the results.
#     """
#     if group_type == 'sex':
#         subfolder = os.path.join(output_folder, 'sex')
#         group_columns = [sex_column]
#         group_labels = {0: 'Female', 1:'Male'}
#     elif group_type == 'deprivation':
#         subfolder = os.path.join(output_folder, 'deprivation')
#         group_columns = ['Dep_quintile_1', 'Dep_quintile_2', 'Dep_quintile_3', 'Dep_quintile_4']
#         group_labels = {col: col for col in group_columns}
#     else:
#         raise ValueError("Unsupported group_type. Use 'sex' or 'deprivation'.")

#     if not os.path.exists(subfolder):
#         os.makedirs(subfolder)
    
#     results = []
#     for i,model in enumerate(models):
#         # X_model = X_data[model.selected_features]     # selected
        
#         full_features = model.full_binary_vars + model.full_continuous_vars     # full
#         X_model = X_data[full_features]                                         # full

#         competence_scores = []
#         for idx, row in X_model.iterrows():
#             # cs = calculate_competence(row, model.model_data, model.new_binary_vars, model.new_continuous_vars)      # selected
#             cs = calculate_competence(row, model.full_model_data, model.full_binary_vars, model.full_continuous_vars) # full
#             competence_scores.append(cs)
#         competence_scores = np.array(competence_scores)
        
#         model_result = {'model_index': i}
#         if group_type == 'sex':
#             group_values = X_data[sex_column].values
#             for val, label in group_labels.items():
#                 mask = (group_values == val)
#                 avg = np.mean(competence_scores[mask]) if np.sum(mask) > 0 else np.nan
#                 model_result[label] = avg
#         elif group_type == 'deprivation':
#             for col in group_columns:
#                 group_values = X_data[col].values
#                 mask = (group_values == 1)
#                 avg = np.mean(competence_scores[mask]) if np.sum(mask) > 0 else np.nan
#                 model_result[col] = avg
#         results.append(model_result)
    
#     results_df = pd.DataFrame(results)

#     # Calculate the overall average competence score for all models in each subgroup
#     overall_avgs = {}
#     if group_type == 'sex':
#         for label in group_labels.values():
#             overall_avgs[label] = results_df[label].mean()
#     elif group_type == 'deprivation':
#         for col in group_columns:
#             overall_avgs[col] = results_df[col].mean()
    
#     results_csv = os.path.join(subfolder, f"results_competence_{group_type}_full.csv")
#     results_df.to_csv(results_csv, index=False)
#     print(f"Reults saved to {results_csv}.")

#     plt.figure(figsize=(8,6))
#     if group_type == 'sex':
#         categories = list(group_labels.values())  # ['Male', 'Female']
#         colors = ['#6baed6', '#f768a1']  # ['blue', 'pink']
#     else:
#         categories = group_columns
#         colors = ['#7fc97f', '#fdc086', '#ffff99', '#a6cee3']   #['green', 'orange', 'yellow', 'blue']
    
#     overall_values = [overall_avgs[cat] for cat in categories]
#     plt.bar(categories, overall_values, color=colors)
#     plt.xlabel('Group')
#     plt.ylabel('Average Competence Score')
#     plt.title(f'Overall Average Competence Score by {group_type.capitalize()} Group')
#     plt.ylim(0, 1)
#     overall_pdf = os.path.join(subfolder, f"overall_{group_type}_average_competence.pdf")
#     plt.savefig(overall_pdf)
#     plt.close()


#     plt.figure(figsize=(50,10))
#     x = np.arange(len(results_df))
#     if group_type == 'sex':
#         width = 0.35
#         plt.bar(x - width/2, results_df['Male'], width, label='Male', color='#6baed6')
#         plt.bar(x + width/2, results_df['Female'], width, label='Female', color='#f768a1')
#         plt.title('Model-wise Average Competence Score by Sex Group')
#         plt.legend()
#     else:
#         width = 0.2
#         plt.bar(x - 1.5*width, results_df['Dep_quintile_1'], width, label='Dep_quintile_1', color='#7fc97f')
#         plt.bar(x - 0.5*width, results_df['Dep_quintile_2'], width, label='Dep_quintile_2', color='#fdc086')
#         plt.bar(x + 0.5*width, results_df['Dep_quintile_3'], width, label='Dep_quintile_3', color='#ffff99')
#         plt.bar(x + 1.5*width, results_df['Dep_quintile_4'], width, label='Dep_quintile_4', color='#a6cee3')
#         plt.title('Model-wise Average Competence Score by Deprivation Group')
#         plt.legend()
    
#     plt.xlabel('Model Index')
#     plt.ylabel('Average Competence score')
#     plt.xticks(x, results_df['model_index'])
#     plt.ylim(0, 1)
#     modelwise_pdf = os.path.join(subfolder, f"model_wise_competence_{group_type}.pdf")
#     plt.savefig(modelwise_pdf)
#     plt.close()

#     return results_df, overall_avgs

# def compute_competence_by_group_selected(X_data, models, group_type = 'sex', sex_column = 'sex_1', output_folder='cs_group_selected'):
#     """
#     Calculate the average competence score of each model for each subclassification under the specified grouping feature and visualise the results.
#     """
#     if group_type == 'sex':
#         subfolder = os.path.join(output_folder, 'sex')
#         group_columns = [sex_column]
#         group_labels = {0: 'Female', 1:'Male'}
#     elif group_type == 'deprivation':
#         subfolder = os.path.join(output_folder, 'deprivation')
#         group_columns = ['Dep_quintile_1', 'Dep_quintile_2', 'Dep_quintile_3', 'Dep_quintile_4']
#         group_labels = {col: col for col in group_columns}
#     else:
#         raise ValueError("Unsupported group_type. Use 'sex' or 'deprivation'.")

#     if not os.path.exists(subfolder):
#         os.makedirs(subfolder)
    
#     results = []
#     for i,model in enumerate(models):
#         X_model = X_data[model.selected_features]     # selected
#         competence_scores = []
#         for idx, row in X_model.iterrows():
#             cs = calculate_competence(row, model.model_data, model.new_binary_vars, model.new_continuous_vars)      # selected
#             competence_scores.append(cs)
#         competence_scores = np.array(competence_scores)
        
#         model_result = {'model_index': i}
#         if group_type == 'sex':
#             group_values = X_data[sex_column].values
#             for val, label in group_labels.items():
#                 mask = (group_values == val)
#                 avg = np.mean(competence_scores[mask]) if np.sum(mask) > 0 else np.nan
#                 model_result[label] = avg
#         elif group_type == 'deprivation':
#             for col in group_columns:
#                 group_values = X_data[col].values
#                 mask = (group_values == 1)
#                 avg = np.mean(competence_scores[mask]) if np.sum(mask) > 0 else np.nan
#                 model_result[col] = avg
#         results.append(model_result)
    
#     results_df = pd.DataFrame(results)

#     # Calculate the overall average competence score for all models in each subgroup
#     overall_avgs = {}
#     if group_type == 'sex':
#         for label in group_labels.values():
#             overall_avgs[label] = results_df[label].mean()
#     elif group_type == 'deprivation':
#         for col in group_columns:
#             overall_avgs[col] = results_df[col].mean()
    
#     results_csv = os.path.join(subfolder, f"results_competence_{group_type}_selected.csv")
#     results_df.to_csv(results_csv, index=False)
#     print(f"Reults(selected features) saved to {results_csv}.")

#     plt.figure(figsize=(8,6))
#     if group_type == 'sex':
#         categories = list(group_labels.values())  # ['Male', 'Female']
#         colors = ['#6baed6', '#f768a1']  # ['blue', 'pink']
#     else:
#         categories = group_columns
#         colors = ['#7fc97f', '#fdc086', '#ffff99', '#a6cee3']   #['green', 'orange', 'yellow', 'blue']
    
#     overall_values = [overall_avgs[cat] for cat in categories]
#     plt.bar(categories, overall_values, color=colors)
#     plt.xlabel('Group')
#     plt.ylabel('Average Competence Score')
#     plt.title(f'Overall Average Competence Score (Selected features) by {group_type.capitalize()} Group')
#     plt.ylim(0, 1)
#     overall_pdf = os.path.join(subfolder, f"overall_{group_type}_average_competence_selected.pdf")
#     plt.savefig(overall_pdf)
#     plt.close()


#     plt.figure(figsize=(50,10))
#     x = np.arange(len(results_df))
#     if group_type == 'sex':
#         width = 0.35
#         plt.bar(x - width/2, results_df['Male'], width, label='Male', color='#6baed6')
#         plt.bar(x + width/2, results_df['Female'], width, label='Female', color='#f768a1')
#         plt.title('Model-wise Average Competence Score (Selected features) by Sex Group')
#         plt.legend()
#     else:
#         width = 0.2
#         plt.bar(x - 1.5*width, results_df['Dep_quintile_1'], width, label='Dep_quintile_1', color='#7fc97f')
#         plt.bar(x - 0.5*width, results_df['Dep_quintile_2'], width, label='Dep_quintile_2', color='#fdc086')
#         plt.bar(x + 0.5*width, results_df['Dep_quintile_3'], width, label='Dep_quintile_3', color='#ffff99')
#         plt.bar(x + 1.5*width, results_df['Dep_quintile_4'], width, label='Dep_quintile_4', color='#a6cee3')
#         plt.title('Model-wise Average Competence Score (Selected fefatures) by Deprivation Group')
#         plt.legend()
#     plt.xlabel('Model Index')
#     plt.ylabel('Average Competence score')
#     plt.xticks(x, results_df['model_index'])
#     plt.ylim(0, 1)
#     modelwise_pdf = os.path.join(subfolder, f"model_wise_competence_{group_type}_selected.pdf")
#     plt.savefig(modelwise_pdf)
#     plt.close()

#     return results_df, overall_avgs
