import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

with open('updated_models_RandomForest_50_lasso.pkl', 'rb') as f:
    models = pickle.load(f)

pipeline = models[1]

feat_names0 = pipeline.named_steps['pre'].get_feature_names_out()
print(feat_names0)
print(len(feat_names0))

importances = pipeline.named_steps['model'].feature_importances_

idx_sorted = np.argsort(importances)[::-1][:10]
top_feats = feat_names0[idx_sorted]
top_scores = importances[idx_sorted]

print('Top 10 features for partition 0:')
for name,score in zip(top_feats, top_scores):
    print(f" {name:30s}  {score:.4f}")

counter = Counter()
sum_imps = defaultdict(float)

for model in models:
    feat_names = model.named_steps['pre'].get_feature_names_out()
    imps = model.named_steps['model'].feature_importances_
    top10_idx = np.argsort(imps)[::-1][:10]
    for i in top10_idx:
        name = feat_names[i]
        counter[feat_names[i]] += 1
        sum_imps[name] += imps[i]

top10 = [feat for feat,_ in counter.most_common(10)]

rows = []
for feat in top10:
    count = counter[feat]
    avg_imp = sum_imps[feat]/count
    rows.append((feat, count, avg_imp))

df_summary = pd.DataFrame(rows, columns=['Features', 'Count', 'Average Importance'])
df_summary = df_summary.sort_values('Average Importance', ascending=False)
print(df_summary)

# df_top = (
#     pd.DataFrame.from_dict(counter, orient='index', columns=['count']).sort_values('count', ascending=False)
# )
# print(df_top.head(10))

df_visual = df_summary.copy()
df_visual['Features'] = ['sex', 'previous_absence', 'AGECAT', 'Gest_age', 'Maternal_AGECAT', 'previous_exclusion1', 'Dep_quintile', 'bweight_centile', 'Maternal_smoking', 'Parity']
print(df_visual)

color_palette = ['#084081','#0868ac','#2b8cbe','#4eb3d3','#7bccc4','#a8ddb5','#ccebc5','#e0f3db','#edf8b1','#ffffd9']

plt.figure(figsize=(8,6))
sns.set_style('white')
ax = sns.barplot(
    x = 'Average Importance',
    y = 'Features',
    data = df_visual,
    palette = color_palette
)

ax.set_xticks([0.05, 0.1, 0.15, 0.2])
ax.set_xlim(0, 0.21)
# ax.set_title('Top 10 Feature Importances (Average over all models)', fontsize=14)
ax.set_xlabel('Average Importance', fontsize=12)
ax.set_ylabel(' ')
ax.grid(False, axis='x')

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 0.001, p.get_y()+p.get_height()/2, f'{width:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance_top10_update.pdf')
