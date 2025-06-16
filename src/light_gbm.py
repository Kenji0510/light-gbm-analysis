import warnings
warnings.filterwarnings('ignore')  # ← この行を追加

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import time

def load_clusters(path, label):
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for cid, c in data.items():
        row = {'cluster_id': cid}
        # サマリー特徴量
        for k,v in c['summary'].items():
            row[k] = v
        # FPFH ヒストグラム
        for i,v in enumerate(c['fpfh']['fpfh']):
            row[f'fpfh_{i}'] = v
        row['label'] = label
        rows.append(row)
    return pd.DataFrame(rows)

# ——— 陽例・陰例を読み込む ———
df_pos = load_clusters('../data/combined_data/20250530-1635/classification_data/human_clusters.json', label=1)
df_pos_extra = load_clusters('../data/combined_data/20250530-1713/classification_data/human_clusters.json', label=1)
df_pos_extra_02 = load_clusters('../data/combined_data/20250530-1643/classification_data/human_clusters.json', label=1)
df_pos = pd.concat([df_pos, df_pos_extra], ignore_index=True)
df_pos = pd.concat([df_pos, df_pos_extra_02], ignore_index=True)

df_neg = load_clusters('../data/combined_data/20250530-1635/classification_data/nonhuman_clusters.json', label=0)
df_neg_extra = load_clusters('../data/combined_data/20250530-1713/classification_data/nonhuman_clusters.json', label=0)
df_neg_extra_02 = load_clusters('../data/combined_data/20250530-1643/classification_data/nonhuman_clusters.json', label=0)
df_neg = pd.concat([df_neg, df_neg_extra], ignore_index=True)
df_neg = pd.concat([df_neg, df_neg_extra_02], ignore_index=True)

# ——— データ結合 ———
df = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1, random_state=42)
print(f"Total clusters: {len(df)} (Positive: {len(df_pos)}, Negative: {len(df_neg)})")

# ——— 特徴量／ターゲット分割 ———
X = df.drop(columns=['cluster_id', 'label'])
y = df['label']

# ——— train/test 分割 ———
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ——— LightGBM モデル定義＆学習 ———
model = lgb.LGBMClassifier(
    device='gpu',
    max_depth=4,            # さらに浅く
    num_leaves=6,           # 葉の数もさらに減らす
    min_data_in_leaf=7,     # 葉に必要な最小データ数を増やす
    max_bin=64,             # ビン数も減らす
    learning_rate=0.1,     # 学習率を下げる
    n_estimators=60,        # 木の本数も減らす
    scale_pos_weight=352/268, # 陰陽例の比率に合わせる
    feature_fraction=0.6,
    bagging_fraction=0.6,
    bagging_freq=1,
    random_state=42,
    verbose=-1              # LightGBMの出力抑制
)
model.fit(X_train, y_train)

# ——— 評価 ———
start = time.time()
y_pred = model.predict(X_test)
elapsed = time.time() - start
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(f"Inference time for {len(X_test)} samples: {elapsed:.6f} sec")
print(f"Average inference time per sample: {elapsed/len(X_test):.6f} sec")

import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_
features = X.columns

# 上位20特徴だけプロット
indices = np.argsort(importances)[-20:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

plt.savefig('../data/figure/feature_importance.png', dpi=300)