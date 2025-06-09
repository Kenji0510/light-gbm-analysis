import json
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

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
df_neg = load_clusters('../data/combined_data/20250530-1635/classification_data/nonhuman_clusters.json', label=0)

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
    max_depth=7,
    num_leaves=63,
    min_data_in_leaf=15,
    max_bin=512,
    learning_rate=0.05,         # 小さめでゆっくり学習
    n_estimators=500,           # 木の本数を増やす
    scale_pos_weight=161/79,    # クラス不均衡対策（Negative/Positive ≒ 2.04）
    feature_fraction=0.8,       # 過学習防止に
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)
model.fit(X_train, y_train)

# ——— 評価 ———
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
