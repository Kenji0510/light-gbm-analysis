import warnings
#warnings.filterwarnings('ignore')  # ← この行を追加

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import time
import glob

def load_clusters_from_dir(dir_path, label):
    dfs = []
    for json_path in glob.glob(f"{dir_path}/*.json"):
        # print(f"Loading clusters from: {json_path}")
        df = load_clusters(json_path, label)
        if not df.empty:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def load_clusters(path, label):
    # JSON を読み込む
    with open(path, 'r') as f:
        data = json.load(f)

    # ファイル名やキーからクラスタIDを推測
    # （ここでは "cluster_数字_数字" で始まるキーを探す例）
    cluster_key = next(k for k in data.keys() if k.startswith('cluster_'))
    
    rows = []
    row = {'cluster_id': cluster_key}
    
    # GRSD 特徴量を展開
    for grsd_key in ['ALL', 'LOWER', 'UPPER']:
        for i, v in enumerate(data['GRSD'].get(grsd_key, [])):
            row[f'grsd_{grsd_key.lower()}_{i}'] = v
    
    # cluster_info を展開
    for part in ['all', 'lower', 'upper']:
        info = data['cluster_info'].get(part, {})
        for k, v in info.items():
            row[f'{part}_{k}'] = v

    # ラベルをセット
    row['label'] = label
    rows.append(row)
    
    return pd.DataFrame(rows)

# ——— 陽例・陰例を読み込む ———
# df_pos = load_clusters('../data/combined_data/20250530-1635/classification_data/human_clusters.json', label=1)
# df_pos_extra = load_clusters('../data/combined_data/20250530-1713/classification_data/human_clusters.json', label=1)
# df_pos_extra_02 = load_clusters('../data/combined_data/20250530-1643/classification_data/human_clusters.json', label=1)
# df_pos = pd.concat([df_pos, df_pos_extra], ignore_index=True)
# df_pos = pd.concat([df_pos, df_pos_extra_02], ignore_index=True)

df_only_person = load_clusters_from_dir('/home/kenji/workspace/cpp/create-features-pcl/data/output/grsd-results/20250630/lidar03/person', label=1)
print(f"Loaded person clusters: {len(df_only_person)}")
df_bring_box = load_clusters_from_dir('/home/kenji/workspace/cpp/create-features-pcl/data/output/grsd-results/20250630/lidar03/box', label=2)
print(f"Loaded box clusters: {len(df_bring_box)}")
df_bring_suitcase = load_clusters_from_dir('/home/kenji/workspace/cpp/create-features-pcl/data/output/grsd-results/20250630/lidar03/suitcase', label=3)
print(f"Loaded suitcase clusters: {len(df_bring_suitcase)}")


# --- データ結合 ---
df = pd.concat([df_only_person, df_bring_box, df_bring_suitcase], ignore_index=True).sample(frac=1, random_state=42)
print(f"Total clusters: {len(df)} (Person: {len(df_only_person)}, Box: {len(df_bring_box)}, Suitcase: {len(df_bring_suitcase)})")

# --- 特徴量／ターゲット分割 ---
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
    #verbose=-1              # LightGBMの出力抑制
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

# --- 予測結果と正解データの対応をJSONで保存 ---
results = []
for idx, (true_label, pred_label) in zip(X_test.index, zip(y_test, y_pred)):
    result = {
        "index": int(idx),
        "cluster_id": df.loc[idx, "cluster_id"] if "cluster_id" in df.columns else None,
        "true_label": int(true_label),
        "pred_label": int(pred_label)
    }
    results.append(result)

with open("prediction_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

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