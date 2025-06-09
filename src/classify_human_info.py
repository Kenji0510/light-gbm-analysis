import json

# 1. 元データ読み込み
with open('../data/combined_data/20250530-1635/combined_clusters.json', 'r') as f:
    data = json.load(f)

# 2. 分割用辞書
human = {}
nonhuman = {}

# 3. summary.height で振り分け
for cid, info in data.items():
    height = info.get('summary', {}).get('height', 0.0)
    if height >= 1.0:
        human[cid] = info
    else:
        nonhuman[cid] = info

# 4. JSON ファイルとして保存
with open('../data/combined_data/20250530-1635/classification_data/human_clusters.json', 'w', encoding='utf-8') as f:
    json.dump(human, f, ensure_ascii=False, indent=2)

with open('../data/combined_data/20250530-1635/classification_data/nonhuman_clusters.json', 'w', encoding='utf-8') as f:
    json.dump(nonhuman, f, ensure_ascii=False, indent=2)

print(f"保存完了: human_clusters.json ({len(human)} 件), nonhuman_clusters.json ({len(nonhuman)} 件)")
