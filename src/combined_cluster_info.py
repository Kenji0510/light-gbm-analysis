import os
import glob
import re
import json
from collections import OrderedDict

def sorted_cluster_paths(dir_path: str, pattern: str):
    """ glob(pattern) → 'cluster_x_y...' の x,y でソート """
    pat = re.compile(r'cluster_(\d+)_(\d+)')
    paths = glob.glob(os.path.join(dir_path, pattern))
    def keyfn(p):
        m = pat.search(os.path.basename(p))
        return tuple(map(int, m.groups())) if m else (float('inf'), float('inf'))
    return sorted(paths, key=keyfn)

if __name__ == '__main__':
    # ディレクトリ
    summary_dir = '/home/kenji/workspace/python3/pcd_operation/data/averaging_data/20250530-1635-onlyPerson_json'
    fpfh_dir    = '/home/kenji/workspace/python3/pcd_operation/data/averaging_data/20250530-1635-onlyPerson_choiced_fpfh'

    # サマリー JSON は *_top3.json
    summary_paths = sorted_cluster_paths(summary_dir, '*_top3.json')
    # FPFH JSON は *_top3.json を除外
    all_json = glob.glob(os.path.join(fpfh_dir, '*.json'))
    fpfh_paths = [p for p in all_json if not p.endswith('_top3.json')]
    fpfh_paths = sorted_cluster_paths(fpfh_dir, '*.json')
    fpfh_paths = [p for p in fpfh_paths if not p.endswith('_top3.json')]

    # デバッグ出力
    print(f"Summary files: {len(summary_paths)}")
    print(f"FPFH files:    {len(fpfh_paths)}")

    # FPFH ファイル名 → パス のマップ
    fpfh_map = {}
    for p in fpfh_paths:
        name = os.path.splitext(os.path.basename(p))[0]  # 'cluster_00_01' など
        fpfh_map[name] = p
    print("FPFH map keys:", sorted(fpfh_map.keys())[:10], "...")

    combined = {}

    for s_path in summary_paths:
        with open(s_path, 'r') as f:
            summary_json = json.load(f)

        # ファイルごとではなく、中のキー単位でループ
        for key, summary_entry in summary_json.items():
            # summary_entry = { num_points, width, ... } 
            # FPFH 側を探す
            combined[key] = {
                'summary': summary_entry,
                'fpfh':    {}  # デフォルト
            }
            fp = fpfh_map.get(key)
            if fp:
                with open(fp, 'r') as f2:
                    fpfh_json = json.load(f2)
                # fpfh_json の中身が { "cluster_00_01": {...} } の場合
                if key in fpfh_json:
                    combined[key]['fpfh'] = fpfh_json[key]
                else:
                    # 直接 { "fpfh": [...] } 構造ならそのまま
                    combined[key]['fpfh'] = fpfh_json

    # 最終結果を確認
    print(f"Combined entries: {len(combined)}")
    print(json.dumps(combined, indent=2, ensure_ascii=False))

# キーを数値順にソートして OrderedDict 化
sorted_keys = sorted(
    combined.keys(),
    key=lambda k: tuple(map(int, k.split('_')[1:])))
ordered = OrderedDict((k, combined[k]) for k in sorted_keys)

# ファイルへ書き出し
out_path = os.path.join("../data/combined_data/20250530-1635", 'combined_clusters.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(ordered, f, ensure_ascii=False, indent=2)

print(f"Saved combined JSON to {out_path}")