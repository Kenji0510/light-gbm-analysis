import os
import glob
import re
import json

def sorted_cluster_paths(dir_path: str, ext: str):
    """
    dir_path 以下のファイル *.{ext} を
    ファイル名に含まれる 'cluster_x_y' の x, y をキーに昇順ソートして返す
    """
    # 例: cluster_00_03.pcd -> groups = ['00','03']
    pat = re.compile(r'cluster_(\d+)[_](\d+)')

    paths = glob.glob(os.path.join(dir_path, f'*.{ext}'))
    def sort_key(p: str):
        m = pat.search(os.path.basename(p))
        if not m:
            # マッチしないものは後ろへ回す
            return (float('inf'), float('inf'))
        x, y = map(int, m.groups())
        return (x, y)

    return sorted(paths, key=sort_key)

if __name__ == '__main__':
    # PCD ファイルをソート
    pcd_dir = '/home/kenji/workspace/python3/pcd_operation/data/averaging_data/20250530-1635-onlyPerson_json'
    pcd_list = sorted_cluster_paths(pcd_dir, 'json')
    print("JSON files (sorted):")
    for p in pcd_list:
        print("  ", os.path.basename(p))

    # JSON (サマリー) ファイルをソート
    json_dir = '/home/kenji/workspace/python3/pcd_operation/data/averaging_data/20250530-1635-onlyPerson_choiced_fpfh'
    # もし JSON 名にも y があれば同じ関数で OK
    json_list = sorted_cluster_paths(json_dir, 'json')
    print("\nSummary JSON files (sorted):")
    for j in json_list:
        print("  ", os.path.basename(j))

    # 各 JSON を順番に読み込む例
    for jpath in json_list:
        with open(jpath, 'r') as f:
            data = json.load(f)
        # data は { "cluster_00_01": {...}, "cluster_00_03": {...}, ... }
        # ここでもキー（cluster_x_y）をソートして処理できます：
        for key in sorted(data.keys(), key=lambda k: tuple(map(int, k.split('_')[1:]))):
            entry = data[key]
            # --- ここに entry の処理を書く ---

            print(f"{os.path.basename(jpath)} -> {key}: {entry}")