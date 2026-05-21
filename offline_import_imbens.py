import os
import pandas as pd
import numpy as np
from collections import Counter
from imbens.datasets import fetch_zenodo_datasets

# ================= 配置区域 =================
# 1. 本地缓存路径 (保持不变)
LOCAL_DATA_HOME = "/home/liuhaoxuan/NAIM_labelweight+jingiam/my_dataset_cache"

# 2. 目标保存目录
DATA_ROOT_DIR = "/home/liuhaoxuan/NAIM_labelweight+jingiam/datasets/tabular/classification/"
CONF_SAVE_DIR = "/home/liuhaoxuan/NAIM_labelweight+jingiam/confs/experiment/databases/"

# 3. 目标数据集列表
TARGET_DATASETS = [
    'letter_img',    # IR=26:1, N=20000
    'webpage',       # IR=33:1, N=34780
    'ozone_level',   # IR=34:1, N=2536
    'mammography',   # IR=42:1, N=11183
]
# ===========================================

def normalize_labels(y):
    """
    核心修复函数：将任意二分类标签强制转换为 [0, 1]
    规则：数量多的类 -> 0 (多数类)，数量少的类 -> 1 (少数类/异常类)
    """
    counts = Counter(y)
    # 按数量降序排列 (多数类在前)
    sorted_classes = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    
    if len(sorted_classes) != 2:
        print(f"  [警告] 该数据集有 {len(sorted_classes)} 个类别，不是标准的二分类！")
        # 如果不是二分类，尝试直接映射 (0, 1, 2...)
        mapping = {k: i for i, k in enumerate(sorted(counts.keys()))}
    else:
        # 二分类标准映射：多数类->0, 少数类->1
        maj_class, min_class = sorted_classes[0], sorted_classes[1]
        mapping = {maj_class: 0, min_class: 1}
        print(f"  [清洗] 标签映射: {maj_class}(多数) -> 0, {min_class}(少数) -> 1")

    # 执行映射
    y_new = np.array([mapping[val] for val in y])
    return y_new

def generate_yaml_content(dataset_name, rel_csv_path, columns):
    """生成标准的 YAML 配置文件，固定 classes 为 [0, 1]"""
    cols_str = ""
    for col in columns:
        if col == "target":
            cols_str += "  target:              target  # DO NOT CHANGE THE VALUE\n"
        else:
            cols_str += f"  {col}:                  float\n"

    return f"""_target_: CMC_utils.datasets.ClassificationDataset # DO NOT CHANGE
_convert_: all # DO NOT CHANGE

name: {dataset_name} # dataset name
db_type: tabular # DO NOT CHANGE
# 因为我们已经在 CSV 里把数据清洗为 0 和 1 了，所以这里可以放心地写 [0, 1]
classes: [ 0, 1 ] 
label_type: multiclass 

task: classification # DO NOT CHANGE

path: ${{data_path}}/tabular/classification/{rel_csv_path} 

columns:  
{cols_str}

pandas_load_kwargs:
  header: 0  
  sep: ","   

dataset_class:  # DO NOT CHANGE
  _target_: CMC_utils.datasets.SupervisedTabularDatasetTorch # DO NOT CHANGE
  _convert_: all # DO NOT CHANGE
"""

def main():
    print(f"本地数据源: {LOCAL_DATA_HOME}/zenodo/")
    
    # 确保目录存在
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)
    os.makedirs(CONF_SAVE_DIR, exist_ok=True)

    print("-" * 50)
    print("开始加载并清洗数据集...")
    
    try:
        # 从本地缓存加载
        datasets = fetch_zenodo_datasets(filter_data=TARGET_DATASETS, data_home=LOCAL_DATA_HOME)
    except Exception as e:
        print(f"\n[Error] 加载出错: {e}")
        return

    for db_name, db_data in datasets.items():
        print(f"\n处理数据集: {db_name} ...")
        
        X = db_data.data
        y_raw = db_data.target
        
        # === 核心修复步骤：标准化标签 ===
        y_clean = normalize_labels(y_raw)
        
        # 转换 DataFrame
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y_clean  # 使用清洗后的标签
        
        # 1. 保存 CSV
        dataset_folder = os.path.join(DATA_ROOT_DIR, db_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        csv_filename = f"{db_name}.csv"
        csv_full_path = os.path.join(dataset_folder, csv_filename)
        df.to_csv(csv_full_path, index=False)
        print(f"  [√] CSV 已保存 (标签已修正为 0/1): {csv_full_path}")
        
        # 2. 生成 YAML
        rel_path_for_yaml = f"{db_name}/{csv_filename}"
        yaml_content = generate_yaml_content(db_name, rel_path_for_yaml, df.columns)
        
        yaml_filename = f"{db_name}.yaml"
        yaml_full_path = os.path.join(CONF_SAVE_DIR, yaml_filename)
        
        with open(yaml_full_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"  [√] 配置文件已更新: {yaml_full_path}")

    print("\n" + "=" * 50)
    print("修复完成！现在所有数据集的标签都统一为 0 和 1 了。")
    print("你可以直接运行实验，无需修改任何 YAML。")

if __name__ == "__main__":
    main()