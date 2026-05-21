import pandas as pd
import os
from io import StringIO

def convert_keel_dat_to_csv(file_path):
    """
    读取 KEEL .dat 文件，处理分类特征和标签，并保存为干净的 .csv
    """
    filename = os.path.basename(file_path)
    print(f"🔄 正在处理: {filename} ...")
    
    if not os.path.exists(file_path):
        print(f"   ❌ 错误: 找不到文件 {file_path}")
        return

    # 1. 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 分离元数据(Header)和数据(Data)
    try:
        # KEEL 格式通常以 @data 标记数据的开始
        # split 可能会切分出多段，我们只取 @data 后的部分作为数据
        header_part, data_part = content.split('@data', 1)
    except ValueError:
        print(f"   ❌ 错误: 文件格式不对，未找到 @data 标记")
        return

    # 3. 解析列名 (Attributes)
    col_names = []
    for line in header_part.split('\n'):
        line = line.strip()
        # 解析 @attribute 行，例如: @attribute Sex {M, F, I}
        if line.lower().startswith('@attribute'):
            parts = line.split()
            if len(parts) >= 2:
                # 获取属性名 (parts[1])
                col_names.append(parts[1])
    
    print(f"   🔍 提取到 {len(col_names)} 个特征列名")

    # 4. 加载数据到 Pandas
    # skipinitialspace=True 用于去除逗号后的空格
    df = pd.read_csv(StringIO(data_part), header=None, names=col_names, skipinitialspace=True)

    # 5. 处理标签列 (Target)
    # 假设最后一列是 Class/Target
    target_col = df.columns[-1]
    print(f"   🎯 标签列检测为: '{target_col}'")
    
    # 将标签标准化：positive -> 1, negative -> 0
    # 注意：先转为字符串处理，防止本身就是数字
    # strip() 去除可能的空格
    df[target_col] = df[target_col].astype(str).str.strip().apply(
        lambda x: 1 if 'pos' in x.lower() else 0
    )
    
    # 打印类别分布
    counts = df[target_col].value_counts()
    print(f"   📊 类别分布: 多数类(0): {counts.get(0, 0)}, 少数类(1): {counts.get(1, 0)}")

    # 6. 特征数值化 (One-Hot Encoding)
    # 分离特征和标签
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 对特征进行 One-Hot 编码 (主要针对 car-good 和 abalone 的 Sex 列)
    # pd.get_dummies 会自动忽略已经是数值的列，只转换字符串列
    X_encoded = pd.get_dummies(X)
    
    # 如果维度发生了变化（说明有分类变量被展开了），打印提示
    if X_encoded.shape[1] != X.shape[1]:
        print(f"   ⚠️ 检测到分类特征，已执行 One-Hot 编码。特征数从 {X.shape[1]} 变为 {X_encoded.shape[1]}")

    # 7. 重新组合并保存
    df_clean = pd.concat([X_encoded, y], axis=1)
    
    output_filename = filename.replace('.dat', '_clean.csv')
    df_clean.to_csv(output_filename, index=False)
    print(f"   ✅ 转换成功! 已保存为: {output_filename}\n")

# ================= 执行区 =================
# 请确保以下文件在同一目录下，或者修改为绝对路径
files_to_convert = [
    'winequality-white-3_vs_7.dat', 
]

if __name__ == "__main__":
    for f in files_to_convert:
        convert_keel_dat_to_csv(f)