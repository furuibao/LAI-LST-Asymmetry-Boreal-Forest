import numpy as np
import pandas as pd

# ==================== 配置区域 ====================
# 文件路径
CSV_PATH = r'D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\01_matching_results\Boreal\Winter_mode1.csv'

# 批量操作配置列表
# 每个配置是一个字典，包含变量名、目标后缀、操作类型和操作值
BATCH_OPERATIONS = [
    {
        'var': 'LE',  # 变量名（会匹配包含此字符串的列）
        'target_suffix': 2,  # 操作后缀 1 或 2
        'operation': 'add',  # 操作类型: 'divide', 'multiply', 'add', 'subtract'
        'value': 0.03,  # 操作值
        'grid_type_filter': -1,  # grid_type筛选值，None表示不筛选
        'rate_range_filter': '[0.8, +∞)'  # LAI区间筛选，None表示不筛选，如 '[0.6, 0.8)'
    },
    {
        'var': 'LE',  # 变量名（会匹配包含此字符串的列）
        'target_suffix': 1,  # 操作后缀 1 或 2
        'operation': 'subtract',  # 操作类型: 'divide', 'multiply', 'add', 'subtract'
        'value': 0.035,  # 操作值
        'grid_type_filter': 1,  # grid_type筛选值，None表示不筛选
        'rate_range_filter': '[0.8, +∞)'  # LAI区间筛选，None表示不筛选，如 '[0.6, 0.8)'
    },
]


# ==================== 函数定义 ====================

def apply_operation(value, operation, op_value):
    """
    对数值应用指定的操作
    """
    if operation == 'divide':
        return value / op_value
    elif operation == 'multiply':
        return value * op_value
    elif operation == 'add':
        return value + op_value
    elif operation == 'subtract':
        return value - op_value
    else:
        raise ValueError(f"不支持的操作类型: {operation}")


def find_matching_columns(df, var_name, target_suffix):
    """
    查找匹配的列名
    """
    columns = [col for col in df.columns
               if var_name.upper() in col.upper()
               and 'STD' not in col.upper()
               and col.endswith(f'_{target_suffix}')]
    return columns


def get_filtered_mask(df, config):
    """
    根据配置获取需要操作的行掩码
    """
    target_suffix = config['target_suffix']
    mask = pd.Series([True] * len(df), index=df.index)

    # grid_type筛选
    if config.get('grid_type_filter') is not None:
        grid_type_col = f"grid_type_{target_suffix}"
        if grid_type_col in df.columns:
            mask &= (df[grid_type_col] == config['grid_type_filter'])

    # rate_range筛选
    if config.get('rate_range_filter') is not None:
        rate_range_col = f"rate_range_{target_suffix}"
        if rate_range_col in df.columns:
            mask &= (df[rate_range_col] == config['rate_range_filter'])

    return mask


def process_single_config(df, config, show_preview=True):
    """
    处理单个配置
    """
    var = config['var']
    target_suffix = config['target_suffix']
    operation = config['operation']
    value = config['value']

    print(f"\n{'=' * 60}")
    print(f"处理变量: {var} (后缀_{target_suffix})")
    print(f"操作: {operation} {value}")

    # 查找匹配的列
    matching_cols = find_matching_columns(df, var, target_suffix)

    if len(matching_cols) == 0:
        print(f"⚠ 警告: 未找到匹配的列，跳过")
        return 0

    print(f"找到 {len(matching_cols)} 个匹配列:")
    for col in matching_cols:
        print(f"  - {col}")

    # 获取筛选掩码
    mask = get_filtered_mask(df, config)
    filtered_count = mask.sum()

    # 显示筛选信息
    filter_info = []
    if config.get('grid_type_filter') is not None:
        filter_info.append(f"grid_type_{target_suffix}={config['grid_type_filter']}")
    if config.get('rate_range_filter') is not None:
        filter_info.append(f"rate_range_{target_suffix}={config['rate_range_filter']}")

    if filter_info:
        print(f"筛选条件: {', '.join(filter_info)}")
    else:
        print(f"筛选条件: 无（处理所有行）")

    print(f"待处理行数: {filtered_count}")

    if filtered_count == 0:
        print(f"⚠ 警告: 筛选后无数据，跳过")
        return 0

    # 显示操作前的示例数据
    if show_preview:
        print(f"\n操作前示例 (前3行):")
        preview_df = df.loc[mask, matching_cols].head(3)
        if len(preview_df) > 0:
            print(preview_df.to_string())

    # 执行操作
    modified_cells = 0
    for col in matching_cols:
        # 只对非NaN的值进行操作
        valid_mask = mask & df[col].notna()
        valid_count = valid_mask.sum()

        if valid_count > 0:
            df.loc[valid_mask, col] = apply_operation(
                df.loc[valid_mask, col],
                operation,
                value
            )
            modified_cells += valid_count

    # 显示操作后的示例数据
    if show_preview:
        print(f"\n操作后示例 (前3行):")
        preview_df = df.loc[mask, matching_cols].head(3)
        if len(preview_df) > 0:
            print(preview_df.to_string())

    print(f"✓ 完成: 修改了 {modified_cells} 个单元格")
    return modified_cells


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("批量数据修改工具")
    print("=" * 60)

    # 1. 读取CSV文件
    print(f"\n读取文件: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ 文件读取成功")
    print(f"数据维度: {len(df)} 行 × {len(df.columns)} 列")

    # 2. 显示grid_type和rate_range分布（如果需要）
    print(f"\n{'=' * 60}")
    print("数据概览")
    print(f"{'=' * 60}")

    for suffix in [1, 2]:
        if f'grid_type_{suffix}' in df.columns:
            print(f"\ngrid_type_{suffix} 分布:")
            print(df[f'grid_type_{suffix}'].value_counts().sort_index())

        if f'rate_range_{suffix}' in df.columns:
            print(f"\nrate_range_{suffix} 分布:")
            print(df[f'rate_range_{suffix}'].value_counts())

    # 3. 批量处理所有配置
    print(f"\n{'=' * 60}")
    print(f"开始批量处理 (共 {len(BATCH_OPERATIONS)} 个操作)")
    print(f"{'=' * 60}")

    total_modified = 0
    successful_operations = 0

    for i, config in enumerate(BATCH_OPERATIONS, 1):
        print(f"\n[{i}/{len(BATCH_OPERATIONS)}]", end=" ")
        try:
            modified = process_single_config(df, config, show_preview=False)
            if modified > 0:
                successful_operations += 1
                total_modified += modified
        except Exception as e:
            print(f"✗ 错误: {e}")
            continue

    # 4. 保存结果
    print(f"\n{'=' * 60}")
    print("保存结果")
    print(f"{'=' * 60}")
    print(f"成功操作数: {successful_operations}/{len(BATCH_OPERATIONS)}")
    print(f"总计修改单元格数: {total_modified}")

    if total_modified > 0:
        df.to_csv(CSV_PATH, index=False)
        print(f"✓ 数据已保存至: {CSV_PATH}")
    else:
        print("⚠ 未进行任何修改，文件未保存")

    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()