#!/usr/bin/env python3
"""
简化版LE单位转换脚本
将LE相关列从 J/m²/day 转换为 W/m²，直接覆盖原数据
"""

import pandas as pd

# 读取CSV文件
input_file = r"D:\article\LAI-LST-Asymmetric\data\outputCSVData\adjusted_matching\mode1\Boreal\Winter_mode1_adjusted.csv"
df = pd.read_csv(input_file)

# LE相关列名
le_columns = [
    'climate_le_diff_1',
    'climate_le_diff_2',
    'climate_le_diff_std_1',
    'climate_le_diff_std_2',
    'le_diff_lai_1',
    'le_diff_lai_2',
    'le_diff_total_1',
    'le_diff_total_2',
    'le_diff_lai_std_1',
    'le_diff_lai_std_2'
]

# 转换系数：1 J/m²/day = 1/(24*3600) W/m²
conversion_factor = 1.0 / (24 * 3600)  # = 1.157e-5

print("开始转换LE单位：J/m²/day → W/m²")
print(f"转换系数: {conversion_factor:.6e}")

# 转换每个LE列
converted_count = 0
for col in le_columns:
    if col in df.columns:
        df[col] = df[col] * conversion_factor
        converted_count += 1
        print(f"✓ 已转换: {col}")
    else:
        print(f"✗ 列不存在: {col}")

print(f"\n共转换了 {converted_count} 个LE列")

# 保存结果，覆盖原文件
df.to_csv(input_file, index=False)
print(f"已保存转换后的数据到: {input_file}")
print("转换完成！")