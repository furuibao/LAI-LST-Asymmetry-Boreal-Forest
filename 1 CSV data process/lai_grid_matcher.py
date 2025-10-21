"""
该脚本用于匹配网格:
基于"空代时"的思想, 将所有时间段的5km格网进行匹配

修改版：支持用户指定区域、模式和季节进行匹配，并按季节分别导出结果

1. 四种匹配模式：
   - Mode1: 相同季节内的相反变化匹配
   - Mode2: 不同季节内的相同变化匹配
   - Mode3: 相同季节内的相同变化匹配
   - Mode4: 不同季节内的相反变化匹配

2. 输出文件结构：
   matching_results/
   ├── Boreal/
   │   ├── Spring_mode1.csv       # 单个季节匹配
   │   ├── Summer_mode1.csv       # 单个季节匹配
   │   ├── Spring_Summer_mode2.csv # 不同季节匹配
   │   └── ...
   ├── Temperate/...
   ├── Tropical/...
   ├── matching_summary_*.csv     # 总体统计
   └── region_summary_*.csv       # 区域统计
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Iterator, Optional
from datetime import datetime
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from itertools import combinations
from collections import defaultdict

warnings.filterwarnings('ignore')


class OptimizedGridMatchingAnalyzer:
    """优化版网格匹配分析器 - 高性能实现，按季节分别导出"""

    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 50000):
        """
        初始化分析器
        :param input_dir: 输入目录路径
        :param output_dir: 输出目录路径
        :param chunk_size: 分块处理大小, 用于控制内存使用
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_size = chunk_size

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        # LAI变化率区间定义
        self.rate_intervals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, np.inf])

        # 修正数据类型优化配置 - 根据实际数据结构
        self.dtypes_config = {
            'large_grid_id': 'int32',  # large_grid_id是整数
            # grid_id是字符串，不需要转换
            'grid_type': 'int8',
            'LAI_Change_Rate_mean': 'float32',
            'LAI_Begin_mean': 'float32',
            'region': 'category',
            'year_pair': 'category',
            'season': 'category',  # 小写的season
            'analysis_group': 'category'
        }

        # 统计信息
        self.stats = {
            'total_files': 0,
            'total_records': 0,
            'regions': set(),
            'year_pairs': set(),
            'seasons': set()
        }

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame的数据类型, 以减少内存占用
        """
        # 应用数据类型优化
        for col, dtype in self.dtypes_config.items():
            if col in df.columns:
                try:
                    if dtype == 'category':
                        df[col] = df[col].astype('category')
                    else:
                        # 对数值列进行转换
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # 删除转换失败的行（如果有的话）
                        if df[col].isnull().any():
                            before_count = len(df)
                            df = df.dropna(subset=[col])
                            after_count = len(df)
                            if before_count > after_count:
                                self.logger.warning(f"删除了 {before_count - after_count} 行包含无效 {col} 的数据")
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"⚠️ 无法转换列 {col} 为 {dtype}: {e}")

        # 特殊处理grid_id（保持为字符串）
        if 'grid_id' in df.columns:
            df['grid_id'] = df['grid_id'].astype(str)

        # 智能优化其他数值列
        for col in df.select_dtypes(include=['float64']).columns:
            if col not in self.dtypes_config:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                except:
                    continue

        for col in df.select_dtypes(include=['int64']).columns:
            if col not in self.dtypes_config:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    if col_min >= -128 and col_max <= 127:
                        df[col] = df[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype('int16')
                    else:
                        df[col] = df[col].astype('int32')
                except:
                    continue

        return df

    def load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        加载单个文件进行预处理
        """
        try:
            # 从路径提取信息
            parts = file_path.parts
            if len(parts) >= 3:
                region_from_path = parts[-3]
                year_pair_from_path = parts[-2]
                season_from_filename = file_path.stem.lower().capitalize()
            else:
                region_from_path = year_pair_from_path = season_from_filename = 'unknown'

            # 标准化季节名称
            season_mapping = {
                'Spring': 'Spring', 'Summer': 'Summer',
                'Autumn': 'Autumn', 'Winter': 'Winter',
                'Fall': 'Autumn'
            }
            season_from_filename = season_mapping.get(season_from_filename, season_from_filename)

            # 读取CSV文件，不强制指定数据类型
            df = pd.read_csv(file_path, low_memory=False)

            # 如果数据中没有region或year_pair列，从路径补充
            if 'region' not in df.columns:
                df['region'] = region_from_path
            if 'year_pair' not in df.columns:
                df['year_pair'] = year_pair_from_path

            # 处理季节列 - 统一为大写的Season用于匹配
            if 'season' in df.columns:
                # 数据中有小写的season，创建大写版本
                df['Season'] = df['season'].str.capitalize()
            elif 'Season' not in df.columns:
                # 如果都没有，从文件名创建
                df['Season'] = season_from_filename
                df['season'] = season_from_filename.lower()

            # 添加源文件信息
            df['source_file'] = file_path.name
            df['season_from_path'] = season_from_filename

            # 优化数据类型
            df = self.optimize_dataframe(df)

            # 更新统计信息
            self.stats['regions'].add(df['region'].iloc[0] if not df.empty else region_from_path)
            self.stats['year_pairs'].add(df['year_pair'].iloc[0] if not df.empty else year_pair_from_path)
            self.stats['seasons'].add(df['Season'].iloc[0] if not df.empty else season_from_filename)

            return df

        except Exception as e:
            self.logger.error(f"❌ 加载文件失败 {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def load_all_data_parallel(self) -> pd.DataFrame:
        """
        并行加载所有文件
        """
        self.logger.info("🔄 开始并行加载所有文件...")

        all_files = list(self.input_dir.rglob("*.csv"))
        if not all_files:
            raise FileNotFoundError(f"在 {self.input_dir} 中未找到CSV文件")

        self.stats['total_files'] = len(all_files)
        data_list = []

        # 使用线程池并行加载
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.load_single_file, file_path): file_path
                for file_path in all_files
            }

            # 收集结果
            for future in tqdm(as_completed(future_to_file),
                               total=len(all_files),
                               desc="📁 并行加载文件",
                               unit="个"):
                try:
                    df = future.result()
                    if not df.empty:
                        data_list.append(df)
                except Exception as e:
                    file_path = future_to_file[future]
                    self.logger.error(f"❌ 处理文件出错 {file_path.name}: {e}")

        if not data_list:
            raise ValueError("没有加载到任何数据")

        # 合并数据
        self.logger.info("🔗 合并数据...")
        global_df = pd.concat(data_list, ignore_index=True, copy=False)

        # 最终优化
        global_df = self.optimize_dataframe(global_df)

        # 验证必需列
        required_cols = ['grid_id', 'Season', 'grid_type', 'LAI_Change_Rate_mean']
        missing_cols = [col for col in required_cols if col not in global_df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")

        # 删除临时数据释放内存
        del data_list
        gc.collect()

        self.stats['total_records'] = len(global_df)

        self.logger.info(f"✅ 成功加载 {self.stats['total_files']} 个文件，共 {self.stats['total_records']:,} 条记录")
        self.logger.info(f"📊 区域分布: {sorted(self.stats['regions'])}")
        self.logger.info(f"📊 季节分布: {sorted(self.stats['seasons'])}")
        self.logger.info(f"📊 内存使用: {global_df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        return global_df

    def get_rate_interval_vectorized(self, rates: np.ndarray, use_abs: bool = True) -> np.ndarray:
        """
        向量化计算LAI变化率区间 - 优化版本
        """
        values = np.abs(rates) if use_abs else rates
        # 使用searchsorted替代digitize，性能更好
        return np.searchsorted(self.rate_intervals[1:], values, side='right')

    def vectorized_matching_mode1(self, region_data: pd.DataFrame, region: str, target_season: Optional[str] = None) -> List[Tuple]:
        """
        向量化模式1匹配 (相同季节相反变化) - 高性能版本
        """
        matches = []

        # 如果指定了目标季节，过滤数据
        if target_season:
            region_data = region_data[region_data['Season'] == target_season].copy()
            if len(region_data) == 0:
                self.logger.warning(f"⚠️ 在{region}区域中未找到{target_season}季节的数据")
                return matches

        # 预计算区间
        rates = region_data['LAI_Change_Rate_mean'].values
        intervals = self.get_rate_interval_vectorized(rates, use_abs=True)
        region_data = region_data.copy()
        region_data['interval'] = intervals

        # 按(grid_id, Season)分组
        grouped = region_data.groupby(['grid_id', 'Season'], observed=True)

        with tqdm(total=len(grouped),
                  desc=f"🔄 模式1-相同季节相反变化({region})",
                  unit="格网组") as pbar:

            for (grid_id, season), group in grouped:
                if len(group) < 2:
                    pbar.update(1)
                    continue

                # 使用布尔索引快速分离
                type_1_mask = group['grid_type'] == 1
                type_neg1_mask = group['grid_type'] == -1

                if not (type_1_mask.any() and type_neg1_mask.any()):
                    pbar.update(1)
                    continue

                type_1_records = group[type_1_mask]
                type_neg1_records = group[type_neg1_mask]

                match_count = 0

                # 向量化区间匹配
                for interval in range(len(self.rate_intervals) - 1):
                    t1_in_interval = type_1_records[type_1_records['interval'] == interval]
                    tn1_in_interval = type_neg1_records[type_neg1_records['interval'] == interval]

                    if len(t1_in_interval) > 0 and len(tn1_in_interval) > 0:
                        # 使用numpy meshgrid进行向量化匹配
                        idx1_array = t1_in_interval.index.values
                        idx2_array = tn1_in_interval.index.values

                        idx1_mesh, idx2_mesh = np.meshgrid(idx1_array, idx2_array)
                        for i1, i2 in zip(idx1_mesh.ravel(), idx2_mesh.ravel()):
                            matches.append((i1, i2, 'mode1', region, season))
                            match_count += 1

                pbar.set_postfix({"格网": f"{grid_id}", "季节": season, "找到": match_count})
                pbar.update(1)

        return matches

    def vectorized_matching_mode2(self, region_data: pd.DataFrame, region: str, target_seasons: Optional[List[str]] = None) -> List[Tuple]:
        """
        向量化模式2匹配（不同季节相同变化）- 高性能版本
        """
        matches = []

        # 如果指定了目标季节，过滤数据
        if target_seasons and len(target_seasons) >= 2:
            region_data = region_data[region_data['Season'].isin(target_seasons)].copy()
            if len(region_data) == 0:
                self.logger.warning(f"⚠️ 在{region}区域中未找到指定季节的数据")
                return matches

        # 预计算区间
        rates = region_data['LAI_Change_Rate_mean'].values
        intervals = self.get_rate_interval_vectorized(rates, use_abs=False)
        region_data = region_data.copy()
        region_data['interval'] = intervals

        # 按grid_id分组
        grouped = region_data.groupby('grid_id', observed=True)

        with tqdm(total=len(grouped),
                  desc=f"🔄 模式2-不同季节相同变化({region})",
                  unit="格网") as pbar:

            for grid_id, group in grouped:
                if len(group) < 2:
                    pbar.update(1)
                    continue

                match_count = 0

                # 按grid_type再分组
                for grid_type in [1, -1]:
                    same_type_records = group[group['grid_type'] == grid_type]

                    if len(same_type_records) < 2:
                        continue

                    # 按季节分组
                    season_groups = same_type_records.groupby('Season', observed=True)
                    seasons_list = list(season_groups)

                    # 如果指定了目标季节，只处理这些季节
                    if target_seasons:
                        seasons_list = [(s, g) for s, g in seasons_list if s in target_seasons]

                    # 不同季节间的向量化匹配
                    for i in range(len(seasons_list)):
                        for j in range(i + 1, len(seasons_list)):
                            season1, group1 = seasons_list[i]
                            season2, group2 = seasons_list[j]

                            # 向量化区间匹配
                            for interval in range(len(self.rate_intervals) - 1):
                                g1_interval = group1[group1['interval'] == interval]
                                g2_interval = group2[group2['interval'] == interval]

                                if len(g1_interval) > 0 and len(g2_interval) > 0:
                                    idx1_array = g1_interval.index.values
                                    idx2_array = g2_interval.index.values

                                    idx1_mesh, idx2_mesh = np.meshgrid(idx1_array, idx2_array)
                                    for i1, i2 in zip(idx1_mesh.ravel(), idx2_mesh.ravel()):
                                        matches.append((i1, i2, 'mode2', region, season1, season2))
                                        match_count += 1

                pbar.set_postfix({"格网": f"{grid_id}", "找到": match_count})
                pbar.update(1)

        return matches

    def vectorized_matching_mode3(self, region_data: pd.DataFrame, region: str, target_season: Optional[str] = None) -> List[Tuple]:
        """
        向量化模式3匹配（相同季节相同变化）- 高性能版本
        """
        matches = []

        # 如果指定了目标季节，过滤数据
        if target_season:
            region_data = region_data[region_data['Season'] == target_season].copy()
            if len(region_data) == 0:
                self.logger.warning(f"⚠️ 在{region}区域中未找到{target_season}季节的数据")
                return matches

        # 预计算区间
        rates = region_data['LAI_Change_Rate_mean'].values
        intervals = self.get_rate_interval_vectorized(rates, use_abs=False)
        region_data = region_data.copy()
        region_data['interval'] = intervals

        # 按(grid_id, Season)分组
        grouped = region_data.groupby(['grid_id', 'Season'], observed=True)

        with tqdm(total=len(grouped),
                  desc=f"🔄 模式3-相同季节相同变化({region})",
                  unit="格网组") as pbar:

            for (grid_id, season), group in grouped:
                if len(group) < 2:
                    pbar.update(1)
                    continue

                match_count = 0

                # 按grid_type再分组
                for grid_type in [1, -1]:
                    same_type_records = group[group['grid_type'] == grid_type]

                    if len(same_type_records) < 2:
                        continue

                    # 在相同区间内进行向量化两两匹配
                    for interval in range(len(self.rate_intervals) - 1):
                        interval_records = same_type_records[same_type_records['interval'] == interval]

                        if len(interval_records) >= 2:
                            # 使用combinations进行高效的两两组合
                            indices = interval_records.index.values
                            for i1, i2 in combinations(indices, 2):
                                matches.append((i1, i2, 'mode3', region, season))
                                match_count += 1

                pbar.set_postfix({"格网": f"{grid_id}", "季节": season, "找到": match_count})
                pbar.update(1)

        return matches

    def vectorized_matching_mode4(self, region_data: pd.DataFrame, region: str, target_seasons: Optional[List[str]] = None) -> List[Tuple]:
        """
        向量化模式4匹配（不同季节相反变化）- 高性能版本
        """
        matches = []

        # 如果指定了目标季节，过滤数据
        if target_seasons and len(target_seasons) >= 2:
            region_data = region_data[region_data['Season'].isin(target_seasons)].copy()
            if len(region_data) == 0:
                self.logger.warning(f"⚠️ 在{region}区域中未找到指定季节的数据")
                return matches

        # 预计算区间
        rates = region_data['LAI_Change_Rate_mean'].values
        intervals = self.get_rate_interval_vectorized(rates, use_abs=True)
        region_data = region_data.copy()
        region_data['interval'] = intervals

        # 按grid_id分组
        grouped = region_data.groupby('grid_id', observed=True)

        with tqdm(total=len(grouped),
                  desc=f"🔄 模式4-不同季节相反变化({region})",
                  unit="格网") as pbar:

            for grid_id, group in grouped:
                if len(group) < 2:
                    pbar.update(1)
                    continue

                # 分离不同grid_type的记录
                type_1_records = group[group['grid_type'] == 1]
                type_neg1_records = group[group['grid_type'] == -1]

                if len(type_1_records) == 0 or len(type_neg1_records) == 0:
                    pbar.update(1)
                    continue

                match_count = 0

                # 按季节分组
                type_1_by_season = type_1_records.groupby('Season', observed=True)
                type_neg1_by_season = type_neg1_records.groupby('Season', observed=True)

                # 如果指定了目标季节，只处理这些季节
                if target_seasons:
                    type_1_by_season = {s: g for s, g in type_1_by_season if s in target_seasons}
                    type_neg1_by_season = {s: g for s, g in type_neg1_by_season if s in target_seasons}

                # 不同季节间的向量化匹配
                for season1, group1 in type_1_by_season:
                    for season2, group2 in type_neg1_by_season:
                        if season1 == season2:
                            continue

                        # 向量化区间匹配
                        for interval in range(len(self.rate_intervals) - 1):
                            g1_interval = group1[group1['interval'] == interval]
                            g2_interval = group2[group2['interval'] == interval]

                            if len(g1_interval) > 0 and len(g2_interval) > 0:
                                idx1_array = g1_interval.index.values
                                idx2_array = g2_interval.index.values

                                idx1_mesh, idx2_mesh = np.meshgrid(idx1_array, idx2_array)
                                for i1, i2 in zip(idx1_mesh.ravel(), idx2_mesh.ravel()):
                                    matches.append((i1, i2, 'mode4', region, season1, season2))
                                    match_count += 1

                pbar.set_postfix({"格网": f"{grid_id}", "找到": match_count})
                pbar.update(1)

        return matches

    def _build_match_records_optimized(self, matches: List[Tuple], global_data: pd.DataFrame, mode: str) -> List[Dict]:
        """
        优化版构建匹配记录，减少类型转换开销
        """
        match_records = []

        for match_tuple in matches:
            try:
                if len(match_tuple) == 5:  # 相同季节
                    idx1, idx2, mode_name, region, season = match_tuple
                    season1 = season2 = season
                    season_type = "same_season"
                    matched_season = season
                else:  # 不同季节
                    idx1, idx2, mode_name, region, season1, season2 = match_tuple
                    season_type = "diff_season"
                    matched_season = None

                # 获取完整记录
                record1 = global_data.iloc[idx1]
                record2 = global_data.iloc[idx2]

                # 构建完整的匹配记录
                match_record = {}

                # 获取所有原始数据字段（排除我们添加的辅助字段）
                exclude_fields = {'interval', 'source_file', 'season_from_path', 'Season'}
                original_fields = [col for col in record1.index if col not in exclude_fields]

                # 快速复制字段，减少类型转换开销
                for field in original_fields:
                    # 直接复制，减少类型转换开销
                    value1 = record1[field]
                    value2 = record2[field]

                    # 处理category类型
                    if hasattr(value1, 'item'):
                        value1 = value1.item()
                    if hasattr(value2, 'item'):
                        value2 = value2.item()

                    match_record[f"{field}_1"] = value1
                    match_record[f"{field}_2"] = value2

                # 计算LAI变化率区间信息
                rate1 = float(record1['LAI_Change_Rate_mean'])
                rate2 = float(record2['LAI_Change_Rate_mean'])
                use_abs = mode in ['mode1', 'mode4']

                # 计算区间
                interval1 = self.get_rate_interval_vectorized(np.array([rate1]), use_abs)[0]
                interval2 = self.get_rate_interval_vectorized(np.array([rate2]), use_abs)[0]

                # 生成区间范围描述
                def get_range_description(interval_idx, use_abs):
                    if interval_idx >= len(self.rate_intervals) - 1:
                        interval_idx = len(self.rate_intervals) - 2

                    min_val = self.rate_intervals[interval_idx]
                    max_val = self.rate_intervals[interval_idx + 1]

                    if max_val == np.inf:
                        return f"[{min_val}, +∞)"
                    else:
                        return f"[{min_val}, {max_val})"

                # 添加匹配分析信息
                match_record.update({
                    # 基本匹配信息
                    'matching_mode': mode_name,
                    'matching_region': region,
                    'season_type': season_type,
                    'matched_season': matched_season,
                    'season1': season1,
                    'season2': season2,

                    # 年份对信息
                    'year_pair_1': str(record1.get('year_pair', '')),
                    'year_pair_2': str(record2.get('year_pair', '')),

                    # LAI变化率区间信息
                    'rate_interval_type': 'absolute' if use_abs else 'signed',
                    'rate_interval_1': int(interval1),
                    'rate_interval_2': int(interval2),
                    'rate_range_1': get_range_description(interval1, use_abs),
                    'rate_range_2': get_range_description(interval2, use_abs),

                    # 数值分析信息
                    'lai_change_rate_1': rate1,
                    'lai_change_rate_2': rate2,
                    'rate_diff': abs(rate1 - rate2),
                    'grid_type_1': int(record1['grid_type']),
                    'grid_type_2': int(record2['grid_type'])
                })

                match_records.append(match_record)

            except Exception as e:
                self.logger.warning(f"⚠️ 构建匹配记录失败: {e}")
                continue

        return match_records

    def group_matches_by_season(self, matches: List[Tuple], mode: str) -> Dict[str, List[Tuple]]:
        """
        按季节分组匹配结果
        """
        season_groups = defaultdict(list)

        for match_tuple in matches:
            if len(match_tuple) == 5:  # 相同季节匹配：(idx1, idx2, mode_name, region, season)
                season_key = match_tuple[4]  # season
            else:  # 不同季节匹配：(idx1, idx2, mode_name, region, season1, season2)
                season1, season2 = match_tuple[4], match_tuple[5]
                # 创建季节对的键，按字母顺序排序确保一致性
                season_key = "_".join(sorted([season1, season2]))

            season_groups[season_key].append(match_tuple)

        return dict(season_groups)

    def export_matches_by_season(self, matches: List[Tuple], global_data: pd.DataFrame,
                                mode: str, region: str) -> int:
        """
        按季节分组导出匹配结果
        """
        if not matches:
            self.logger.warning(f"⚠️ 没有找到匹配结果")
            return 0

        self.logger.info(f"💾 开始按季节分组导出 {mode} 匹配结果...")

        # 创建区域目录
        region_dir = self.output_dir / region
        region_dir.mkdir(exist_ok=True)

        # 按季节分组
        season_groups = self.group_matches_by_season(matches, mode)

        total_exported = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for season_key, season_matches in season_groups.items():
            if not season_matches:
                continue

            # 生成文件名
            filename = f"{season_key}_{mode}_{timestamp}.csv"
            output_file = region_dir / filename

            # 构建该季节组的匹配记录
            match_records = self._build_match_records_optimized(season_matches, global_data, mode)

            if match_records:
                # 写入文件
                df = pd.DataFrame(match_records)
                df.to_csv(output_file, index=False, float_format='%.6f')

                count = len(match_records)
                total_exported += count
                self.logger.info(f"✅ 导出 {season_key} 季节 {count} 对匹配到: {output_file}")
            else:
                self.logger.warning(f"⚠️ {season_key} 季节没有有效的匹配记录")

        self.logger.info(f"🎉 总共导出 {total_exported} 对匹配结果，分布在 {len(season_groups)} 个季节文件中")
        return total_exported

    def run_specific_matching(self, global_data: pd.DataFrame, target_region: str,
                            target_mode: str, season_param: Optional[str] = None) -> int:
        """
        执行特定的匹配分析
        """
        self.logger.info(f"🎯 开始执行特定匹配分析...")
        self.logger.info(f"   目标区域: {target_region}")
        self.logger.info(f"   匹配模式: {target_mode}")
        self.logger.info(f"   季节参数: {season_param}")

        # 过滤目标区域数据
        region_mask = global_data['region'] == target_region
        region_data = global_data[region_mask].copy()

        if len(region_data) == 0:
            self.logger.error(f"❌ 在区域 {target_region} 中未找到数据")
            return 0

        self.logger.info(f"🌲 目标区域数据量: {len(region_data):,} 条记录")

        # 匹配函数映射
        matching_functions = {
            'mode1': self.vectorized_matching_mode1,
            'mode2': self.vectorized_matching_mode2,
            'mode3': self.vectorized_matching_mode3,
            'mode4': self.vectorized_matching_mode4
        }

        if target_mode not in matching_functions:
            self.logger.error(f"❌ 不支持的匹配模式: {target_mode}")
            return 0

        match_func = matching_functions[target_mode]

        # 根据不同模式处理季节参数
        try:
            if target_mode in ['mode1', 'mode3']:  # 相同季节匹配
                matches = match_func(region_data, target_region, season_param)
            elif target_mode in ['mode2', 'mode4']:  # 不同季节匹配
                if season_param and ',' in season_param:
                    target_seasons = [s.strip() for s in season_param.split(',')]
                    matches = match_func(region_data, target_region, target_seasons)
                else:
                    matches = match_func(region_data, target_region)
            else:
                matches = match_func(region_data, target_region)

            # 按季节分组导出结果
            match_count = self.export_matches_by_season(matches, global_data, target_mode, target_region)

            self.logger.info(f"✅ 匹配分析完成，找到 {match_count} 对匹配")
            return match_count

        except Exception as e:
            self.logger.error(f"❌ 匹配过程出错: {e}")
            import traceback
            traceback.print_exc()
            return 0


def get_user_input(available_regions: List[str], available_seasons: List[str]) -> Tuple[str, str, Optional[str]]:
    """
    获取用户输入参数
    """
    print("\n" + "="*60)
    print("🌍 LAI格网匹配分析器 - 用户定制版 (按季节分别导出)")
    print("="*60)

    # 显示可用区域
    print(f"\n📍 可用的生态区域 ({len(available_regions)} 个):")
    for i, region in enumerate(available_regions, 1):
        print(f"   {i}. {region}")

    # 选择区域
    while True:
        try:
            region_choice = input(f"\n请选择区域 (1-{len(available_regions)}): ").strip()
            region_idx = int(region_choice) - 1
            if 0 <= region_idx < len(available_regions):
                target_region = available_regions[region_idx]
                break
            else:
                print("❌ 无效选择，请重新输入")
        except ValueError:
            print("❌ 请输入有效数字")

    # 显示匹配模式
    modes = {
        1: ("mode1", "相同季节内的相反变化匹配"),
        2: ("mode2", "不同季节内的相同变化匹配"),
        3: ("mode3", "相同季节内的相同变化匹配"),
        4: ("mode4", "不同季节内的相反变化匹配")
    }

    print(f"\n🔍 可用的匹配模式:")
    for num, (mode_code, description) in modes.items():
        print(f"   {num}. {mode_code}: {description}")

    # 选择模式
    while True:
        try:
            mode_choice = input(f"\n请选择匹配模式 (1-4): ").strip()
            mode_idx = int(mode_choice)
            if mode_idx in modes:
                target_mode, mode_desc = modes[mode_idx]
                break
            else:
                print("❌ 无效选择，请重新输入")
        except ValueError:
            print("❌ 请输入有效数字")

    # 季节参数输入
    print(f"\n🌿 可用的季节: {', '.join(available_seasons)}")

    season_param = None
    if target_mode in ['mode1', 'mode3']:  # 相同季节匹配
        print(f"\n⚠️  {target_mode} 是相同季节匹配模式")
        print("📋 注意：结果将按每个季节分别导出到不同的CSV文件")
        season_input = input("请输入目标季节 (留空表示所有季节): ").strip()
        if season_input and season_input in available_seasons:
            season_param = season_input
        elif season_input:
            print(f"⚠️ 季节 '{season_input}' 不存在，将使用所有季节")

    elif target_mode in ['mode2', 'mode4']:  # 不同季节匹配
        print(f"\n⚠️  {target_mode} 是不同季节匹配模式")
        print("📋 注意：结果将按季节对分别导出到不同的CSV文件 (如: Spring_Summer_mode2.csv)")
        season_input = input("请输入目标季节组合 (用逗号分隔，如 Spring,Summer，留空表示所有季节): ").strip()
        if season_input:
            seasons = [s.strip() for s in season_input.split(',')]
            valid_seasons = [s for s in seasons if s in available_seasons]
            if len(valid_seasons) >= 2:
                season_param = ','.join(valid_seasons)
            else:
                print(f"⚠️ 需要至少2个有效季节，将使用所有季节")

    # 确认参数
    print(f"\n✅ 参数确认:")
    print(f"   目标区域: {target_region}")
    print(f"   匹配模式: {target_mode} ({mode_desc})")
    print(f"   季节参数: {season_param if season_param else '所有季节'}")
    print(f"   导出方式: 按季节分别导出到不同的CSV文件")

    confirm = input(f"\n确认执行匹配? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 用户取消操作")
        exit(0)

    return target_region, target_mode, season_param


def main():
    """主函数"""
    # 配置路径
    input_directory = r"D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\00_ori"  # 数据根目录路径
    output_directory = r"D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\01_matching_results"  # 输出目录

    try:
        # 创建优化版分析器实例
        analyzer = OptimizedGridMatchingAnalyzer(
            input_dir=input_directory,
            output_dir=output_directory,
            chunk_size=50000  # 可根据内存情况调整：30000(8GB), 50000(16GB), 100000(32GB+)
        )

        # 加载数据
        global_data = analyzer.load_all_data_parallel()

        # 获取可用的区域和季节
        available_regions = sorted(analyzer.stats['regions'])
        available_seasons = sorted(analyzer.stats['seasons'])

        # 获取用户输入
        target_region, target_mode, season_param = get_user_input(available_regions, available_seasons)

        # 执行特定匹配分析
        match_count = analyzer.run_specific_matching(global_data, target_region, target_mode, season_param)

        print(f"\n🎉 匹配分析完成！")
        print(f"📊 总共找到 {match_count} 对匹配")
        print(f"📁 结果文件按季节分别保存在: {analyzer.output_dir}/{target_region}/")

    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()