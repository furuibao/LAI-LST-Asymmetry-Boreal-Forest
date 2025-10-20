"""优化的反距离权重(IDW)处理模块 - 修复版本(包含Begin值)"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy.spatial.distance import cdist

from constants import ChangeType, MIN_DISTANCE_M

logger = logging.getLogger(__name__)


class OptimizedIDWProcessor:
    """优化的反距离权重处理器"""

    def __init__(self):
        """初始化IDW处理器"""
        self.min_distance = MIN_DISTANCE_M

    def process_incremental_analysis(self,
                                    grid_types: pd.DataFrame,
                                    stats_data: pd.DataFrame,
                                    variable_group: str,
                                    exclude_zero: bool = False) -> pd.DataFrame:
        """
        对单个变量组进行增量IDW分析
        *** 修复: 同时保留Begin值和IDW分析结果 ***
        """
        logger.info(f"开始{variable_group}变量组的IDW分析...")

        # 验证输入数据
        if len(grid_types) == 0 or len(stats_data) == 0:
            logger.warning(f"{variable_group}变量组输入数据为空")
            return pd.DataFrame()

        # 合并网格分类和统计数据
        merged_data = pd.merge(grid_types, stats_data,
                              on=['large_grid_id', 'grid_id', 'center_x', 'center_y'],
                              how='inner')

        if len(merged_data) == 0:
            logger.warning(f"{variable_group}变量组数据合并失败")
            return pd.DataFrame()

        logger.info(f"合并后数据: {len(merged_data)}个网格")

        # 分离网格类型
        grid_separation = self._separate_grid_types_in_large(merged_data)

        if len(grid_separation) == 0:
            logger.warning(f"{variable_group}变量组没有有效的网格分离结果")
            return pd.DataFrame()

        # 进行IDW分析(仅对变化网格)
        idw_results = self._process_grids_by_large(grid_separation, variable_group, exclude_zero)

        # *** 关键修复: 添加所有网格的Begin值统计 ***
        # 从stats_data中提取Begin值列
        begin_cols = [col for col in stats_data.columns if 'Begin' in col or 'begin' in col]
        end_cols = [col for col in stats_data.columns if 'End' in col or 'end' in col]
        total_cols = [col for col in stats_data.columns if 'total' in col]

        # 需要保留的原始统计列
        preserve_cols = ['large_grid_id', 'grid_id', 'center_x', 'center_y'] + begin_cols + end_cols + total_cols
        available_preserve_cols = [col for col in preserve_cols if col in stats_data.columns]

        # 提取这些列
        stats_subset = stats_data[available_preserve_cols].copy()

        # 合并IDW结果和原始统计值
        if len(idw_results) > 0:
            common_cols = ['large_grid_id', 'grid_id', 'center_x', 'center_y']
            final_results = pd.merge(idw_results, stats_subset, on=common_cols, how='left')
            logger.info(f"{variable_group}变量组IDW分析完成，结果: {len(final_results)}个网格")
            return final_results
        else:
            logger.warning(f"{variable_group}变量组IDW分析无结果")
            return pd.DataFrame()

    def _separate_grid_types_in_large(self, grid_data: pd.DataFrame) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """在每个大网格内分离LAI变化和稳定的小网格"""
        if 'grid_type' not in grid_data.columns or 'large_grid_id' not in grid_data.columns:
            logger.error("grid_data中缺少必要的列")
            return {}

        grid_separation = {}
        unique_large_grids = grid_data['large_grid_id'].unique()

        # 确定实际的变化类型值
        unique_types = grid_data['grid_type'].unique()
        increase_values = [ChangeType.INCREASE.value, '增加', 1]
        decrease_values = [ChangeType.DECREASE.value, '减少', -1]
        stable_values = [ChangeType.NO_CHANGE.value, '稳定', 0]

        actual_increase = [v for v in increase_values if v in unique_types]
        actual_decrease = [v for v in decrease_values if v in unique_types]
        actual_stable = [v for v in stable_values if v in unique_types]

        for large_grid_id in unique_large_grids:
            large_grid_data = grid_data[grid_data['large_grid_id'] == large_grid_id].copy()

            if len(large_grid_data) == 0:
                continue

            # 分离变化和稳定网格
            changed_grids = large_grid_data[
                large_grid_data['grid_type'].isin(actual_increase + actual_decrease)
            ].copy()

            stable_grids = large_grid_data[
                large_grid_data['grid_type'].isin(actual_stable)
            ].copy()

            # 过滤掉有缺失关键数据的网格
            if len(stable_grids) > 0:
                key_columns = [col for col in stable_grids.columns if col.endswith('_mean')]
                if key_columns:
                    stable_grids = stable_grids.dropna(subset=key_columns, how='all')

            # 只保留有效的大网格
            if len(changed_grids) > 0 and len(stable_grids) > 0:
                grid_separation[large_grid_id] = (changed_grids, stable_grids)

        logger.info(f"有效大网格数量: {len(grid_separation)}")
        return grid_separation

    def _process_grids_by_large(self,
                                grid_separation: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
                                variable_group: str,
                                exclude_zero: bool = False) -> pd.DataFrame:
        """按大网格处理IDW分析"""
        results = []
        total_processed = 0
        total_failed = 0

        # 根据变量组确定要处理的变量(仅diff变量)
        if variable_group == 'temperature':
            variables = ['LST_Daily_diff', 'LST_Day_diff', 'LST_Night_diff']
        elif variable_group == 'energy':
            variables = ['Albedo_diff', 'ET_diff', 'LE_diff', 'DSR_diff']
        else:
            logger.error(f"不支持的变量组: {variable_group}")
            return pd.DataFrame()

        with tqdm(total=len(grid_separation), desc=f"处理{variable_group}组") as pbar:
            for large_grid_id, (changed_grids, stable_grids) in grid_separation.items():
                try:
                    if len(changed_grids) == 0 or len(stable_grids) == 0:
                        pbar.update(1)
                        continue

                    # 处理该大网格内的所有变化网格
                    for _, changed_grid in changed_grids.iterrows():
                        result = self._process_single_grid(
                            changed_grid, stable_grids, variables, exclude_zero
                        )

                        if result:
                            results.append(result)
                            total_processed += 1
                        else:
                            total_failed += 1

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"处理大网格 {large_grid_id} 时出错: {e}")
                    total_failed += len(changed_grids) if len(changed_grids) > 0 else 1
                    pbar.update(1)

        logger.info(f"{variable_group}组IDW分析完成: 成功 {total_processed}, 失败 {total_failed}")
        return pd.DataFrame(results)

    def _process_single_grid(self,
                            changed_grid: pd.Series,
                            stable_grids: pd.DataFrame,
                            variables: List[str],
                            exclude_zero: bool = False) -> Dict:
        """处理单个LAI变化网格的分析"""
        if len(stable_grids) == 0:
            return {}

        try:
            # 计算反距离权重
            weighted_stable = self._compute_idw_weights(changed_grid, stable_grids, variables, exclude_zero)

            if len(weighted_stable) == 0:
                return {}

            # 检查权重总和
            total_weight = weighted_stable['weight'].sum()
            if total_weight == 0 or np.isnan(total_weight):
                logger.debug(f"网格 {changed_grid['grid_id']} 权重总和为0")
                return {}

            # 计算气候背景贡献
            climate_contrib = self._calculate_climate_contribution(weighted_stable, variables)

            if not climate_contrib:
                return {}

            # 计算植被贡献
            veg_contrib = self._calculate_vegetation_contribution(changed_grid, climate_contrib, variables)

            # 构建结果
            result = {
                'large_grid_id': changed_grid['large_grid_id'],
                'grid_id': changed_grid['grid_id'],
                'center_x': changed_grid['center_x'],
                'center_y': changed_grid['center_y'],
                'grid_type': changed_grid['grid_type'],
            }

            # 添加气候背景贡献和植被贡献
            result.update({f'climate_{k}': v for k, v in climate_contrib.items()})
            result.update(veg_contrib)

            return result

        except Exception as e:
            logger.error(f"处理单个网格时出错: {e}")
            return {}

    def _compute_idw_weights(self,
                            changed_grid: pd.Series,
                            stable_grids: pd.DataFrame,
                            variables: List[str],
                            exclude_zero: bool = False) -> pd.DataFrame:
        """计算反距离权重"""
        stable_grids = stable_grids.copy()

        # 目标网格坐标
        target_coords = np.array([changed_grid['center_x'], changed_grid['center_y']])

        # 参考网格坐标
        reference_coords = stable_grids[['center_x', 'center_y']].values

        # 计算距离
        target_coords = target_coords.reshape(1, -1)
        distances = cdist(target_coords, reference_coords, metric='euclidean')[0]

        # 计算反距离权重
        inv_distances = 1.0 / np.maximum(distances, self.min_distance)

        # 处理零值排除选项
        if exclude_zero:
            zero_mask = np.ones(len(stable_grids), dtype=bool)
            for var in variables:
                mean_col = f'{var}_mean'
                if mean_col in stable_grids.columns:
                    zero_mask &= (stable_grids[mean_col].abs() < 1e-6)
            inv_distances[zero_mask] = 0

        # 归一化权重
        total_weight = inv_distances.sum()
        if total_weight > 0:
            weights = inv_distances / total_weight
        else:
            logger.debug("权重总和为0,无法归一化")
            weights = np.zeros_like(inv_distances)

        stable_grids['weight'] = weights

        # 计算加权值
        for var in variables:
            mean_col = f'{var}_mean'
            std_col = f'{var}_std'

            if mean_col in stable_grids.columns:
                stable_grids[f'weighted_{var.lower()}'] = stable_grids[mean_col] * weights

            if std_col in stable_grids.columns:
                stable_grids[f'weighted_var_{var.lower()}'] = (stable_grids[std_col] ** 2) * (weights ** 2)

        return stable_grids

    def _calculate_climate_contribution(self,
                                       weighted_stable: pd.DataFrame,
                                       variables: List[str]) -> Dict[str, float]:
        """计算气候背景贡献"""
        if len(weighted_stable) == 0:
            return {}

        climate_contrib = {}

        for var in variables:
            var_lower = var.lower()
            weighted_col = f'weighted_{var_lower}'
            weighted_var_col = f'weighted_var_{var_lower}'

            if weighted_col in weighted_stable.columns:
                climate_value = weighted_stable[weighted_col].sum()
                climate_contrib[var_lower] = climate_value

            if weighted_var_col in weighted_stable.columns:
                climate_std = np.sqrt(weighted_stable[weighted_var_col].sum())
                climate_contrib[f'{var_lower}_std'] = climate_std

        return climate_contrib

    def _calculate_vegetation_contribution(self,
                                          changed_grid: pd.Series,
                                          climate_contrib: Dict[str, float],
                                          variables: List[str]) -> Dict[str, float]:
        """计算植被贡献"""
        if not climate_contrib:
            return {}

        veg_contrib = {}

        for var in variables:
            var_lower = var.lower()
            mean_col = f'{var}_mean'
            std_col = f'{var}_std'

            if mean_col in changed_grid.index:
                total_val = changed_grid[mean_col]
                climate_val = climate_contrib.get(var_lower, 0)

                # 计算植被贡献
                veg_contrib[f'{var_lower}_lai'] = total_val - climate_val
                veg_contrib[f'{var_lower}_total'] = total_val

            if std_col in changed_grid.index:
                total_std = changed_grid[std_col]
                climate_std = climate_contrib.get(f'{var_lower}_std', 0)

                # 计算植被贡献的标准差
                veg_contrib[f'{var_lower}_lai_std'] = np.sqrt(
                    max(0, total_std ** 2 + climate_std ** 2)
                )

        return veg_contrib

    @staticmethod
    def merge_incremental_results(results_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        合并多个增量分析结果
        *** 修复: 保留所有列,避免丢失Begin值 ***
        """
        if not results_list:
            return pd.DataFrame()

        # 过滤空结果
        valid_results = [r for r in results_list if len(r) > 0]

        if not valid_results:
            logger.warning("没有有效的增量分析结果")
            return pd.DataFrame()

        logger.info(f"开始合并{len(valid_results)}个增量分析结果...")

        # 以第一个结果为基础
        merged = valid_results[0].copy()

        # 依次合并其他结果
        for i, result in enumerate(valid_results[1:], 1):
            # 按网格ID合并,使用outer保留所有数据
            common_cols = ['large_grid_id', 'grid_id', 'center_x', 'center_y', 'grid_type']
            # 只保留两个DataFrame都有的common_cols
            actual_common_cols = [col for col in common_cols if col in merged.columns and col in result.columns]

            merged = pd.merge(merged, result, on=actual_common_cols, how='outer', suffixes=('', '_dup'))

            # 删除重复列
            dup_cols = [col for col in merged.columns if col.endswith('_dup')]
            if dup_cols:
                merged = merged.drop(columns=dup_cols)

        logger.info(f"结果合并完成，最终形状: {merged.shape}")
        logger.info(f"输出列: {list(merged.columns)}")

        return merged