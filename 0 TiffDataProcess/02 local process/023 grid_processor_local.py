"""优化的网格处理模块 - 修复版本"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from constants import (
    ChangeType, ChangeRate, PIXEL_SIZE_M, DEFAULT_THRESHOLD,
    DEFAULT_SMALL_GRID_SIZE_KM, DEFAULT_LARGE_GRID_SIZE_KM, DEFAULT_CHUNK_SIZE
)

logger = logging.getLogger(__name__)


@dataclass
class GridInfo:
    """网格信息类"""
    grid_id: int
    center_x: float
    center_y: float
    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass
class ValidGridInfo:
    """有效网格信息,包含统计结果"""
    large_grid_id: int
    small_grids: List[GridInfo]
    pixel_stats: pd.DataFrame
    grid_types: pd.DataFrame


class OptimizedGridProcessor:
    """优化的网格处理器"""

    def __init__(self, data_shape: Tuple[int, int], transform,
                 large_grid_size_km: int = DEFAULT_LARGE_GRID_SIZE_KM,
                 small_grid_size_km: int = DEFAULT_SMALL_GRID_SIZE_KM,
                 chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.height, self.width = data_shape
        self.transform = transform
        self.large_grid_size_km = large_grid_size_km
        self.small_grid_size_km = small_grid_size_km
        self.chunk_size = chunk_size

        # 计算网格像元大小
        self.large_grid_pixels = (large_grid_size_km * 1000) // PIXEL_SIZE_M
        self.small_grid_pixels = (small_grid_size_km * 1000) // PIXEL_SIZE_M

        # 计算网格数量
        self.n_large_grids_y = self.height // self.large_grid_pixels
        self.n_large_grids_x = self.width // self.large_grid_pixels
        self.total_large_grids = self.n_large_grids_y * self.n_large_grids_x

        logger.info(f"网格配置: 大网格{large_grid_size_km}km({self.large_grid_pixels}px), "
                   f"小网格{small_grid_size_km}km({self.small_grid_pixels}px)")
        logger.info(f"数据形状: {self.height}x{self.width}, 大网格数量: {self.total_large_grids}")

    def identify_valid_grids(self, essential_data: Dict[str, np.ndarray]) -> Dict[int, ValidGridInfo]:
        """
        使用必要数据识别和筛选有效网格
        *** 修复: 使用正确的变量名 LAI_Begin 和 LAI_diff ***
        """
        logger.info("开始识别有效网格...")

        # 检查必要的变量是否存在
        if 'LAI_Begin' not in essential_data or 'LAI_diff' not in essential_data:
            logger.error(f"必要数据缺失. 可用变量: {list(essential_data.keys())}")
            return {}

        # 检查数据大小,决定是否使用分块处理
        lai_begin = essential_data['LAI_Begin']
        lai_diff = essential_data['LAI_diff']

        data_size = lai_begin.nbytes + lai_diff.nbytes
        data_size_mb = data_size / (1024 * 1024)

        if data_size_mb > 1024:
            logger.info(f"数据较大({data_size_mb:.1f}MB),使用分块处理模式")
            return self._identify_valid_grids_chunked(essential_data)
        else:
            logger.info(f"数据大小适中({data_size_mb:.1f}MB),使用常规处理模式")
            return self._identify_valid_grids_normal(essential_data)

    def _identify_valid_grids_normal(self, essential_data: Dict[str, np.ndarray]) -> Dict[int, ValidGridInfo]:
        """常规模式识别有效网格"""
        # 首先进行LAI变化分类
        lai_change_type = self._classify_lai_changes(essential_data)

        # 生成大网格
        large_grids = self._generate_large_grids()

        # 筛选有效的大网格
        valid_large_grid_ids = self._filter_valid_large_grids(lai_change_type, large_grids)

        if len(valid_large_grid_ids) == 0:
            logger.warning("没有找到有效的大网格")
            return {}

        logger.info(f"找到{len(valid_large_grid_ids)}个有效大网格")

        # 为每个有效大网格生成小网格并进行基础统计
        valid_grids = {}
        large_grid_dict = {grid.grid_id: grid for grid in large_grids}

        for large_grid_id in tqdm(valid_large_grid_ids, desc="处理有效大网格"):
            large_grid = large_grid_dict[large_grid_id]

            # 在大网格内生成小网格
            small_grids = self._generate_small_grids_in_large_grid(large_grid)

            if len(small_grids) == 0:
                continue

            # 统计小网格像元
            pixel_stats = self._count_pixels_by_category(lai_change_type, small_grids, large_grid_id)

            if len(pixel_stats) == 0:
                continue

            # 对小网格进行分类
            grid_types = self._classify_grid_types(pixel_stats)

            valid_grids[large_grid_id] = ValidGridInfo(
                large_grid_id=large_grid_id,
                small_grids=small_grids,
                pixel_stats=pixel_stats,
                grid_types=grid_types
            )

        logger.info(f"有效网格处理完成,共{len(valid_grids)}个大网格")
        return valid_grids

    def _identify_valid_grids_chunked(self, essential_data: Dict[str, np.ndarray]) -> Dict[int, ValidGridInfo]:
        """分块模式识别有效网格"""
        logger.info("使用分块模式进行LAI变化分类...")

        # 分块进行LAI变化分类
        lai_change_type = self._classify_lai_changes_chunked(essential_data)

        # 生成大网格
        large_grids = self._generate_large_grids()

        # 筛选有效的大网格
        valid_large_grid_ids = self._filter_valid_large_grids_chunked(lai_change_type, large_grids)

        if len(valid_large_grid_ids) == 0:
            logger.warning("没有找到有效的大网格")
            return {}

        logger.info(f"找到{len(valid_large_grid_ids)}个有效大网格")

        # 为每个有效大网格生成小网格并进行基础统计
        valid_grids = {}
        large_grid_dict = {grid.grid_id: grid for grid in large_grids}

        for large_grid_id in tqdm(valid_large_grid_ids, desc="处理有效大网格"):
            large_grid = large_grid_dict[large_grid_id]

            small_grids = self._generate_small_grids_in_large_grid(large_grid)

            if len(small_grids) == 0:
                continue

            # 分块统计小网格像元
            pixel_stats = self._count_pixels_by_category_chunked(lai_change_type, small_grids, large_grid_id)

            if len(pixel_stats) == 0:
                continue

            grid_types = self._classify_grid_types(pixel_stats)

            valid_grids[large_grid_id] = ValidGridInfo(
                large_grid_id=large_grid_id,
                small_grids=small_grids,
                pixel_stats=pixel_stats,
                grid_types=grid_types
            )

        logger.info(f"分块处理完成,共{len(valid_grids)}个大网格")
        return valid_grids

    def _classify_lai_changes_chunked(self, essential_data: Dict[str, np.ndarray]) -> np.ndarray:
        """分块对LAI变化进行分类"""
        lai_begin = essential_data['LAI_Begin']
        lai_diff = essential_data['LAI_diff']

        # 初始化结果数组
        lai_change_type = np.full_like(lai_begin, np.nan)

        # 计算分块数量
        num_chunks_y = int(np.ceil(self.height / self.chunk_size))
        num_chunks_x = int(np.ceil(self.width / self.chunk_size))

        logger.info(f"分块处理LAI变化分类: {num_chunks_y}x{num_chunks_x} = {num_chunks_y * num_chunks_x} 个块")

        # 分块处理
        for i in tqdm(range(num_chunks_y), desc="分块处理LAI分类"):
            for j in range(num_chunks_x):
                # 计算块边界
                row_start = i * self.chunk_size
                row_end = min((i + 1) * self.chunk_size, self.height)
                col_start = j * self.chunk_size
                col_end = min((j + 1) * self.chunk_size, self.width)

                # 提取块数据
                chunk_lai_begin = lai_begin[row_start:row_end, col_start:col_end]
                chunk_lai_diff = lai_diff[row_start:row_end, col_start:col_end]

                # 计算LAI相对变化率
                chunk_lai_change_rate = np.divide(
                    chunk_lai_diff,
                    chunk_lai_begin,
                    out=np.full_like(chunk_lai_diff, np.nan),
                    where=(chunk_lai_begin != 0) & ~np.isnan(chunk_lai_begin) & ~np.isnan(chunk_lai_diff)
                )

                # 分类LAI变化
                chunk_lai_change_type = np.full_like(chunk_lai_change_rate, np.nan)

                # LAI增加
                increase_mask = chunk_lai_change_rate > ChangeRate.LAI_INCREASE_RATE.value
                chunk_lai_change_type[increase_mask] = ChangeType.INCREASE.value

                # LAI减少
                decrease_mask = chunk_lai_change_rate < ChangeRate.LAI_DECREASE_RATE.value
                chunk_lai_change_type[decrease_mask] = ChangeType.DECREASE.value

                # LAI稳定
                stable_mask = (
                    (chunk_lai_change_rate >= ChangeRate.LAI_DECREASE_RATE.value) &
                    (chunk_lai_change_rate <= ChangeRate.LAI_INCREASE_RATE.value)
                )
                chunk_lai_change_type[stable_mask] = ChangeType.NO_CHANGE.value

                # 将结果写回主数组
                lai_change_type[row_start:row_end, col_start:col_end] = chunk_lai_change_type

        logger.info("LAI变化分类完成")
        return lai_change_type

    def _classify_lai_changes(self, essential_data: Dict[str, np.ndarray]) -> np.ndarray:
        """对LAI变化进行分类(原始方法)"""
        lai_begin = essential_data['LAI_Begin']
        lai_diff = essential_data['LAI_diff']

        # 计算LAI相对变化率
        lai_change_rate = np.divide(
            lai_diff,
            lai_begin,
            out=np.full_like(lai_diff, np.nan),
            where=(lai_begin != 0) & ~np.isnan(lai_begin) & ~np.isnan(lai_diff)
        )

        # 分类LAI变化
        lai_change_type = np.full_like(lai_change_rate, np.nan)

        # LAI增加
        increase_mask = lai_change_rate > ChangeRate.LAI_INCREASE_RATE.value
        lai_change_type[increase_mask] = ChangeType.INCREASE.value

        # LAI减少
        decrease_mask = lai_change_rate < ChangeRate.LAI_DECREASE_RATE.value
        lai_change_type[decrease_mask] = ChangeType.DECREASE.value

        # LAI稳定
        stable_mask = (
            (lai_change_rate >= ChangeRate.LAI_DECREASE_RATE.value) &
            (lai_change_rate <= ChangeRate.LAI_INCREASE_RATE.value)
        )
        lai_change_type[stable_mask] = ChangeType.NO_CHANGE.value

        logger.info("LAI变化分类完成")
        return lai_change_type

    def _generate_large_grids(self) -> List[GridInfo]:
        """生成大网格信息列表"""
        grids = []
        grid_id = 0

        for i in range(self.n_large_grids_y):
            for j in range(self.n_large_grids_x):
                row_start = i * self.large_grid_pixels
                row_end = min((i + 1) * self.large_grid_pixels, self.height)
                col_start = j * self.large_grid_pixels
                col_end = min((j + 1) * self.large_grid_pixels, self.width)

                # 计算中心坐标
                center_pixel_x = (col_start + col_end) / 2
                center_pixel_y = (row_start + row_end) / 2
                center_x = self.transform[2] + center_pixel_x * self.transform[0]
                center_y = self.transform[5] + center_pixel_y * self.transform[4]

                grid_info = GridInfo(
                    grid_id=grid_id,
                    center_x=center_x,
                    center_y=center_y,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end
                )
                grids.append(grid_info)
                grid_id += 1

        return grids

    def _filter_valid_large_grids(self, lai_change_type: np.ndarray,
                                  large_grids: List[GridInfo]) -> Set[int]:
        """筛选有效的大网格(原始方法)"""
        valid_grids = set()
        required_changes = {ChangeType.INCREASE.value, ChangeType.DECREASE.value, ChangeType.NO_CHANGE.value}

        for grid in large_grids:
            grid_data = lai_change_type[grid.row_start:grid.row_end, grid.col_start:grid.col_end]
            valid_lai_values = grid_data[~np.isnan(grid_data)]

            if len(valid_lai_values) == 0:
                continue

            unique_changes = set(valid_lai_values)

            # 检查是否包含所有三种变化类型
            if required_changes.issubset(unique_changes):
                valid_grids.add(grid.grid_id)

        return valid_grids

    def _filter_valid_large_grids_chunked(self, lai_change_type: np.ndarray,
                                         large_grids: List[GridInfo]) -> Set[int]:
        """分块筛选有效的大网格"""
        valid_grids = set()

        logger.info("分块筛选有效大网格...")

        for grid in tqdm(large_grids, desc="筛选大网格"):
            grid_height = grid.row_end - grid.row_start
            grid_width = grid.col_end - grid.col_start

            # 如果大网格本身不太大,直接处理
            if grid_height * grid_width < self.chunk_size * self.chunk_size:
                grid_data = lai_change_type[grid.row_start:grid.row_end, grid.col_start:grid.col_end]
                valid_lai_values = grid_data[~np.isnan(grid_data)]

                if len(valid_lai_values) == 0:
                    continue

                unique_changes = set(valid_lai_values)
                required_changes = {ChangeType.INCREASE.value, ChangeType.DECREASE.value, ChangeType.NO_CHANGE.value}

                if required_changes.issubset(unique_changes):
                    valid_grids.add(grid.grid_id)
            else:
                # 对大网格进行分块检查
                if self._check_large_grid_validity_chunked(lai_change_type, grid):
                    valid_grids.add(grid.grid_id)

        return valid_grids

    def _check_large_grid_validity_chunked(self, lai_change_type: np.ndarray,
                                          grid: GridInfo) -> bool:
        """分块检查大网格的有效性"""
        found_changes = set()
        required_changes = {ChangeType.INCREASE.value, ChangeType.DECREASE.value, ChangeType.NO_CHANGE.value}

        # 在大网格内分块检查
        grid_chunk_size = min(self.chunk_size, grid.row_end - grid.row_start, grid.col_end - grid.col_start)

        for i in range(grid.row_start, grid.row_end, grid_chunk_size):
            for j in range(grid.col_start, grid.col_end, grid_chunk_size):
                chunk_row_end = min(i + grid_chunk_size, grid.row_end)
                chunk_col_end = min(j + grid_chunk_size, grid.col_end)

                chunk_data = lai_change_type[i:chunk_row_end, j:chunk_col_end]
                valid_values = chunk_data[~np.isnan(chunk_data)]

                if len(valid_values) > 0:
                    found_changes.update(valid_values)

                    # 如果已经找到所有需要的变化类型,提前结束
                    if required_changes.issubset(found_changes):
                        return True

        return False

    def _generate_small_grids_in_large_grid(self, large_grid: GridInfo) -> List[GridInfo]:
        """在单个大网格内生成小网格"""
        large_height = large_grid.row_end - large_grid.row_start
        large_width = large_grid.col_end - large_grid.col_start

        n_small_y = large_height // self.small_grid_pixels
        n_small_x = large_width // self.small_grid_pixels

        small_grids = []
        small_grid_id = 0

        for i in range(n_small_y):
            for j in range(n_small_x):
                row_start = large_grid.row_start + i * self.small_grid_pixels
                row_end = min(row_start + self.small_grid_pixels, large_grid.row_end)
                col_start = large_grid.col_start + j * self.small_grid_pixels
                col_end = min(col_start + self.small_grid_pixels, large_grid.col_end)

                center_pixel_x = (col_start + col_end) / 2
                center_pixel_y = (row_start + row_end) / 2
                center_x = self.transform[2] + center_pixel_x * self.transform[0]
                center_y = self.transform[5] + center_pixel_y * self.transform[4]

                # 使用字符串ID避免冲突
                combined_grid_id = f"{large_grid.grid_id}_{small_grid_id}"

                grid_info = GridInfo(
                    grid_id=combined_grid_id,
                    center_x=center_x,
                    center_y=center_y,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end
                )
                small_grids.append(grid_info)
                small_grid_id += 1

        return small_grids

    def _count_pixels_by_category(self, lai_change_type: np.ndarray,
                                  small_grids: List[GridInfo],
                                  large_grid_id: int) -> pd.DataFrame:
        """统计小网格中各类别的像元数量(原始方法)"""
        results = []

        for grid in small_grids:
            grid_data = lai_change_type[grid.row_start:grid.row_end, grid.col_start:grid.col_end]
            valid_mask = ~np.isnan(grid_data)
            total_count = np.sum(valid_mask)

            if total_count == 0:
                continue

            increase_count = np.sum((grid_data == ChangeType.INCREASE.value) & valid_mask)
            decrease_count = np.sum((grid_data == ChangeType.DECREASE.value) & valid_mask)
            stable_count = np.sum((grid_data == ChangeType.NO_CHANGE.value) & valid_mask)

            increase_rate = increase_count / total_count if total_count > 0 else 0
            decrease_rate = decrease_count / total_count if total_count > 0 else 0
            stable_rate = stable_count / total_count if total_count > 0 else 0

            results.append({
                "large_grid_id": large_grid_id,
                "grid_id": grid.grid_id,
                "center_x": grid.center_x,
                "center_y": grid.center_y,
                "total_count": total_count,
                "increase_count": increase_count,
                "decrease_count": decrease_count,
                "stable_count": stable_count,
                "increase_rate": increase_rate,
                "decrease_rate": decrease_rate,
                "stable_rate": stable_rate
            })

        return pd.DataFrame(results)

    def _count_pixels_by_category_chunked(self, lai_change_type: np.ndarray,
                                         small_grids: List[GridInfo],
                                         large_grid_id: int) -> pd.DataFrame:
        """分块统计小网格中各类别的像元数量"""
        # 对于小网格统计,通常不需要分块
        # 这里保持与原始方法一致
        return self._count_pixels_by_category(lai_change_type, small_grids, large_grid_id)

    def _classify_grid_types(self, grid_stats: pd.DataFrame,
                            threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
        """根据占比阈值对网格进行分类"""
        def determine_grid_type(row):
            if row['increase_rate'] >= threshold:
                return ChangeType.INCREASE.value
            elif row['decrease_rate'] >= threshold:
                return ChangeType.DECREASE.value
            elif row['stable_rate'] >= threshold:
                return ChangeType.NO_CHANGE.value
            else:
                return 9999  # 混合类型

        grid_stats = grid_stats.copy()
        grid_stats['grid_type'] = grid_stats.apply(determine_grid_type, axis=1)
        return grid_stats

    def compute_incremental_statistics(self, data_dict: Dict[str, np.ndarray],
                                      valid_grids: Dict[int, ValidGridInfo],
                                      variable_names: List[str]) -> pd.DataFrame:
        """对有效网格进行增量统计计算"""
        # 检查数据大小,决定是否使用分块处理
        total_data_size = sum(data_dict[var].nbytes for var in variable_names if var in data_dict)
        data_size_mb = total_data_size / (1024 * 1024)

        if data_size_mb > 512:
            logger.info(f"数据较大({data_size_mb:.1f}MB),使用分块统计计算")
            return self._compute_incremental_statistics_chunked(data_dict, valid_grids, variable_names)
        else:
            logger.info(f"数据大小适中({data_size_mb:.1f}MB),使用常规统计计算")
            return self._compute_incremental_statistics_normal(data_dict, valid_grids, variable_names)

    def _compute_incremental_statistics_normal(self, data_dict: Dict[str, np.ndarray],
                                              valid_grids: Dict[int, ValidGridInfo],
                                              variable_names: List[str]) -> pd.DataFrame:
        """常规模式增量统计计算"""
        results = []

        logger.info(f"开始增量统计计算,变量: {variable_names}")

        for large_grid_id, grid_info in tqdm(valid_grids.items(), desc="增量统计"):
            for grid in grid_info.small_grids:
                try:
                    grid_result = {
                        'large_grid_id': large_grid_id,
                        'grid_id': grid.grid_id,
                        'center_x': grid.center_x,
                        'center_y': grid.center_y
                    }

                    # 提取网格数据并计算统计值
                    for var_name in variable_names:
                        if var_name in data_dict:
                            grid_data = data_dict[var_name][grid.row_start:grid.row_end,
                                                           grid.col_start:grid.col_end]
                            valid_mask = ~np.isnan(grid_data)
                            valid_data = grid_data[valid_mask]

                            if len(valid_data) > 0:
                                grid_result[f'{var_name}_mean'] = np.mean(valid_data)
                                grid_result[f'{var_name}_std'] = np.std(valid_data)
                            else:
                                grid_result[f'{var_name}_mean'] = np.nan
                                grid_result[f'{var_name}_std'] = np.nan
                        else:
                            grid_result[f'{var_name}_mean'] = np.nan
                            grid_result[f'{var_name}_std'] = np.nan

                    results.append(grid_result)

                except Exception as e:
                    logger.error(f"计算网格 {grid.grid_id} 统计值时出错: {e}")
                    continue

        return pd.DataFrame(results)

    def _compute_incremental_statistics_chunked(self, data_dict: Dict[str, np.ndarray],
                                               valid_grids: Dict[int, ValidGridInfo],
                                               variable_names: List[str]) -> pd.DataFrame:
        """分块模式增量统计计算"""
        results = []

        logger.info(f"开始分块增量统计计算,变量: {variable_names}")

        for large_grid_id, grid_info in tqdm(valid_grids.items(), desc="分块增量统计"):
            for grid in grid_info.small_grids:
                try:
                    grid_result = {
                        'large_grid_id': large_grid_id,
                        'grid_id': grid.grid_id,
                        'center_x': grid.center_x,
                        'center_y': grid.center_y
                    }

                    # 逐个变量处理以减少内存使用
                    for var_name in variable_names:
                        if var_name in data_dict:
                            grid_stats = self._compute_grid_statistics_chunked(
                                data_dict[var_name], grid
                            )
                            grid_result[f'{var_name}_mean'] = grid_stats['mean']
                            grid_result[f'{var_name}_std'] = grid_stats['std']
                        else:
                            grid_result[f'{var_name}_mean'] = np.nan
                            grid_result[f'{var_name}_std'] = np.nan

                    results.append(grid_result)

                except Exception as e:
                    logger.error(f"计算网格 {grid.grid_id} 统计值时出错: {e}")
                    continue

        return pd.DataFrame(results)

    def _compute_grid_statistics_chunked(self, data_array: np.ndarray,
                                        grid: GridInfo) -> Dict[str, float]:
        """分块计算单个网格的统计值"""
        grid_height = grid.row_end - grid.row_start
        grid_width = grid.col_end - grid.col_start

        # 如果网格较小,直接计算
        if grid_height * grid_width < self.chunk_size * self.chunk_size:
            grid_data = data_array[grid.row_start:grid.row_end, grid.col_start:grid.col_end]
            valid_mask = ~np.isnan(grid_data)
            valid_data = grid_data[valid_mask]

            if len(valid_data) > 0:
                return {
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data)
                }
            else:
                return {'mean': np.nan, 'std': np.nan}

        # 对大网格进行分块计算
        valid_values = []
        chunk_size = min(self.chunk_size, grid_height, grid_width)

        for i in range(grid.row_start, grid.row_end, chunk_size):
            for j in range(grid.col_start, grid.col_end, chunk_size):
                chunk_row_end = min(i + chunk_size, grid.row_end)
                chunk_col_end = min(j + chunk_size, grid.col_end)

                chunk_data = data_array[i:chunk_row_end, j:chunk_col_end]
                valid_mask = ~np.isnan(chunk_data)
                chunk_valid_data = chunk_data[valid_mask]

                if len(chunk_valid_data) > 0:
                    valid_values.extend(chunk_valid_data.flatten())

        if len(valid_values) > 0:
            valid_array = np.array(valid_values)
            return {
                'mean': np.mean(valid_array),
                'std': np.std(valid_array)
            }
        else:
            return {'mean': np.nan, 'std': np.nan}
