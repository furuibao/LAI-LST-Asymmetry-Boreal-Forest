"""优化的数据加载模块 - 修复版本"""

import os
import logging
import rasterio
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Generator
from pathlib import Path
import math

from constants import (
    CATEGORIES, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
    MAX_MEMORY_MB, BAND_NAME_MAPPING, NODATA_VALUE
)

logger = logging.getLogger(__name__)


class ChunkedDataLoader:
    """分块数据加载器"""

    def __init__(self, root_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_OVERLAP):
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_file_chunks(self, file_path: Path) -> Generator:
        """分块读取文件"""
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return

        try:
            with rasterio.open(file_path) as src:
                height, width = src.height, src.width
                transform = src.transform
                crs = src.crs
                count = src.count

                # 计算分块数量
                num_chunks_y = math.ceil(height / self.chunk_size)
                num_chunks_x = math.ceil(width / self.chunk_size)
                total_chunks = num_chunks_y * num_chunks_x

                logger.info(f"文件 {file_path.name} 将被分为 {num_chunks_y}x{num_chunks_x} = {total_chunks} 个块")

                for i in range(num_chunks_y):
                    for j in range(num_chunks_x):
                        # 计算当前块的边界(包含overlap)
                        row_start = max(0, i * self.chunk_size - self.overlap)
                        row_end = min((i + 1) * self.chunk_size + self.overlap, height)
                        col_start = max(0, j * self.chunk_size - self.overlap)
                        col_end = min((j + 1) * self.chunk_size + self.overlap, width)

                        # 创建窗口
                        window = rasterio.windows.Window(
                            col_start, row_start,
                            col_end - col_start, row_end - row_start
                        )

                        try:
                            data_chunk = src.read(window=window)

                            # 检查内存使用
                            chunk_memory_mb = data_chunk.nbytes / (1024 * 1024)
                            if chunk_memory_mb > MAX_MEMORY_MB:
                                logger.warning(f"数据块内存使用过大: {chunk_memory_mb:.1f}MB")

                            # 创建数据字典
                            chunk_dict = {
                                'data': data_chunk,
                                'transform': transform,
                                'crs': crs,
                                'chunk_bounds': (row_start, row_end, col_start, col_end),
                                'chunk_id': i * num_chunks_x + j,
                                'total_chunks': total_chunks,
                                'count': count
                            }

                            yield chunk_dict, (row_start, row_end, col_start, col_end)

                        except Exception as e:
                            logger.error(f"读取数据块失败 [{i},{j}]: {e}")
                            continue

        except Exception as e:
            logger.error(f"打开文件失败: {file_path}, 错误: {e}")


class OptimizedDataLoader:
    """优化的数据加载类"""

    def __init__(self, root_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_OVERLAP):
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunked_loader = ChunkedDataLoader(root_dir, chunk_size, overlap)

    def _get_file_path(self, region: str, year_pair: str, season: str, category: str) -> Path:
        """构建文件路径"""
        return self.root_dir / region / year_pair / season / f"{category}.tif"

    def _apply_scale_factors(self, data: np.ndarray, category: str) -> Dict[str, np.ndarray]:
        """
        应用缩放因子还原真实数值,并映射波段名称
        """
        bands = CATEGORIES[category]["bands"]
        scale_factors = CATEGORIES[category]["scale_factors"]

        if len(bands) != data.shape[0]:
            logger.warning(f"波段数量不匹配: 期望{len(bands)}, 实际{data.shape[0]}")

        result = {}
        for i, (band_name, scale_factor) in enumerate(zip(bands, scale_factors)):
            if i >= data.shape[0]:
                logger.warning(f"波段索引{i}超出范围,跳过波段{band_name}")
                continue

            # 应用缩放因子
            scaled_data = data[i].astype(np.float32) / scale_factor

            # 处理NoData值
            scaled_data[data[i] == NODATA_VALUE] = np.nan

            # 映射波段名称
            mapped_name = BAND_NAME_MAPPING.get(band_name, band_name)
            result[mapped_name] = scaled_data

        return result

    def get_file_info(self, region: str, year_pair: str, season: str,
                     category: str) -> Optional[Dict]:
        """获取文件基本信息"""
        file_path = self._get_file_path(region, year_pair, season, category)

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None

        try:
            with rasterio.open(file_path) as src:
                return {
                    'shape': (src.height, src.width),
                    'transform': src.transform,
                    'crs': src.crs,
                    'dtype': src.dtypes[0],
                    'count': src.count,
                    'bounds': src.bounds,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'descriptions': src.descriptions
                }
        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")
            return None

    def load_essential_data_chunked(self, region: str, year_pair: str,
                                    season: str) -> Optional[Dict[str, np.ndarray]]:
        """
        分块加载用于网格筛选的必要数据
        """
        essential_categories = ['vegetation', 'classification']

        # 检查文件大小
        total_size_mb = 0
        file_infos = {}
        for category in essential_categories:
            file_info = self.get_file_info(region, year_pair, season, category)
            if file_info:
                total_size_mb += file_info['file_size_mb']
                file_infos[category] = file_info

        logger.info(f"必要数据总大小: {total_size_mb:.1f}MB")

        # 如果文件较小,使用常规加载
        if total_size_mb < MAX_MEMORY_MB / 2:
            return self.load_essential_data(region, year_pair, season)

        # 使用分块加载
        logger.info("文件较大,使用分块加载模式...")
        return self._load_essential_data_with_chunks(region, year_pair, season,
                                                     essential_categories, file_infos)

    def _load_essential_data_with_chunks(self, region: str, year_pair: str,
                                        season: str, categories: List[str],
                                        file_infos: Dict) -> Optional[Dict[str, np.ndarray]]:
        """使用分块方式加载必要数据"""
        # 获取第一个文件的基本信息
        first_category = categories[0]
        if first_category not in file_infos:
            logger.error("无法获取文件基本信息")
            return None

        file_info = file_infos[first_category]
        height, width = file_info['shape']
        result_data = {}

        # 初始化结果数组
        for category in categories:
            bands = CATEGORIES[category]["bands"]
            for band_name in bands:
                mapped_name = BAND_NAME_MAPPING.get(band_name, band_name)
                result_data[mapped_name] = np.full((height, width), np.nan, dtype=np.float32)

        # 添加基本信息
        result_data['transform'] = file_info['transform']
        result_data['crs'] = file_info['crs']
        result_data['shape'] = (height, width)

        # 分块处理每个类别
        for category in categories:
            file_path = self._get_file_path(region, year_pair, season, category)
            logger.info(f"分块加载 {category} 数据...")

            chunk_count = 0
            for chunk_dict, (row_start, row_end, col_start, col_end) in self.chunked_loader.get_file_chunks(file_path):
                try:
                    chunk_data = chunk_dict['data']
                    scaled_chunk = self._apply_scale_factors(chunk_data, category)

                    # 将块数据放入结果数组的对应位置
                    for band_name, band_data in scaled_chunk.items():
                        result_data[band_name][row_start:row_end, col_start:col_end] = band_data

                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        logger.info(f"已处理 {chunk_count}/{chunk_dict['total_chunks']} 个数据块")

                except Exception as e:
                    logger.error(f"处理数据块失败 [{row_start}:{row_end}, {col_start}:{col_end}]: {e}")
                    continue

            logger.info(f"{category} 数据分块加载完成,共处理 {chunk_count} 个块")

        return result_data

    def load_essential_data(self, region: str, year_pair: str,
                           season: str) -> Optional[Dict[str, np.ndarray]]:
        """
        加载用于网格筛选的必要数据(原始方法,用于小文件)
        只加载vegetation和classification数据
        """
        essential_categories = ['vegetation', 'classification']
        essential_data = {}

        logger.info("加载必要数据用于网格筛选...")

        for category in essential_categories:
            file_path = self._get_file_path(region, year_pair, season, category)

            if not file_path.exists():
                logger.error(f"必要文件不存在: {file_path}")
                return None

            try:
                with rasterio.open(file_path) as src:
                    data = src.read()
                    if category == 'vegetation':
                        essential_data['transform'] = src.transform
                        essential_data['crs'] = src.crs
                        essential_data['shape'] = data.shape[1:]

                scaled_data = self._apply_scale_factors(data, category)
                essential_data.update(scaled_data)

            except Exception as e:
                logger.error(f"加载必要数据失败: {file_path}, 错误: {e}")
                return None

        logger.info(f"必要数据加载完成,包含变量: {[k for k in essential_data.keys() if isinstance(essential_data[k], np.ndarray)]}")
        return essential_data

    def load_single_category_chunked(self, region: str, year_pair: str,
                                    season: str, category: str) -> Optional[Dict[str, np.ndarray]]:
        """分块加载单个数据类别"""
        file_path = self._get_file_path(region, year_pair, season, category)
        file_info = self.get_file_info(region, year_pair, season, category)

        if file_info is None:
            logger.warning(f"无法获取文件信息: {file_path}")
            return None

        # 检查文件大小
        if file_info['file_size_mb'] < MAX_MEMORY_MB / 4:
            return self.load_single_category(region, year_pair, season, category)

        logger.info(f"分块加载 {category} 数据 (文件大小: {file_info['file_size_mb']:.1f}MB)...")

        height, width = file_info['shape']
        result_data = {}

        # 初始化结果数组
        bands = CATEGORIES[category]["bands"]
        for band_name in bands:
            mapped_name = BAND_NAME_MAPPING.get(band_name, band_name)
            result_data[mapped_name] = np.full((height, width), np.nan, dtype=np.float32)

        # 添加基本信息
        result_data['transform'] = file_info['transform']
        result_data['crs'] = file_info['crs']
        result_data['shape'] = (height, width)

        # 分块处理
        chunk_count = 0
        for chunk_dict, (row_start, row_end, col_start, col_end) in self.chunked_loader.get_file_chunks(file_path):
            try:
                chunk_data = chunk_dict['data']
                scaled_chunk = self._apply_scale_factors(chunk_data, category)

                # 将块数据放入结果数组的对应位置
                for band_name, band_data in scaled_chunk.items():
                    result_data[band_name][row_start:row_end, col_start:col_end] = band_data

                chunk_count += 1

            except Exception as e:
                logger.error(f"处理数据块失败: {e}")
                continue

        logger.info(f"{category} 分块加载完成,共处理 {chunk_count} 个块")
        return result_data

    def load_single_category(self, region: str, year_pair: str, season: str,
                            category: str) -> Optional[Dict[str, np.ndarray]]:
        """加载单个数据类别(原始方法,用于小文件)"""
        file_path = self._get_file_path(region, year_pair, season, category)

        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return None

        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                transform = src.transform
                crs = src.crs

            scaled_data = self._apply_scale_factors(data, category)
            scaled_data['transform'] = transform
            scaled_data['crs'] = crs
            scaled_data['shape'] = data.shape[1:]

            logger.info(f"加载{category}数据完成,包含波段: {[k for k in scaled_data.keys() if isinstance(scaled_data[k], np.ndarray)]}")
            return scaled_data

        except Exception as e:
            logger.error(f"加载数据失败: {file_path}, 错误: {e}")
            return None

    def check_data_integrity(self, region: str, year_pair: str,
                            season: str) -> Dict[str, bool]:
        """检查数据完整性"""
        integrity = {}
        for category in CATEGORIES.keys():
            file_path = self._get_file_path(region, year_pair, season, category)
            integrity[category] = file_path.exists()
        return integrity

    def estimate_memory_usage(self, region: str, year_pair: str,
                             season: str) -> Dict[str, float]:
        """估算内存使用情况"""
        memory_info = {}
        total_size = 0

        for category in CATEGORIES.keys():
            file_info = self.get_file_info(region, year_pair, season, category)
            if file_info:
                file_size_mb = file_info['file_size_mb']
                memory_info[category] = file_size_mb
                total_size += file_size_mb

        memory_info['total'] = total_size
        memory_info['recommended_chunked'] = total_size > MAX_MEMORY_MB / 2

        return memory_info
