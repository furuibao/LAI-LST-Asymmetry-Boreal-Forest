"""优化的主分析模块 - 增强版本(添加初始值+优化写入速度)"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import gc

from constants import (
    REGIONS, SEASONS, CATEGORIES, YEAR_PAIRS,
    DEFAULT_SMALL_GRID_SIZE_KM, DEFAULT_LARGE_GRID_SIZE_KM,
    DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, BAND_NAME_MAPPING
)
from data_loader_local import OptimizedDataLoader
from grid_processor_local import OptimizedGridProcessor
from idw_processor_local import OptimizedIDWProcessor

logger = logging.getLogger(__name__)


class OptimizedVegetationClimateAnalyzer:
    """优化的植被-气候相互作用分析器"""

    def __init__(self, data_root_dir: str, output_root_dir: str,
                 chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """初始化分析器"""
        self.data_loader = OptimizedDataLoader(data_root_dir, chunk_size, overlap)
        self.output_root_dir = Path(output_root_dir)
        self.output_root_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def analyze_single_period_optimized(
            self,
            region: str,
            year_pair: str,
            season: str,
            large_grid_size_km: int = DEFAULT_LARGE_GRID_SIZE_KM,
            small_grid_size_km: int = DEFAULT_SMALL_GRID_SIZE_KM,
            threshold: float = 0.7,
            exclude_zero: bool = False
    ) -> Optional[pd.DataFrame]:
        """优化的单期分析方法"""
        try:
            logger.info(f"开始优化分析: {region}/{year_pair}/{season}")

            # 第一步:检查数据完整性和内存估算
            integrity = self.data_loader.check_data_integrity(region, year_pair, season)
            missing_categories = [cat for cat, exists in integrity.items() if not exists]

            if missing_categories:
                logger.error(f"缺少数据类别: {missing_categories}")
                return None

            # 估算内存使用
            memory_usage = self.data_loader.estimate_memory_usage(region, year_pair, season)
            logger.info(f"数据文件总大小: {memory_usage.get('total', 0):.1f}MB")

            if memory_usage.get('recommended_chunked', False):
                logger.info("推荐使用分块处理模式")

            # 第二步:智能加载必要数据进行网格筛选
            logger.info("步骤1: 智能加载必要数据并筛选网格...")
            essential_data = self.data_loader.load_essential_data_chunked(region, year_pair, season)

            if essential_data is None:
                logger.error("无法加载必要数据")
                return None

            # 验证必要变量存在
            required_vars = ['LAI_Begin', 'LAI_diff', 'LAI_Change_Type']
            missing_vars = [v for v in required_vars if v not in essential_data]
            if missing_vars:
                logger.error(f"必要数据缺少变量: {missing_vars}")
                logger.info(f"可用变量: {[k for k in essential_data.keys() if isinstance(essential_data[k], np.ndarray)]}")
                return None

            # 初始化网格处理器
            grid_processor = OptimizedGridProcessor(
                data_shape=essential_data['shape'],
                transform=essential_data['transform'],
                large_grid_size_km=large_grid_size_km,
                small_grid_size_km=small_grid_size_km,
                chunk_size=self.chunk_size
            )

            # 识别和筛选有效网格
            valid_grids = grid_processor.identify_valid_grids(essential_data)

            if len(valid_grids) == 0:
                logger.warning("没有找到有效网格")
                return None

            logger.info(f"找到{len(valid_grids)}个有效大网格")

            # 释放必要数据的内存(保留grid_types)
            grid_types_all = pd.concat([info.grid_types for info in valid_grids.values()], ignore_index=True)
            del essential_data
            gc.collect()

            # 第三步:智能增量加载和分析其他数据类别
            logger.info("步骤2: 开始智能增量分析...")

            incremental_results = []

            # *** 新增: 处理temperature数据(包含初始值) ***
            logger.info("智能加载temperature数据...")
            temp_data = self._smart_load_category(region, year_pair, season, 'temperature')

            if temp_data is not None:
                # 获取temperature变量(包含Begin值和diff值)
                temp_bands = CATEGORIES['temperature']['bands']
                temp_vars = [BAND_NAME_MAPPING.get(b, b) for b in temp_bands]
                available_temp_vars = [v for v in temp_vars if v in temp_data]

                logger.info(f"可用temperature变量: {available_temp_vars}")

                # 计算所有temperature变量的统计值(包含Begin值)
                temp_stats = grid_processor.compute_incremental_statistics(
                    temp_data, valid_grids, available_temp_vars
                )

                if len(temp_stats) > 0:
                    # 仅对diff变量进行IDW分析
                    temp_diff_vars = ['LST_Daily_diff', 'LST_Day_diff', 'LST_Night_diff']

                    idw_processor = OptimizedIDWProcessor()
                    temp_results = idw_processor.process_incremental_analysis(
                        grid_types_all, temp_stats, 'temperature', exclude_zero
                    )

                    if len(temp_results) > 0:
                        incremental_results.append(temp_results)
                        logger.info(f"temperature分析完成: {len(temp_results)}个网格")

                del temp_data, temp_stats
                gc.collect()

            # *** 新增: 处理energy数据(包含初始值) ***
            logger.info("智能加载energy数据...")
            energy_data = self._smart_load_category(region, year_pair, season, 'energy')

            if energy_data is not None:
                # 获取energy变量(包含Begin值和diff值)
                energy_bands = CATEGORIES['energy']['bands']
                energy_vars = [BAND_NAME_MAPPING.get(b, b) for b in energy_bands]
                available_energy_vars = [v for v in energy_vars if v in energy_data]

                logger.info(f"可用energy变量: {available_energy_vars}")

                # 计算所有energy变量的统计值(包含Begin值)
                energy_stats = grid_processor.compute_incremental_statistics(
                    energy_data, valid_grids, available_energy_vars
                )

                if len(energy_stats) > 0:
                    # 仅对diff变量进行IDW分析
                    energy_diff_vars = ['Albedo_diff', 'ET_diff', 'LE_diff', 'DSR_diff']

                    idw_processor = OptimizedIDWProcessor()
                    energy_results = idw_processor.process_incremental_analysis(
                        grid_types_all, energy_stats, 'energy', exclude_zero
                    )

                    if len(energy_results) > 0:
                        incremental_results.append(energy_results)
                        logger.info(f"energy分析完成: {len(energy_results)}个网格")

                del energy_data, energy_stats
                gc.collect()

            # 第四步:合并所有增量结果
            if not incremental_results:
                logger.warning("没有任何增量分析结果")
                return None

            logger.info("步骤3: 合并增量分析结果...")
            final_results = OptimizedIDWProcessor.merge_incremental_results(incremental_results)

            if len(final_results) == 0:
                logger.warning("结果合并失败")
                return None

            # 第五步:添加必要的基础信息
            logger.info("步骤4: 添加基础信息...")

            # 智能重新加载vegetation数据获取LAI相关信息
            vegetation_data = self._smart_load_category(region, year_pair, season, 'vegetation')

            if vegetation_data is not None:
                # 使用映射后的变量名(包含Begin值)
                lai_vars = ['LAI_Begin', 'LAI_End', 'LAI_diff', 'LAI_Change_Rate']
                available_lai_vars = [v for v in lai_vars if v in vegetation_data]

                if available_lai_vars:
                    lai_stats = grid_processor.compute_incremental_statistics(
                        vegetation_data, valid_grids, available_lai_vars
                    )

                    if len(lai_stats) > 0:
                        common_cols = ['large_grid_id', 'grid_id', 'center_x', 'center_y']
                        final_results = pd.merge(final_results, lai_stats, on=common_cols, how='left')

                del vegetation_data, lai_stats
                gc.collect()

            # 添加元数据
            final_results['region'] = region
            final_results['year_pair'] = year_pair
            final_results['season'] = season
            final_results['large_grid_size_km'] = large_grid_size_km
            final_results['small_grid_size_km'] = small_grid_size_km
            final_results['chunk_processed'] = memory_usage.get('recommended_chunked', False)

            logger.info(f"优化分析完成: {region}/{year_pair}/{season}, 得到 {len(final_results)} 个有效网格")
            return final_results

        except Exception as e:
            logger.error(f"优化分析失败 {region}/{year_pair}/{season}: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    def _smart_load_category(self, region: str, year_pair: str, season: str,
                            category: str) -> Optional[Dict[str, np.ndarray]]:
        """智能加载数据类别"""
        file_info = self.data_loader.get_file_info(region, year_pair, season, category)

        if file_info is None:
            logger.warning(f"无法获取{category}数据文件信息")
            return None

        file_size_mb = file_info['file_size_mb']
        logger.info(f"{category}数据文件大小: {file_size_mb:.1f}MB")

        # 根据文件大小自动选择加载方式
        if file_size_mb > 512:
            logger.info(f"使用分块模式加载{category}数据")
            return self.data_loader.load_single_category_chunked(region, year_pair, season, category)
        else:
            logger.info(f"使用常规模式加载{category}数据")
            return self.data_loader.load_single_category(region, year_pair, season, category)

    def save_optimized_results(self, results: pd.DataFrame, region: str,
                              year_pair: str, season: str):
        """
        保存优化分析结果 - 优化写入速度
        """
        output_dir = self.output_root_dir / region / year_pair
        output_dir.mkdir(parents=True, exist_ok=True)

        # 优化数据类型以加速写入
        results_optimized = self._optimize_dtypes(results)

        # 保存结果 - 使用高效的CSV写入参数
        csv_file_name = f"{season}.csv"
        output_file = output_dir / csv_file_name

        logger.info(f"开始保存结果到: {output_file}")

        # 使用优化参数快速写入
        results_optimized.to_csv(
            output_file,
            index=False,
            encoding='utf-8-sig',
            chunksize=10000,  # 分块写入
            compression=None,  # 不压缩以加快速度
            float_format='%.6f'  # 限制小数位数
        )

        logger.info(f"结果已保存 ({len(results_optimized)}行, {results_optimized.shape[1]}列)")

        # 保存统计摘要
        self._save_optimized_summary(results_optimized, output_dir, season, region, year_pair)

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame数据类型以加速写入和减小内存
        """
        df_optimized = df.copy()

        # 整数列优化
        int_cols = ['large_grid_id', 'grid_type']
        for col in int_cols:
            if col in df_optimized.columns:
                df_optimized[col] = df_optimized[col].astype('int32')

        # 浮点数列优化为float32
        float_cols = [col for col in df_optimized.columns
                     if df_optimized[col].dtype == 'float64']
        for col in float_cols:
            df_optimized[col] = df_optimized[col].astype('float32')

        # 字符串列优化
        str_cols = ['region', 'year_pair', 'season', 'grid_id']
        for col in str_cols:
            if col in df_optimized.columns:
                df_optimized[col] = df_optimized[col].astype('string')

        # 布尔列
        if 'chunk_processed' in df_optimized.columns:
            df_optimized['chunk_processed'] = df_optimized['chunk_processed'].astype('bool')

        logger.info(f"数据类型优化完成,内存使用: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f}MB")

        return df_optimized

    def _save_optimized_summary(self, results: pd.DataFrame, output_dir: Path,
                               season: str, region: str, year_pair: str):
        """保存优化分析的统计摘要"""
        summary_file = output_dir / f"{season}_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"优化分析结果摘要(支持分块处理)\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"区域: {region}\n")
            f.write(f"年份对: {year_pair}\n")
            f.write(f"季节: {season}\n")
            f.write(f"有效小网格数量: {len(results)}\n\n")

            # 检查是否使用了分块处理
            chunk_processed = results.get('chunk_processed', pd.Series([False])).iloc[0] if len(results) > 0 else False
            f.write(f"分块处理模式: {'启用' if chunk_processed else '未启用'}\n")
            f.write(f"分块大小: {self.chunk_size}像素\n\n")

            # 输出变量列表
            f.write(f"输出变量列表:\n")
            f.write(f"-" * 60 + "\n")

            # 基本信息
            basic_vars = ['large_grid_id', 'grid_id', 'center_x', 'center_y', 'grid_type']
            f.write(f"\n基本信息:\n")
            for var in basic_vars:
                if var in results.columns:
                    f.write(f"  - {var}\n")

            # LAI变量
            lai_vars = [col for col in results.columns if 'LAI' in col or 'lai' in col.lower()]
            if lai_vars:
                f.write(f"\nLAI变量 ({len(lai_vars)}个):\n")
                for var in sorted(lai_vars):
                    f.write(f"  - {var}\n")

            # 温度变量
            temp_vars = [col for col in results.columns if any(x in col for x in ['LST', 'lst', 'temp'])]
            if temp_vars:
                f.write(f"\n温度变量 ({len(temp_vars)}个):\n")
                # 分类显示
                begin_vars = [v for v in temp_vars if 'Begin' in v or 'begin' in v]
                end_vars = [v for v in temp_vars if 'End' in v or 'end' in v]
                diff_vars = [v for v in temp_vars if 'diff' in v]
                climate_vars = [v for v in temp_vars if 'climate' in v]
                lai_contrib_vars = [v for v in temp_vars if '_lai' in v and 'climate' not in v]

                if begin_vars:
                    f.write(f"  初始值:\n")
                    for v in sorted(begin_vars):
                        f.write(f"    - {v}\n")
                if climate_vars:
                    f.write(f"  气候贡献:\n")
                    for v in sorted(climate_vars):
                        f.write(f"    - {v}\n")
                if lai_contrib_vars:
                    f.write(f"  植被贡献:\n")
                    for v in sorted(lai_contrib_vars):
                        f.write(f"    - {v}\n")

            # 能量变量
            energy_vars = [col for col in results.columns if any(x in col.lower() for x in ['albedo', 'et_', 'le_', 'dsr'])]
            if energy_vars:
                f.write(f"\n能量变量 ({len(energy_vars)}个):\n")
                # 分类显示
                begin_vars = [v for v in energy_vars if 'Begin' in v or 'begin' in v]
                climate_vars = [v for v in energy_vars if 'climate' in v]
                lai_contrib_vars = [v for v in energy_vars if '_lai' in v and 'climate' not in v]

                if begin_vars:
                    f.write(f"  初始值:\n")
                    for v in sorted(begin_vars):
                        f.write(f"    - {v}\n")
                if climate_vars:
                    f.write(f"  气候贡献:\n")
                    for v in sorted(climate_vars):
                        f.write(f"    - {v}\n")
                if lai_contrib_vars:
                    f.write(f"  植被贡献:\n")
                    for v in sorted(lai_contrib_vars):
                        f.write(f"    - {v}\n")

            f.write(f"\n" + "=" * 60 + "\n")

            # 内存优化信息
            f.write(f"\n内存优化策略:\n")
            f.write(f"  ✓ 增量数据加载\n")
            f.write(f"  ✓ 智能分块处理\n")
            f.write(f"  ✓ 即时内存释放\n")
            f.write(f"  ✓ 分组IDW分析\n")
            f.write(f"  ✓ 结果增量合并\n")
            f.write(f"  ✓ 自适应文件大小检测\n")
            f.write(f"  ✓ 优化数据类型\n")
            f.write(f"  ✓ 分块CSV写入\n\n")

            # 基本统计
            if len(results) > 0:
                f.write(f"网格统计:\n")
                if 'grid_type' in results.columns:
                    grid_type_counts = results['grid_type'].value_counts()
                    f.write(f"  网格类型分布:\n")
                    for gt, count in grid_type_counts.items():
                        type_name = {1: 'LAI增加', -1: 'LAI减少', 0: 'LAI稳定'}.get(gt, f'未知({gt})')
                        f.write(f"    {type_name}: {count} 个 ({count/len(results)*100:.1f}%)\n")

        logger.info(f"统计摘要已保存到: {summary_file}")

    def run_optimized_analysis(
            self,
            regions: List[str] = None,
            year_pairs: List[str] = None,
            seasons: List[str] = None,
            large_grid_size_km: int = DEFAULT_LARGE_GRID_SIZE_KM,
            small_grid_size_km: int = DEFAULT_SMALL_GRID_SIZE_KM,
            threshold: float = 0.7,
            exclude_zero: bool = False
    ):
        """运行优化的批量分析"""
        # 设置默认值
        if regions is None:
            regions = REGIONS
        if seasons is None:
            seasons = SEASONS
        if year_pairs is None:
            year_pairs = YEAR_PAIRS

        logger.info(f"开始优化批量分析(支持分块处理):")
        logger.info(f"  区域: {regions}")
        logger.info(f"  年份对数量: {len(year_pairs)}")
        logger.info(f"  季节: {seasons}")
        logger.info(f"  分块大小: {self.chunk_size}像素")
        logger.info(f"  内存优化: 启用")
        logger.info(f"  分块处理: 自适应启用")
        logger.info(f"  快速写入: 启用")

        total_tasks = len(regions) * len(year_pairs) * len(seasons)
        completed_tasks = 0
        failed_tasks = []
        chunked_tasks = 0

        # 执行分析
        for region in regions:
            for year_pair in year_pairs:
                for season in seasons:
                    try:
                        logger.info(f"处理: {region}/{year_pair}/{season} ({completed_tasks + len(failed_tasks) + 1}/{total_tasks})")

                        # 预检查文件大小
                        memory_usage = self.data_loader.estimate_memory_usage(region, year_pair, season)
                        if memory_usage.get('recommended_chunked', False):
                            chunked_tasks += 1
                            logger.info(f"文件较大({memory_usage.get('total', 0):.1f}MB),将使用分块处理")

                        results = self.analyze_single_period_optimized(
                            region, year_pair, season,
                            large_grid_size_km, small_grid_size_km,
                            threshold, exclude_zero
                        )

                        if results is not None:
                            self.save_optimized_results(results, region, year_pair, season)
                            completed_tasks += 1
                            logger.info(f"✓ 成功完成")
                        else:
                            failed_tasks.append(f"{region}/{year_pair}/{season}")
                            logger.warning(f"✗ 分析失败")

                        # 显式内存清理
                        gc.collect()

                    except Exception as e:
                        logger.error(f"✗ 处理失败 {region}/{year_pair}/{season}: {e}")
                        failed_tasks.append(f"{region}/{year_pair}/{season}")

        # 输出最终统计
        logger.info(f"\n优化批量分析完成:")
        logger.info(f"  成功: {completed_tasks}/{total_tasks}")
        logger.info(f"  失败: {len(failed_tasks)}/{total_tasks}")
        logger.info(f"  成功率: {completed_tasks/total_tasks*100:.1f}%")
        logger.info(f"  使用分块处理: {chunked_tasks}/{total_tasks}")

        if failed_tasks:
            logger.info(f"失败的任务: {failed_tasks[:5]}{'...' if len(failed_tasks) > 5 else ''}")

    def estimate_memory_savings(self, region: str, year_pair: str,
                               season: str) -> Dict[str, str]:
        """估算内存节省情况"""
        try:
            memory_usage = self.data_loader.estimate_memory_usage(region, year_pair, season)

            if not memory_usage:
                return {}

            total_size_mb = memory_usage.get('total', 0)

            # 估算传统模式的内存使用
            traditional_memory = total_size_mb * 2

            # 估算优化模式的内存使用
            max_single_file = max([size for cat, size in memory_usage.items()
                                 if cat not in ['total', 'recommended_chunked']] + [0])

            if memory_usage.get('recommended_chunked', False):
                chunk_memory = (self.chunk_size * self.chunk_size * 4 * 8) / (1024 * 1024)
                optimized_memory = max(chunk_memory, max_single_file * 0.2)
            else:
                optimized_memory = max_single_file * 2

            savings = traditional_memory - optimized_memory
            savings_percent = (savings / traditional_memory) * 100 if traditional_memory > 0 else 0

            return {
                'traditional_memory': f"{traditional_memory:.1f} MB",
                'optimized_memory': f"{optimized_memory:.1f} MB",
                'memory_savings': f"{savings:.1f} MB",
                'savings_percent': f"{savings_percent:.1f}%",
                'chunked_processing': '启用' if memory_usage.get('recommended_chunked', False) else '未启用',
                'chunk_size': f"{self.chunk_size}像素",
                'file_count': str(len([k for k in memory_usage.keys() if k not in ['total', 'recommended_chunked']]))
            }

        except Exception as e:
            logger.error(f"内存估算失败: {e}")
            return {}

    def get_processing_statistics(self, region: str, year_pair: str,
                                 season: str) -> Dict[str, any]:
        """获取处理统计信息"""
        try:
            stats = {}

            # 获取文件信息
            file_info = {}
            for category in CATEGORIES.keys():
                info = self.data_loader.get_file_info(region, year_pair, season, category)
                if info:
                    file_info[category] = info

            stats['file_info'] = file_info

            # 计算总数据大小
            total_size = sum(info['file_size_mb'] for info in file_info.values())
            stats['total_size_mb'] = total_size

            # 估算分块数量
            if file_info:
                max_shape = max([info['shape'] for info in file_info.values()], key=lambda x: x[0] * x[1])
                height, width = max_shape
                chunks_y = int(np.ceil(height / self.chunk_size))
                chunks_x = int(np.ceil(width / self.chunk_size))
                stats['estimated_chunks'] = chunks_y * chunks_x
            else:
                stats['estimated_chunks'] = 0

            # 推荐处理模式
            stats['recommended_chunked'] = total_size > 1024

            return stats

        except Exception as e:
            logger.error(f"获取处理统计失败: {e}")
            return {}
