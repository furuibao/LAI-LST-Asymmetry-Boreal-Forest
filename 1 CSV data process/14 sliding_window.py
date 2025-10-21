#!/usr/bin/env python3
"""
配对网格环境变量空间关系分析脚本
用于分析LAI变化与环境变量的空间关系
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import sys
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """分析配置类"""
    INPUT_FILE: str = r"D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\01_matching_results\Boreal\Spring_mode1.csv"
    OUTPUT_DIR: str = r"D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\02_sliding_window\Boreal"
    WINDOW_SIZE: float = 6.0  # 度
    STEP_SIZE: float = 2.0  # 度
    MIN_POINTS_GRID1: int = 3  # Grid1最小数据点
    MIN_POINTS_GRID2: int = 3  # Grid2最小数据点
    ANALYSIS_TYPE: str = 'both'  # 'grid1', 'grid2', 'both'
    ANALYSIS_VARIABLES: List[str] = None
    USE_ROBUST_REGRESSION: bool = True
    LATITUDE_BAND_WIDTH: float = 1.0  # 纬度带宽度
    GRID_AGGREGATION_SIZE: float = 2.0  # 网格聚合大小
    LAI_RATE_INTERVAL: float = 0.2  # LAI变化率分类间隔
    USE_LAND_MASK: bool = True
    LAND_MASK_FILE: Optional[str] = r"D:\article\SynologyDrive\shared_data\shpData\world\world.shp"

    def __post_init__(self):
        if self.ANALYSIS_VARIABLES is None:
            self.ANALYSIS_VARIABLES = [
                'lst_daily', 'lst_day', 'lst_night',
                'le', 'et', 'sw', 'lw', 'hg', 'albedo', 'dsr'
            ]


class PairedGridAnalyzer:
    """配对网格分析器主类"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.df = None
        self.results = None
        self.debug_stats = {
            'grid1': {'windows_checked': 0, 'insufficient_data': 0,
                      'missing_columns': 0, 'successful': 0},
            'grid2': {'windows_checked': 0, 'insufficient_data': 0,
                      'missing_columns': 0, 'successful': 0}
        }
        self.land_mask = None
        if self.config.USE_LAND_MASK and self.config.LAND_MASK_FILE:
            self.land_mask = self.load_land_mask()
        self.setup_output_dir()

    def setup_output_dir(self):
        """创建输出目录"""
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """加载并验证数据"""
        try:
            logger.info(f"加载数据文件: {self.config.INPUT_FILE}")
            self.df = pd.read_csv(self.config.INPUT_FILE)

            # 验证数据结构
            self._validate_data_structure()

            # 数据预处理
            self._preprocess_data()

            logger.info(f"成功加载 {len(self.df)} 行配对网格数据")
            return self.df

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def _validate_data_structure(self):
        """验证数据结构完整性"""
        required_cols = {
            'spatial': ['center_x_1', 'center_y_1', 'center_x_2', 'center_y_2'],
            'lai': ['lai_change_rate_1', 'lai_change_rate_2'],
            'id': ['grid_id_1', 'grid_id_2']
        }

        missing_cols = []
        for category, cols in required_cols.items():
            for col in cols:
                if col not in self.df.columns:
                    missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # *** 新增: 自动判别LAI变化方向 ***
        self.lai_direction = {}

        # Grid 1统计
        grid1_total = len(self.df)
        grid1_valid = self.df['lai_change_rate_1'].notna().sum()
        grid1_positive = (self.df['lai_change_rate_1'] > 0).sum()
        grid1_negative = (self.df['lai_change_rate_1'] < 0).sum()
        grid1_zero = (self.df['lai_change_rate_1'] == 0).sum()
        grid1_mean = self.df['lai_change_rate_1'].mean()

        # Grid 2统计
        grid2_total = len(self.df)
        grid2_valid = self.df['lai_change_rate_2'].notna().sum()
        grid2_positive = (self.df['lai_change_rate_2'] > 0).sum()
        grid2_negative = (self.df['lai_change_rate_2'] < 0).sum()
        grid2_zero = (self.df['lai_change_rate_2'] == 0).sum()
        grid2_mean = self.df['lai_change_rate_2'].mean()

        # *** 判别LAI主要变化方向 ***
        if grid1_positive > grid1_negative:
            self.lai_direction['grid1'] = '增加'
        elif grid1_positive < grid1_negative:
            self.lai_direction['grid1'] = '减少'
        else:
            self.lai_direction['grid1'] = '混合'

        if grid2_positive > grid2_negative:
            self.lai_direction['grid2'] = '增加'
        elif grid2_positive < grid2_negative:
            self.lai_direction['grid2'] = '减少'
        else:
            self.lai_direction['grid2'] = '混合'

        logger.info(f"\n=== LAI变化方向判别 ===")
        logger.info(f"Grid1 主要方向: {self.lai_direction['grid1']} (均值: {grid1_mean:.4f})")
        logger.info(f"Grid2 主要方向: {self.lai_direction['grid2']} (均值: {grid2_mean:.4f})")

        logger.info(f"\nGrid1 LAI变化统计 (总数: {grid1_total}, 有效: {grid1_valid}):")
        logger.info(f"  增加: {grid1_positive} ({100 * grid1_positive / grid1_valid:.1f}%)")
        logger.info(f"  减少: {grid1_negative} ({100 * grid1_negative / grid1_valid:.1f}%)")
        logger.info(f"  不变: {grid1_zero} ({100 * grid1_zero / grid1_valid:.1f}%)")

        logger.info(f"\nGrid2 LAI变化统计 (总数: {grid2_total}, 有效: {grid2_valid}):")
        logger.info(f"  增加: {grid2_positive} ({100 * grid2_positive / grid2_valid:.1f}%)")
        logger.info(f"  减少: {grid2_negative} ({100 * grid2_negative / grid2_valid:.1f}%)")
        logger.info(f"  不变: {grid2_zero} ({100 * grid2_zero / grid2_valid:.1f}%)")

        # *** 新增: 警告不一致的情况 ***
        if self.lai_direction['grid1'] == self.lai_direction['grid2']:
            logger.warning(f"\n⚠️  警告: Grid1和Grid2的LAI变化方向相同 ({self.lai_direction['grid1']})")
            logger.warning("这可能表明数据配对存在问题,或该区域LAI变化模式复杂")

    def _preprocess_data(self):
        """数据预处理"""
        # 处理缺失值
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # 添加质量标记
        self.df['quality_flag_1'] = ~self.df['lai_change_rate_1'].isna()
        self.df['quality_flag_2'] = ~self.df['lai_change_rate_2'].isna()

    def load_land_mask(self) -> Optional['gpd.GeoDataFrame']:
        """加载陆地掩膜数据"""
        try:
            import geopandas as gpd
            logger.info(f"正在加载陆地掩膜: {self.config.LAND_MASK_FILE}")
            land_mask = gpd.read_file(self.config.LAND_MASK_FILE)
            logger.info(f"陆地掩膜加载成功")
            return land_mask
        except Exception as e:
            logger.warning(f"无法加载陆地掩膜: {e}")
            return None

    def is_point_on_land(self, x: float, y: float) -> bool:
        """检查点是否在陆地上"""
        if self.land_mask is None:
            return True

        try:
            from shapely.geometry import Point
            point = Point(x, y)
            return any(self.land_mask.contains(point))
        except:
            return True

    def create_sliding_windows(self) -> List[Dict]:
        """创建滑动窗口"""
        windows = []

        # 获取空间范围
        lon_min = min(self.df['center_x_1'].min(), self.df['center_x_2'].min())
        lon_max = max(self.df['center_x_1'].max(), self.df['center_x_2'].max())
        lat_min = min(self.df['center_y_1'].min(), self.df['center_y_2'].min())
        lat_max = max(self.df['center_y_1'].max(), self.df['center_y_2'].max())

        # 生成窗口
        lon_starts = np.arange(lon_min, lon_max - self.config.WINDOW_SIZE + self.config.STEP_SIZE,
                               self.config.STEP_SIZE)
        lat_starts = np.arange(lat_min, lat_max - self.config.WINDOW_SIZE + self.config.STEP_SIZE,
                               self.config.STEP_SIZE)

        for lon_start in lon_starts:
            for lat_start in lat_starts:
                window = {
                    'lon_start': lon_start,
                    'lon_end': lon_start + self.config.WINDOW_SIZE,
                    'lat_start': lat_start,
                    'lat_end': lat_start + self.config.WINDOW_SIZE,
                    'center_lon': lon_start + self.config.WINDOW_SIZE / 2,
                    'center_lat': lat_start + self.config.WINDOW_SIZE / 2
                }

                if self.is_point_on_land(window['center_lon'], window['center_lat']):
                    windows.append(window)

        logger.info(f"创建了 {len(windows)} 个滑动窗口")
        return windows

    def select_data_in_window(self, window: Dict, grid_type: int) -> pd.DataFrame:
        """选择窗口内数据"""
        x_col = f'center_x_{grid_type}'
        y_col = f'center_y_{grid_type}'

        mask = (
                (self.df[x_col] >= window['lon_start']) &
                (self.df[x_col] < window['lon_end']) &
                (self.df[y_col] >= window['lat_start']) &
                (self.df[y_col] < window['lat_end'])
        )

        return self.df[mask]

    def perform_regression(self, x: np.ndarray, y: np.ndarray,
                           method: str = 'linear') -> Dict:
        """执行回归分析"""
        result = {
            'n': len(x),
            'method': method,
            'slope': np.nan,
            'intercept': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_error': np.nan
        }

        if len(x) < 2:
            return result

        try:
            if method == 'linear':
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                result.update({
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'std_error': std_err
                })

            elif method == 'robust':
                X = sm.add_constant(x)
                model = RLM(y, X, M=sm.robust.norms.HuberT())
                results = model.fit()
                result.update({
                    'slope': results.params[1] if len(results.params) > 1 else np.nan,
                    'intercept': results.params[0],
                    'p_value': results.pvalues[1] if len(results.pvalues) > 1 else np.nan,
                    'std_error': results.bse[1] if len(results.bse) > 1 else np.nan
                })

                # 计算伪R²
                y_pred = results.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                result['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        except Exception as e:
            logger.debug(f"回归分析失败: {e}")

        return result

    def analyze_window(self, window: Dict, grid_type: int) -> Dict:
        """分析单个窗口"""
        # 获取窗口内数据
        window_data = self.select_data_in_window(window, grid_type)

        # 检查数据点数量
        min_points = (self.config.MIN_POINTS_GRID1 if grid_type == 1
                      else self.config.MIN_POINTS_GRID2)

        self.debug_stats[f'grid{grid_type}']['windows_checked'] += 1

        if len(window_data) < min_points:
            self.debug_stats[f'grid{grid_type}']['insufficient_data'] += 1
            return None

        # LAI变化率列
        lai_col = f'lai_change_rate_{grid_type}'

        # 检查数据变异性
        if window_data[lai_col].std() == 0:
            return None

        results = {
            'window': window,
            'grid_type': grid_type,
            'n_points': len(window_data),
            'lai_mean': window_data[lai_col].mean(),
            'lai_std': window_data[lai_col].std()
        }

        # 对每个分析变量进行回归
        for var in self.config.ANALYSIS_VARIABLES:
            var_col = f'{var}_diff_lai_{grid_type}'

            if var_col not in window_data.columns:
                self.debug_stats[f'grid{grid_type}']['missing_columns'] += 1
                continue

            # 移除缺失值
            valid_mask = ~(window_data[lai_col].isna() | window_data[var_col].isna())
            if valid_mask.sum() < min_points:
                continue

            x = window_data.loc[valid_mask, lai_col].values
            y = window_data.loc[valid_mask, var_col].values

            # *** 新增: 计算环境变量的统计信息 ***
            results[f'{var}_mean'] = y.mean()
            results[f'{var}_std'] = y.std()
            results[f'{var}_min'] = y.min()
            results[f'{var}_max'] = y.max()
            results[f'{var}_n_valid'] = len(y)

            # 线性回归
            linear_result = self.perform_regression(x, y, 'linear')
            results[f'{var}_linear'] = linear_result

            # 稳健回归
            if self.config.USE_ROBUST_REGRESSION:
                robust_result = self.perform_regression(x, y, 'robust')
                results[f'{var}_robust'] = robust_result

        self.debug_stats[f'grid{grid_type}']['successful'] += 1
        return results

    def analyze_all_windows(self, windows: List[Dict]) -> Dict:
        """分析所有窗口"""
        all_results = {'grid1': [], 'grid2': []}

        total = len(windows)
        for i, window in enumerate(windows):
            if (i + 1) % 100 == 0:
                logger.info(f"处理进度: {i + 1}/{total} 窗口")

            # 根据分析类型处理不同网格
            if self.config.ANALYSIS_TYPE in ['grid1', 'both']:
                result = self.analyze_window(window, 1)
                if result:
                    all_results['grid1'].append(result)

            if self.config.ANALYSIS_TYPE in ['grid2', 'both']:
                result = self.analyze_window(window, 2)
                if result:
                    all_results['grid2'].append(result)

        return all_results

    def calculate_latitude_bands(self, results: Dict) -> pd.DataFrame:
        """计算纬度带统计"""
        lat_bands = []

        for grid_type in ['grid1', 'grid2']:
            if grid_type not in results or not results[grid_type]:
                continue

            df_results = pd.DataFrame(results[grid_type])

            # 提取中心纬度
            df_results['center_lat'] = df_results['window'].apply(lambda x: x['center_lat'])

            # 创建纬度带
            lat_min = df_results['center_lat'].min()
            lat_max = df_results['center_lat'].max()
            bands = np.arange(lat_min, lat_max + self.config.LATITUDE_BAND_WIDTH,
                              self.config.LATITUDE_BAND_WIDTH)

            for i in range(len(bands) - 1):
                band_mask = (df_results['center_lat'] >= bands[i]) & \
                            (df_results['center_lat'] < bands[i + 1])
                band_data = df_results[band_mask]

                if len(band_data) == 0:
                    continue

                band_stats = {
                    'grid_type': grid_type,
                    'lat_start': bands[i],
                    'lat_end': bands[i + 1],
                    'lat_center': (bands[i] + bands[i + 1]) / 2,
                    'n_windows': len(band_data),
                    'lai_mean': band_data['lai_mean'].mean(),
                    'lai_std': band_data['lai_std'].mean()
                }

                # 统计每个变量的平均回归结果
                for var in self.config.ANALYSIS_VARIABLES:
                    linear_col = f'{var}_linear'
                    if linear_col in band_data.columns:
                        # 提取有效的回归结果
                        valid_results = []
                        for _, row in band_data.iterrows():
                            if linear_col in row and isinstance(row[linear_col], dict):
                                if not np.isnan(row[linear_col].get('r_squared', np.nan)):
                                    valid_results.append(row[linear_col])

                        if valid_results:
                            band_stats[f'{var}_mean_r2'] = np.mean([r['r_squared'] for r in valid_results])
                            band_stats[f'{var}_mean_slope'] = np.mean([r['slope'] for r in valid_results])
                            band_stats[f'{var}_significant'] = sum([r['p_value'] < 0.05 for r in valid_results])

                lat_bands.append(band_stats)

        return pd.DataFrame(lat_bands)

    def aggregate_by_grid(self, results: Dict) -> pd.DataFrame:
        """按网格聚合结果"""
        grid_results = []

        for grid_type in ['grid1', 'grid2']:
            if grid_type not in results or not results[grid_type]:
                continue

            df_results = pd.DataFrame(results[grid_type])

            # 提取中心坐标
            df_results['center_lon'] = df_results['window'].apply(lambda x: x['center_lon'])
            df_results['center_lat'] = df_results['window'].apply(lambda x: x['center_lat'])

            # 创建网格索引
            grid_size = self.config.GRID_AGGREGATION_SIZE
            df_results['grid_lon'] = (df_results['center_lon'] // grid_size) * grid_size
            df_results['grid_lat'] = (df_results['center_lat'] // grid_size) * grid_size

            # 按网格聚合
            for (grid_lon, grid_lat), group in df_results.groupby(['grid_lon', 'grid_lat']):
                grid_stat = {
                    'grid_type': grid_type,
                    'grid_lon': grid_lon,
                    'grid_lat': grid_lat,
                    'n_windows': len(group),
                    'lai_mean': group['lai_mean'].mean(),
                    'lai_std': group['lai_std'].mean()
                }

                # 聚合回归结果
                for var in self.config.ANALYSIS_VARIABLES:
                    linear_col = f'{var}_linear'
                    if linear_col in group.columns:
                        valid_results = []
                        for _, row in group.iterrows():
                            if linear_col in row and isinstance(row[linear_col], dict):
                                if not np.isnan(row[linear_col].get('r_squared', np.nan)):
                                    valid_results.append(row[linear_col])

                        if valid_results:
                            grid_stat[f'{var}_mean_r2'] = np.mean([r['r_squared'] for r in valid_results])
                            grid_stat[f'{var}_mean_slope'] = np.mean([r['slope'] for r in valid_results])

                grid_results.append(grid_stat)

        return pd.DataFrame(grid_results)

    def classify_by_lai_rate(self, results: Dict) -> pd.DataFrame:
        """按LAI变化率分类统计"""
        lai_classes = []

        for grid_type in ['grid1', 'grid2']:
            if grid_type not in results or not results[grid_type]:
                continue

            df_results = pd.DataFrame(results[grid_type])

            # 创建LAI变化率区间
            lai_min = df_results['lai_mean'].min()
            lai_max = df_results['lai_mean'].max()
            intervals = np.arange(lai_min, lai_max + self.config.LAI_RATE_INTERVAL,
                                  self.config.LAI_RATE_INTERVAL)

            for i in range(len(intervals) - 1):
                interval_mask = (df_results['lai_mean'] >= intervals[i]) & \
                                (df_results['lai_mean'] < intervals[i + 1])
                interval_data = df_results[interval_mask]

                if len(interval_data) == 0:
                    continue

                class_stat = {
                    'grid_type': grid_type,
                    'lai_rate_min': intervals[i],
                    'lai_rate_max': intervals[i + 1],
                    'lai_rate_center': (intervals[i] + intervals[i + 1]) / 2,
                    'n_windows': len(interval_data)
                }

                # 统计每个变量的响应
                for var in self.config.ANALYSIS_VARIABLES:
                    linear_col = f'{var}_linear'
                    if linear_col in interval_data.columns:
                        valid_results = []
                        for _, row in interval_data.iterrows():
                            if linear_col in row and isinstance(row[linear_col], dict):
                                if not np.isnan(row[linear_col].get('slope', np.nan)):
                                    valid_results.append(row[linear_col])

                        if valid_results:
                            class_stat[f'{var}_mean_slope'] = np.mean([r['slope'] for r in valid_results])
                            class_stat[f'{var}_std_slope'] = np.std([r['slope'] for r in valid_results])

                lai_classes.append(class_stat)

        return pd.DataFrame(lai_classes)

    def format_results_for_excel(self, results: Dict) -> Dict[str, pd.DataFrame]:
        """格式化结果为Excel输出"""
        excel_sheets = {}

        # 1. 完整结果表
        all_data = []
        for grid_type in ['grid1', 'grid2']:
            if grid_type not in results or not results[grid_type]:
                continue

            # *** 获取该grid的LAI方向 ***
            lai_direction = self.lai_direction.get(grid_type, '未知')

            for window_result in results[grid_type]:
                row = {
                    'grid_type': grid_type,
                    'lai_direction': lai_direction,  # *** 新增列 ***
                    'lon_start': window_result['window']['lon_start'],
                    'lon_end': window_result['window']['lon_end'],
                    'lat_start': window_result['window']['lat_start'],
                    'lat_end': window_result['window']['lat_end'],
                    'center_lon': window_result['window']['center_lon'],
                    'center_lat': window_result['window']['center_lat'],
                    'n_points': window_result['n_points'],
                    'lai_mean': window_result['lai_mean'],
                    'lai_std': window_result['lai_std']
                }

                # 添加各变量的统计信息和回归结果
                for var in self.config.ANALYSIS_VARIABLES:
                    # 添加变量统计信息
                    for stat in ['mean', 'std', 'min', 'max', 'n_valid']:
                        stat_key = f'{var}_{stat}'
                        if stat_key in window_result:
                            row[stat_key] = window_result[stat_key]

                    # 添加线性回归结果
                    linear_key = f'{var}_linear'
                    if linear_key in window_result:
                        for metric, value in window_result[linear_key].items():
                            row[f'{var}_{metric}'] = value

                    # 添加稳健回归结果
                    if self.config.USE_ROBUST_REGRESSION:
                        robust_key = f'{var}_robust'
                        if robust_key in window_result:
                            for metric, value in window_result[robust_key].items():
                                row[f'{var}_robust_{metric}'] = value

                all_data.append(row)

        excel_sheets['all_variables'] = pd.DataFrame(all_data)

        # 2. 每个变量的单独结果表 (同样添加lai_direction列)
        for var in self.config.ANALYSIS_VARIABLES:
            var_data = []
            for grid_type in ['grid1', 'grid2']:
                if grid_type not in results or not results[grid_type]:
                    continue

                lai_direction = self.lai_direction.get(grid_type, '未知')

                for window_result in results[grid_type]:
                    if f'{var}_mean' not in window_result:
                        continue

                    row = {
                        'grid_type': grid_type,
                        'lai_direction': lai_direction,  # *** 新增列 ***
                        'center_lon': window_result['window']['center_lon'],
                        'center_lat': window_result['window']['center_lat'],
                        'n_points': window_result['n_points'],
                        'lai_mean': window_result['lai_mean'],
                        'lai_std': window_result['lai_std'],
                        f'{var}_mean': window_result.get(f'{var}_mean', np.nan),
                        f'{var}_std': window_result.get(f'{var}_std', np.nan),
                        f'{var}_min': window_result.get(f'{var}_min', np.nan),
                        f'{var}_max': window_result.get(f'{var}_max', np.nan),
                        f'{var}_n_valid': window_result.get(f'{var}_n_valid', 0)
                    }

                    # 添加线性回归结果
                    linear_key = f'{var}_linear'
                    if linear_key in window_result:
                        for metric, value in window_result[linear_key].items():
                            row[f'linear_{metric}'] = value

                    # 添加稳健回归结果
                    if self.config.USE_ROBUST_REGRESSION:
                        robust_key = f'{var}_robust'
                        if robust_key in window_result:
                            for metric, value in window_result[robust_key].items():
                                row[f'robust_{metric}'] = value

                    var_data.append(row)

            if var_data:
                excel_sheets[f'{var}_results'] = pd.DataFrame(var_data)

        # 3-7. 其他表格保持不变
        excel_sheets['latitude_bands'] = self.calculate_latitude_bands(results)
        excel_sheets['grid_aggregation'] = self.aggregate_by_grid(results)
        excel_sheets['lai_rate_classes'] = self.classify_by_lai_rate(results)

        config_df = pd.DataFrame([asdict(self.config)])
        excel_sheets['config'] = config_df

        debug_df = pd.DataFrame(self.debug_stats).T
        debug_df.index.name = 'grid_type'
        excel_sheets['debug_stats'] = debug_df.reset_index()

        # *** 新增: LAI方向摘要表 ***
        lai_direction_df = pd.DataFrame([
            {'grid_type': 'grid1', 'lai_direction': self.lai_direction.get('grid1', '未知')},
            {'grid_type': 'grid2', 'lai_direction': self.lai_direction.get('grid2', '未知')}
        ])
        excel_sheets['lai_direction_summary'] = lai_direction_df

        return excel_sheets

    def save_results(self, results: Dict, output_file: str = None):
        """保存结果到Excel文件"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_stem = Path(self.config.INPUT_FILE).stem
            output_file = self.output_dir / f'{input_stem}_analysis_{timestamp}.xlsx'

        excel_sheets = self.format_results_for_excel(results)

        # 写入Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in excel_sheets.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"已保存 {sheet_name} 表 ({len(df)} 行)")

        logger.info(f"结果已保存至: {output_file}")

        # 打印调试统计
        logger.info("\n=== 调试统计 ===")
        for grid_type, stats in self.debug_stats.items():
            logger.info(f"\n{grid_type.upper()}:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

    def run(self):
        """运行完整分析流程"""
        try:
            # 1. 加载数据
            self.load_data()

            # 2. 创建滑动窗口
            windows = self.create_sliding_windows()

            # 3. 分析所有窗口
            logger.info("开始滑动窗口分析...")
            results = self.analyze_all_windows(windows)

            # 4. 保存结果
            self.save_results(results)

            # 5. 生成摘要报告
            self.generate_summary_report(results)

            logger.info("分析完成!")

        except Exception as e:
            logger.error(f"分析过程出错: {e}")
            raise

    def generate_summary_report(self, results: Dict):
        """生成摘要报告"""
        logger.info("\n=== 分析摘要 ===")

        for grid_type in ['grid1', 'grid2']:
            if grid_type not in results or not results[grid_type]:
                continue

            # *** 使用自动检测的LAI方向 ***
            lai_direction = self.lai_direction.get(grid_type, '未知')
            logger.info(f"\n{grid_type.upper()} (LAI {lai_direction}):")
            logger.info(f"  有效窗口数: {len(results[grid_type])}")

            # 统计显著相关的比例
            for var in self.config.ANALYSIS_VARIABLES:
                significant_count = 0
                total_valid = 0
                positive_slope = 0  # *** 新增: 统计正斜率数量 ***
                negative_slope = 0  # *** 新增: 统计负斜率数量 ***

                for window_result in results[grid_type]:
                    linear_key = f'{var}_linear'
                    if linear_key in window_result:
                        p_val = window_result[linear_key]['p_value']
                        slope = window_result[linear_key]['slope']

                        if not np.isnan(p_val):
                            total_valid += 1
                            if p_val < 0.05:
                                significant_count += 1

                            # *** 新增: 统计斜率方向 ***
                            if not np.isnan(slope):
                                if slope > 0:
                                    positive_slope += 1
                                elif slope < 0:
                                    negative_slope += 1

                if total_valid > 0:
                    logger.info(f"  {var}:")
                    logger.info(f"    显著相关: {significant_count}/{total_valid} "
                                f"({100 * significant_count / total_valid:.1f}%)")
                    logger.info(f"    正相关: {positive_slope} ({100 * positive_slope / total_valid:.1f}%), "
                                f"负相关: {negative_slope} ({100 * negative_slope / total_valid:.1f}%)")


def main():
    """主函数"""
    # 创建配置
    config = AnalysisConfig(
        INPUT_FILE=r'D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\01_matching_results\Boreal\Autumn_mode1.csv',
        OUTPUT_DIR=r'D:\article\SynologyDrive\LAI-LST-Asymmetric\data\outCSV\02_sliding_window\Boreal',
        WINDOW_SIZE=6.0,
        STEP_SIZE=2.0,
        MIN_POINTS_GRID1=3,
        MIN_POINTS_GRID2=2,
        ANALYSIS_TYPE='both',
        ANALYSIS_VARIABLES=['lst_daily', 'lst_day', 'lst_night',
                            'le', 'et', 'sw', 'lw', 'hg', 'albedo', 'dsr'],
        USE_ROBUST_REGRESSION=True,
        LATITUDE_BAND_WIDTH=1.0,
        GRID_AGGREGATION_SIZE=2.0,
        LAI_RATE_INTERVAL=0.2,

        USE_LAND_MASK=True,
        LAND_MASK_FILE=r'D:\article\SynologyDrive\shared_data\shpData\world\world.shp'
    )

    # 创建分析器并运行
    analyzer = PairedGridAnalyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()















