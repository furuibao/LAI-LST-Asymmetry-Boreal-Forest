"""
主处理和导出流程脚本
用于执行LAI-LST非对称性分析的完整数据处理流程
"""

import os
import sys
import logging
import time
import traceback
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import ee
import geemap

# 导入项目模块
from constants import (
    Season,
    ClimateZone,
    VegetationType,
    DATASETS,
    CLIMATE_ZONES,
    DEFAULT_CLIMATE_ZONE,
    YEAR_PAIRS,
    SEASONS,
    DEFAULT_REGION,
)
from dataprocessor_geemap import (
    filter_images,
    filter_dsr_images,
    get_vegetation_mask,
    merge_scale_filter_vegetation_with_original,
    calculate_differences_with_original,
    category_lai_diff,
)
from export_images_geemap import (
    export_change_detection_to_drive,
    monitor_all_tasks,
)

# 配置日志 - 确保使用UTF-8编码并立即刷新
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)  # 明确使用stdout
    ],
    force=True  # 强制重新配置
)
logger = logging.getLogger(__name__)

# 确保日志立即输出
for handler in logger.handlers:
    handler.flush()

print("=" * 60)
print("程序启动中...")
print("=" * 60)

# 初始化Earth Engine
try:
    print("正在初始化Earth Engine...")
    ee.Initialize(project='ee-liupanpan')
    print("✓ Earth Engine初始化成功")
    logger.info("Earth Engine初始化成功")
except Exception as e:
    print(f"✗ Earth Engine初始化失败: {e}")
    logger.error(f"Earth Engine初始化失败: {e}")
    sys.exit(1)


class LAILSTProcessor:
    """LAI-LST非对称性分析处理器"""

    def __init__(
        self,
        climate_zone: ClimateZone = DEFAULT_CLIMATE_ZONE,
        vegetation_type: VegetationType = VegetationType.FOREST,
        drive_folder: str = "LAI_LST_exports"
    ):
        """初始化处理器"""
        self.climate_zone = climate_zone
        self.vegetation_type = vegetation_type
        self.region_name = climate_zone.value
        self.drive_folder = drive_folder

        # 获取区域边界
        try:
            self.region = ee.FeatureCollection(CLIMATE_ZONES[climate_zone])
            logger.info(f"处理器初始化完成 - 区域: {self.region_name}, 植被类型: {vegetation_type.value}")
        except Exception as e:
            logger.error(f"初始化处理器失败: {e}")
            raise

    def process_single_period(
        self,
        start_year: int,
        end_year: int,
        season: Season
    ) -> Optional[ee.Image]:
        """处理单个时期的数据"""
        try:
            print(f"\n  开始处理 {start_year}-{end_year} {season.value}")
            logger.info(f"开始处理 {start_year}-{end_year} {season.value}")

            # 1. 筛选各类影像
            print("    - 筛选影像数据...")
            logger.info("  筛选影像数据...")

            # LAI (包含质量控制)
            lai_image1, lai_image2 = filter_images(
                DATASETS["LAI"].id, start_year, end_year, self.region, season
            )
            print("      ✓ LAI影像")

            # LST (包含质量控制)
            lst_image1, lst_image2 = filter_images(
                DATASETS["LST"].id, start_year, end_year, self.region, season
            )
            print("      ✓ LST影像")

            # Albedo
            albedo_image1, albedo_image2 = filter_images(
                DATASETS["Albedo"].id, start_year, end_year, self.region, season
            )
            print("      ✓ Albedo影像")

            # ET/LE
            etle_image1, etle_image2 = filter_images(
                DATASETS["ETLE"].id, start_year, end_year, self.region, season
            )
            print("      ✓ ET/LE影像")

            # NDVI/EVI
            ndvievi_image1, ndvievi_image2 = filter_images(
                DATASETS["NDVIEVI"].id, start_year, end_year, self.region, season
            )
            print("      ✓ NDVI/EVI影像")

            # DSR (特殊处理)
            dsr_image1, dsr_image2 = filter_dsr_images(
                DATASETS["DSR"].id, start_year, end_year, self.region, season
            )
            print("      ✓ DSR影像")

            # 2. 获取植被掩膜
            print(f"    - 获取{self.vegetation_type.value}植被掩膜...")
            logger.info(f"  获取{self.vegetation_type.value}植被掩膜...")
            vegetation_mask = get_vegetation_mask(
                DATASETS["LC"].id, start_year, end_year, self.region, self.vegetation_type
            )
            print("      ✓ 植被掩膜")

            # 3. 合并、缩放和筛选植被区域
            print("    - 合并和缩放影像...")
            logger.info("  合并和缩放影像...")
            merged_image = merge_scale_filter_vegetation_with_original(
                lai_image1, lst_image1, albedo_image1, etle_image1, ndvievi_image1, dsr_image1,
                lai_image2, lst_image2, albedo_image2, etle_image2, ndvievi_image2, dsr_image2,
                vegetation_mask
            )
            print("      ✓ 影像合并")

            # 4. 计算差值
            print("    - 计算变化量...")
            logger.info("  计算变化量...")
            image_with_diff = calculate_differences_with_original(merged_image)
            print("      ✓ 变化计算")

            # 5. LAI变化分类
            print("    - 分类LAI变化类型...")
            logger.info("  分类LAI变化类型...")
            final_image = category_lai_diff(image_with_diff)
            print("      ✓ 分类完成")

            print(f"  ✓ 成功完成 {start_year}-{end_year} {season.value}")
            logger.info(f"✓ 完成处理 {start_year}-{end_year} {season.value}")
            return final_image

        except Exception as e:
            print(f"  ✗ 处理失败: {start_year}-{end_year} {season.value}")
            print(f"    错误: {str(e)}")
            logger.error(f"处理 {start_year}-{end_year} {season.value} 时出错: {e}")
            logger.error(traceback.format_exc())
            return None

    def process_batch_with_delay(
        self,
        year_pairs: List[str],
        seasons: List[Season],
        delay_seconds: int = 10
    ) -> Dict[Tuple[str, Season], ee.Image]:
        """批量处理，带延时"""
        processed_images = {}
        total_pairs = len(year_pairs)
        total_tasks = total_pairs * len(seasons)
        current_task = 0

        print("\n" + "=" * 60)
        print(f"批量处理配置:")
        print(f"  - 年份对数量: {total_pairs}")
        print(f"  - 季节数量: {len(seasons)}")
        print(f"  - 总任务数: {total_tasks}")
        print(f"  - 年份对间延时: {delay_seconds}秒")
        print("=" * 60)

        for pair_idx, year_pair in enumerate(year_pairs, 1):
            print(f"\n{'='*60}")
            print(f"处理年份对 [{pair_idx}/{total_pairs}]: {year_pair}")
            print(f"{'='*60}")

            start_year, end_year = map(int, year_pair.split('-'))
            year_pair_start_time = datetime.now()
            successful_seasons = []
            failed_seasons = []

            for season in seasons:
                current_task += 1
                progress_percent = (current_task / total_tasks) * 100

                print(f"\n任务 [{current_task}/{total_tasks}] (进度: {progress_percent:.1f}%)")
                print(f"处理: {year_pair} - {season.value}")

                result = self.process_single_period(start_year, end_year, season)
                if result is not None:
                    processed_images[(year_pair, season)] = result
                    successful_seasons.append(season.value)
                else:
                    failed_seasons.append(season.value)

            # 年份对处理完成统计
            year_pair_end_time = datetime.now()
            year_pair_duration = year_pair_end_time - year_pair_start_time

            print(f"\n{year_pair} 处理完成:")
            print(f"  - 成功: {len(successful_seasons)}个季节")
            if successful_seasons:
                print(f"    {', '.join(successful_seasons)}")
            if failed_seasons:
                print(f"  - 失败: {len(failed_seasons)}个季节")
                print(f"    {', '.join(failed_seasons)}")
            print(f"  - 用时: {year_pair_duration}")

            # 如果不是最后一个年份对，添加延时
            if pair_idx < total_pairs:
                print(f"\n等待 {delay_seconds} 秒后处理下一个年份对...")
                sys.stdout.flush()  # 确保输出
                for i in range(delay_seconds, 0, -1):
                    print(f"\r倒计时: {i} 秒  ", end="", flush=True)
                    time.sleep(1)
                print("\r" + " " * 20 + "\r", end="", flush=True)
                print("继续处理...")

        # 最终统计
        print(f"\n{'='*60}")
        print(f"批量处理完成统计:")
        print(f"  - 总任务数: {total_tasks}")
        print(f"  - 成功: {len(processed_images)}")
        print(f"  - 失败: {total_tasks - len(processed_images)}")
        if total_tasks > 0:
            print(f"  - 成功率: {(len(processed_images)/total_tasks*100):.1f}%")
        print(f"{'='*60}")

        return processed_images

    def export_results(self, processed_images: Dict) -> Dict:
        """导出结果"""
        export_tasks = {}

        if not processed_images:
            print("\n没有可导出的数据")
            return export_tasks

        print(f"\n开始导出 {len(processed_images)} 个结果到 Google Drive")
        print(f"目标文件夹: {self.drive_folder}")

        for idx, ((year_pair, season), image) in enumerate(processed_images.items(), 1):
            print(f"\n导出 [{idx}/{len(processed_images)}]: {year_pair} {season.value}")

            try:
                tasks = export_change_detection_to_drive(
                    processed_image=image,
                    year_pair=year_pair,
                    season=season,
                    region=self.region_name,
                    drive_folder=self.drive_folder
                )
                export_tasks[(year_pair, season)] = tasks
                print(f"  ✓ 导出任务已提交")

            except Exception as e:
                print(f"  ✗ 导出失败: {e}")
                logger.error(f"导出失败: {year_pair} {season.value} - {e}")
                export_tasks[(year_pair, season)] = f"Error: {str(e)}"

        return export_tasks


def main():
    """主函数"""

    # ========== 配置区域 ==========
    MODE = 'all'  # 'test', 'partial', 或 'all'
    VEGETATION_TYPE = VegetationType.FOREST
    DRIVE_FOLDER = "LAI_LST_exports"
    DELAY_BETWEEN_PAIRS = 10
    # ==============================

    print("\n" + "=" * 60)
    print("LAI-LST非对称性分析数据处理")
    print("=" * 60)

    # 设置处理参数
    if MODE == 'test':
        year_pairs_to_process = [YEAR_PAIRS[0]]
        seasons_to_process = [Season.SUMMER]
        print("模式: 测试模式")
    elif MODE == 'partial':
        year_pairs_to_process = YEAR_PAIRS[:3]
        seasons_to_process = [Season[s.upper()] for s in SEASONS]
        print("模式: 部分处理")
    else:
        year_pairs_to_process = YEAR_PAIRS
        seasons_to_process = [Season[s.upper()] for s in SEASONS]
        print("模式: 完整处理")

    print(f"\n配置:")
    print(f"  - 区域: {DEFAULT_CLIMATE_ZONE.value}")
    print(f"  - 植被: {VEGETATION_TYPE.value}")
    print(f"  - 年份对: {year_pairs_to_process}")
    print(f"  - 季节: {[s.value for s in seasons_to_process]}")
    print(f"  - Drive文件夹: {DRIVE_FOLDER}")

    # 创建处理器
    try:
        processor = LAILSTProcessor(
            climate_zone=DEFAULT_CLIMATE_ZONE,
            vegetation_type=VEGETATION_TYPE,
            drive_folder=DRIVE_FOLDER
        )
    except Exception as e:
        print(f"\n创建处理器失败: {e}")
        sys.exit(1)

    start_time = datetime.now()
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 改进的处理流程：边处理边导出
    total_pairs = len(year_pairs_to_process)
    all_export_tasks = {}

    for pair_idx, year_pair in enumerate(year_pairs_to_process, 1):
        print(f"\n{'=' * 60}")
        print(f"处理年份对 [{pair_idx}/{total_pairs}]: {year_pair}")
        print(f"{'=' * 60}")

        start_year, end_year = map(int, year_pair.split('-'))
        processed_images_for_pair = {}

        # 处理该年份对的所有季节
        for season in seasons_to_process:
            print(f"\n处理: {year_pair} - {season.value}")
            result = processor.process_single_period(start_year, end_year, season)
            if result is not None:
                processed_images_for_pair[(year_pair, season)] = result
                print(f"  ✓ 处理成功")
            else:
                print(f"  ✗ 处理失败")

        # 立即导出该年份对的结果
        if processed_images_for_pair:
            print(f"\n导出 {year_pair} 的结果...")
            for (yp, s), image in processed_images_for_pair.items():
                try:
                    print(f"  导出 {s.value}...")
                    tasks = export_change_detection_to_drive(
                        processed_image=image,
                        year_pair=yp,
                        season=s,
                        region=processor.region_name,
                        drive_folder=DRIVE_FOLDER
                    )
                    all_export_tasks[(yp, s)] = tasks
                    print(f"    ✓ 已提交到GEE")
                except Exception as e:
                    print(f"    ✗ 导出失败: {e}")
                    all_export_tasks[(yp, s)] = f"Error: {str(e)}"

        # 显示当前进度
        print(f"\n{year_pair} 完成")
        print(f"总进度: {pair_idx}/{total_pairs} 年份对")

        # 如果不是最后一个年份对，添加延时
        if pair_idx < total_pairs:
            print(f"\n等待 {DELAY_BETWEEN_PAIRS} 秒...")
            for i in range(DELAY_BETWEEN_PAIRS, 0, -1):
                print(f"\r倒计时: {i} 秒  ", end="", flush=True)
                time.sleep(1)
            print("\r" + " " * 20 + "\r", end="", flush=True)

    # 完成
    end_time = datetime.now()
    print(f"\n{'=' * 60}")
    print(f"所有处理和导出完成!")
    print(f"总用时: {end_time - start_time}")
    print(f"{'=' * 60}")

    # 最终统计
    print(f"\n最终统计:")
    print(f"  - 导出任务总数: {len(all_export_tasks)}")
    print(f"\n提示:")
    print(f"  1. 所有任务已提交到Google Earth Engine")
    print(f"  2. 访问 https://code.earthengine.google.com/tasks 查看进度")
    print(f"  3. 文件将保存在Google Drive的 '{DRIVE_FOLDER}' 文件夹中")


if __name__ == "__main__":
    main()