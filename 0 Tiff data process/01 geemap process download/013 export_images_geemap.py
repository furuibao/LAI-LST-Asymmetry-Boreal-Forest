"""_Export the processed images to Google drive_
"""

import os
import logging
import time
from typing import Dict, Optional

import ee
import geemap

from constants import (
    Season,
    ClimateZone,
    CATEGORIES,
    DEFAULT_CLIMATE_ZONE,
    CLIMATE_ZONES,
)

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化Earth Engine
try:
    ee.Initialize(project='XXX (Here is your GEE user name)')
except Exception as e:
    logger.error(f"初始化Earth Engine失败: {e}")
    raise


def prepare_export_image(
    processed_image: ee.Image,
    category: str
) -> ee.Image:
    """准备导出的影像，选择特定类别的波段

    Args:
        processed_image: 处理后的包含所有波段的影像
        category: 数据类别 (vegetation/temperature/energy/classification)

    Returns:
        ee.Image: 准备好导出的影像
    """
    try:
        if category not in CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        category_info = CATEGORIES[category]
        bands = category_info["bands"]
        scale_factors = category_info["scale_factors"]

        # 选择该类别的波段
        export_image = processed_image.select(bands)

        # 应用缩放因子（转换为整数以减小文件大小）
        for i, band in enumerate(bands):
            scale_factor = scale_factors[i]
            export_image = export_image.addBands(
                export_image.select(band).multiply(scale_factor).toInt16(),
                overwrite=True
            )

        # 对于classification类别，使用int8
        if category == "classification":
            export_image = export_image.toInt8()

        return export_image

    except Exception as e:
        logger.error(f"准备导出影像时出错: {e}")
        raise


def export_change_detection_to_drive(
    processed_image: ee.Image,
    year_pair: str,
    season: Season,
    region: str = "Boreal",
    drive_folder: str = "LAI_LST_exports",
    scale: int = 500,
    crs: str = "EPSG:4326",
    max_pixels: int = 1e13
) -> Dict[str, str]:
    """导出变化检测结果到Google Drive

    Args:
        processed_image: 处理后的包含所有数据的影像
        year_pair: 年份对，如 "2005-2006"
        season: 季节
        region: 区域名称（默认Boreal）
        drive_folder: Google Drive中的目标文件夹名
        scale: 导出分辨率（米）
        crs: 坐标系
        max_pixels: 最大像素数

    Returns:
        Dict: 导出任务ID字典
    """
    try:
        # 验证输入
        if region != "Boreal":
            logger.warning(f"当前版本主要支持Boreal区域，{region}区域可能需要额外配置")

        # 获取区域边界
        region_fc = ee.FeatureCollection(CLIMATE_ZONES[ClimateZone[region.upper()]])
        geometry = region_fc.geometry()

        export_tasks = {}

        # 按类别导出到Drive
        for category in CATEGORIES.keys():
            try:
                # 准备导出影像
                export_image = prepare_export_image(processed_image, category)

                # 生成文件名
                filename = f"{region}_{year_pair}_{season.value}_{category}"

                # Google Drive导出配置
                export_config = {
                    'image': export_image,
                    'description': f'Export_{filename}',
                    'folder': drive_folder,  # Drive中的文件夹
                    'fileNamePrefix': filename,
                    'scale': scale,
                    'crs': crs,
                    'region': geometry,
                    'maxPixels': max_pixels,
                    'fileFormat': 'GeoTIFF',
                    'formatOptions': {
                        'cloudOptimized': True  # 云优化格式，便于后续处理
                    }
                }

                # 启动导出任务
                task = ee.batch.Export.image.toDrive(**export_config)
                task.start()
                export_tasks[category] = task.id

                logger.info(f"导出任务已启动 - 文件: {filename}.tif, 任务ID: {task.id}")

            except Exception as e:
                logger.error(f"导出{category}类别时出错: {e}")
                export_tasks[category] = f"Error: {str(e)}"

        # 输出Drive路径提示
        logger.info(f"\n文件将保存在Google Drive的 '{drive_folder}' 文件夹中")
        logger.info("请在Earth Engine任务面板查看进度: https://code.earthengine.google.com/tasks")

        return export_tasks

    except Exception as e:
        logger.error(f"导出过程出错: {e}")
        raise


def batch_export_to_drive(
    year_pairs: list,
    seasons: list,
    processed_images_dict: Dict,
    region: str = "Boreal",
    drive_folder: str = "LAI_LST_exports"
) -> Dict:
    """批量导出多个年份和季节的变化检测结果到Drive

    Args:
        year_pairs: 年份对列表
        seasons: 季节列表
        processed_images_dict: 处理后的影像字典，键为(year_pair, season)
        region: 区域名称
        drive_folder: Drive文件夹名

    Returns:
        Dict: 所有导出任务的ID
    """
    all_tasks = {}
    successful_exports = 0
    failed_exports = 0

    total_tasks = len(year_pairs) * len(seasons)
    logger.info(f"开始批量导出: 共{total_tasks}个时期")

    for year_pair in year_pairs:
        for season in seasons:
            key = (year_pair, season)
            if key not in processed_images_dict:
                logger.warning(f"未找到 {year_pair} {season.value} 的处理结果")
                failed_exports += 1
                continue

            processed_image = processed_images_dict[key]

            try:
                tasks = export_change_detection_to_drive(
                    processed_image=processed_image,
                    year_pair=year_pair,
                    season=season,
                    region=region,
                    drive_folder=drive_folder
                )
                all_tasks[key] = tasks
                successful_exports += 1
                logger.info(f"✓ 已启动 {year_pair} {season.value} 的导出任务\n")

            except Exception as e:
                logger.error(f"✗ 导出 {year_pair} {season.value} 时出错: {e}\n")
                all_tasks[key] = f"Error: {str(e)}"
                failed_exports += 1

    # 导出摘要
    logger.info("="*60)
    logger.info(f"批量导出完成摘要:")
    logger.info(f"  成功启动: {successful_exports} 个")
    logger.info(f"  失败: {failed_exports} 个")
    logger.info(f"  Drive文件夹: {drive_folder}")
    logger.info("="*60)

    return all_tasks


def check_task_status(task_id: str) -> Dict:
    """检查单个导出任务状态

    Args:
        task_id: 任务ID

    Returns:
        Dict: 任务状态信息
    """
    try:
        task = ee.batch.Task(task_id)
        status = task.status()
        return {
            'state': status.get('state', 'UNKNOWN'),
            'creation_time': status.get('creation_timestamp_ms', 0),
            'update_time': status.get('update_timestamp_ms', 0),
            'description': status.get('description', ''),
            'error_message': status.get('error_message', '')
        }
    except Exception as e:
        return {'state': 'ERROR', 'error_message': str(e)}


def check_task_status(task_id: str) -> Dict:
    """检查单个导出任务状态

    Args:
        task_id: 任务ID

    Returns:
        Dict: 任务状态信息
    """
    try:
        # 添加短暂延时，确保任务已经注册
        time.sleep(2)

        task = ee.batch.Task(task_id)
        status = task.status()

        # 如果状态为空，返回PENDING
        if not status:
            return {'state': 'PENDING', 'error_message': ''}

        return {
            'state': status.get('state', 'UNKNOWN'),
            'creation_time': status.get('creation_timestamp_ms', 0),
            'update_time': status.get('update_timestamp_ms', 0),
            'description': status.get('description', ''),
            'error_message': status.get('error_message', '')
        }
    except Exception as e:
        # 不立即返回ERROR，可能是任务还未完全初始化
        return {'state': 'PENDING', 'error_message': f'检查中: {str(e)}'}


def monitor_all_tasks(task_dict: Dict) -> None:
    """监控所有导出任务的状态并打印摘要

    Args:
        task_dict: 任务ID字典
    """
    import time

    logger.info("\n" + "=" * 60)
    logger.info("任务状态监控:")
    logger.info("=" * 60)

    # 等待几秒让任务初始化
    logger.info("\n等待任务初始化...")
    time.sleep(5)

    status_count = {
        'READY': 0,
        'RUNNING': 0,
        'COMPLETED': 0,
        'FAILED': 0,
        'CANCELLED': 0,
        'PENDING': 0,
        'ERROR': 0
    }

    for (year_pair, season), tasks in task_dict.items():
        logger.info(f"\n{year_pair} {season.value}:")

        if isinstance(tasks, dict):
            for category, task_id in tasks.items():
                if isinstance(task_id, str) and task_id.startswith("Error"):
                    logger.error(f"  {category}: {task_id}")
                    status_count['ERROR'] += 1
                else:
                    status = check_task_status(task_id)
                    state = status['state']
                    status_count[state] = status_count.get(state, 0) + 1

                    if state == 'COMPLETED':
                        logger.info(f"  {category}: ✓ 完成")
                    elif state == 'FAILED':
                        logger.error(f"  {category}: ✗ 失败 - {status.get('error_message', '')}")
                    elif state == 'RUNNING':
                        logger.info(f"  {category}: ⟳ 运行中")
                    elif state == 'READY':
                        logger.info(f"  {category}: ⏸ 等待中")
                    elif state == 'PENDING':
                        logger.info(f"  {category}: ⏱ 初始化中")
                    else:
                        logger.info(f"  {category}: {state}")
        else:
            logger.error(f"  整体错误: {tasks}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("任务状态摘要:")
    for state, count in status_count.items():
        if count > 0:
            logger.info(f"  {state}: {count}")
    logger.info("=" * 60)

    # 提供下载建议
    logger.info("\n提示:")
    logger.info("  1. 任务已提交到Google Earth Engine")
    logger.info("  2. 访问 https://code.earthengine.google.com/tasks 查看实时进度")
    logger.info("  3. 任务完成后可在Google Drive下载文件")
    if status_count.get('PENDING', 0) > 0 or status_count.get('READY', 0) > 0:
        logger.info("  4. 任务正在队列中等待执行，请耐心等待")


if __name__ == "__main__":
    # 使用示例
    logger.info("Google Drive导出脚本已准备就绪")

    logger.info("请通过main_export.py运行完整处理流程")

