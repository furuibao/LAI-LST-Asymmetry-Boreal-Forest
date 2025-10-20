"""_这个文件包含了整个项目所需要的处理工具函数_
"""

import os
import logging
from typing import Tuple, Optional

import ee
import geemap

from constants import (
    Season,
    ChangeType,
    ChangeRate,
    VegetationType,
    ClimateZone,
    PHENOLOGY_SEASON_DOY,
    DATASETS,
    VEGETATION_TYPE_CODE,
    DEFAULT_CLIMATE_ZONE,
)

# 记录异常日志
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化geemap
try:
    ee.Authenticate()
    ee.Initialize(project='ee-liupanpan')
except Exception as e:
    logger.error(f"初始化geemap失败: {e}")
    raise e


# ==========影像筛选与预处理部分==========
def get_phenology_dates(year: int, season: Season) -> Tuple[str, str]:
    """根据物候季节定义获取日期范围

    Args:
        year: 年份
        season: 季节

    Returns:
        Tuple[start_date, end_date]: 开始和结束日期字符串
    """
    doy_range = PHENOLOGY_SEASON_DOY[season]

    if season == Season.WINTER:
        # 冬季跨年：11月9日到次年4月8日
        start_date = ee.Date.fromYMD(year, 11, 9)
        end_date = ee.Date.fromYMD(year + 1, 4, 8)
    elif season == Season.SPRING:
        # 春季：4月9日到6月13日
        start_date = ee.Date.fromYMD(year, 4, 9)
        end_date = ee.Date.fromYMD(year, 6, 13)
    elif season == Season.SUMMER:
        # 夏季：6月14日到9月9日
        start_date = ee.Date.fromYMD(year, 6, 14)
        end_date = ee.Date.fromYMD(year, 9, 9)
    elif season == Season.AUTUMN:
        # 秋季：9月10日到11月8日
        start_date = ee.Date.fromYMD(year, 9, 10)
        end_date = ee.Date.fromYMD(year, 11, 8)

    return start_date, end_date


def filter_images(
        dataset_id: str,
        start_year: int,
        end_year: int,
        region: ee.FeatureCollection,
        season: Optional[Season] = None
) -> Tuple[ee.Image, ee.Image]:
    """_根据给定的参数筛选给定时间点的影像并进行质量控制_

    Args:
        dataset_id (str): _被筛选影像的ID_
        start_year (int): _第一个年份_
        end_year (int): _第二个年份_
        region (ee.FeatureCollection): _筛选区域 (Boreal)_
        season (Optional[Season], optional): _要筛选的季节_. Defaults to None.

    Returns:
        Tuple[ee.Image, ee.Image]: _筛选且质量控制后的影像元组_
    """
    try:
        # 验证输入数据集ID是否存在
        if dataset_id not in [dataset.id for dataset in DATASETS.values()]:
            raise ValueError(f"Invalid dataset ID: {dataset_id}")
        if dataset_id != DATASETS["LC"].id and season is None:
            raise ValueError("Season must be specified for non-LC dataset")

        # 加载原始数据集和区域
        collection = ee.ImageCollection(dataset_id)
        region = ee.FeatureCollection(region)

        # 使用物候定义的日期
        start_date1, end_date1 = get_phenology_dates(start_year, season)
        start_date2, end_date2 = get_phenology_dates(end_year, season)

        logger.debug(f"Filter {dataset_id} images for {season.name} season using phenology dates")

        # 对LST和LAI影像进行质量控制
        if dataset_id in [DATASETS["LST"].id, DATASETS["LAI"].id]:
            dataset_name = "LST" if dataset_id == DATASETS["LST"].id else "LAI"
            logger.debug(f"Applying quality control to {dataset_name} images")
            filtered_image1 = (
                collection.filterDate(start_date1, end_date1)
                .map(lambda image: apply_quality_mask(image, dataset_id))
                .mean()
                .clip(region)
            )
            filtered_image2 = (
                collection.filterDate(start_date2, end_date2)
                .map(lambda image: apply_quality_mask(image, dataset_id))
                .mean()
                .clip(region)
            )
        else:
            # 其他数据集暂不应用质量控制
            filtered_image1 = (
                collection.filterDate(start_date1, end_date1)
                .mean()
                .clip(region)
            )
            filtered_image2 = (
                collection.filterDate(start_date2, end_date2)
                .mean()
                .clip(region)
            )
        return filtered_image1, filtered_image2
    except ee.EEException as e:
        logger.error(f"Error processing {dataset_id} images for {season.name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {dataset_id} images for {season.name}: {str(e)}")
        raise


def filter_dsr_images(
        dataset_id: str,
        start_year: int,
        end_year: int,
        region: ee.FeatureCollection,
        season: Season
) -> Tuple[ee.Image, ee.Image]:
    """_筛选DSR影像并计算每日8个时段的平均值_

    Args:
        dataset_id (str): _DSR影像集合ID_
        start_year (int): _第一个年份_
        end_year (int): _第二个年份_
        region (ee.FeatureCollection): _筛选区域_
        season (Season): _季节_

    Returns:
        Tuple[ee.Image, ee.Image]: _处理后的DSR影像元组_
    """
    try:
        if dataset_id != DATASETS["DSR"].id:
            raise ValueError("This function is only for DSR dataset")

        # 加载DSR数据集
        collection = ee.ImageCollection(dataset_id)
        region = ee.FeatureCollection(region)

        # 使用物候定义的日期
        start_date1, end_date1 = get_phenology_dates(start_year, season)
        start_date2, end_date2 = get_phenology_dates(end_year, season)

        logger.debug(f"Filter DSR images for {season.name} season using phenology dates")

        # 定义8个时段的波段名称
        time_bands = [
            'GMT_0000_DSR', 'GMT_0300_DSR', 'GMT_0600_DSR', 'GMT_0900_DSR',
            'GMT_1200_DSR', 'GMT_1500_DSR', 'GMT_1800_DSR', 'GMT_2100_DSR'
        ]

        # 处理第一期影像
        collection1 = collection.filterDate(start_date1, end_date1).filterBounds(region)
        daily_mean1 = collection1.select(time_bands).mean()
        dsr_daily1 = daily_mean1.select(time_bands).reduce(ee.Reducer.mean()).rename('DSR')
        dsr_image1 = dsr_daily1.clip(region)

        # 处理第二期影像
        collection2 = collection.filterDate(start_date2, end_date2).filterBounds(region)
        daily_mean2 = collection2.select(time_bands).mean()
        dsr_daily2 = daily_mean2.select(time_bands).reduce(ee.Reducer.mean()).rename('DSR')
        dsr_image2 = dsr_daily2.clip(region)

        return dsr_image1, dsr_image2

    except ee.EEException as e:
        logger.error(f"Error processing DSR images for {season.name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing DSR images for {season.name}: {str(e)}")
        raise


def get_vegetation_mask(
        dataset_id: str,
        start_year: int,
        end_year: int,
        region: ee.FeatureCollection,
        vegetation_type: Optional[VegetationType]
) -> ee.Image:
    """_筛选指定时间段内始终为某种植被类型的掩膜影像_

    使用LC_Type1波段进行筛选

    Args:
        dataset_id (str): _LC影像集合ID_
        start_year (int): _开始年份_
        end_year (int): _结束年份_
        region (ee.FeatureCollection): _筛选区域_
        vegetation_type (Optional[VegetationType]): _植被类型_

    Returns:
        ee.Image: _某种植被的影像_
    """
    try:
        # 输入验证
        if dataset_id not in [dataset.id for dataset in DATASETS.values()]:
            raise ValueError(f"Invalid dataset ID: {dataset_id}")
        if dataset_id != DATASETS["LC"].id:
            raise ValueError("Input dataset must be LC dataset.")
        if vegetation_type not in VegetationType:
            raise ValueError("Invalid vegetation type.")

        lc_imageCol = ee.ImageCollection(dataset_id)
        region = ee.FeatureCollection(region)

        # 从constants读取植被类型编码
        vegetation_codes = VEGETATION_TYPE_CODE[vegetation_type]

        # 定义植被类型筛选函数 - 使用LC_Type1
        def filter_vegetation_type(image):
            mask = image.select(["LC_Type1"]).eq(vegetation_codes[0])
            for code in vegetation_codes[1:]:
                mask = mask.Or(image.select(["LC_Type1"]).eq(code))
            return mask

        # 按年份逐年处理
        years = range(start_year, end_year + 1)
        yearly_masks = []
        for year in years:
            # 筛选当年的影像
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            yearly_col = lc_imageCol.filterDate(start_date, end_date).filterBounds(region)
            # 生成当年的植被掩膜
            yearly_mask = yearly_col.map(filter_vegetation_type).mosaic()
            yearly_masks.append(yearly_mask)

        # 计算所有年份的交集(始终为某种植被类型的区域)
        persistent_mask = ee.Image(yearly_masks[0])
        for mask in yearly_masks[1:]:
            persistent_mask = persistent_mask.And(mask)

        # 裁剪到指定区域并返回
        return persistent_mask.clip(region.geometry())
    except Exception as e:
        logger.error(f"Error occurred while getting persistent vegetation mask: {e}")
        raise e


def merge_scale_filter_vegetation_with_original(
        lai_image1: ee.Image,
        lst_image1: ee.Image,
        albedo_image1: ee.Image,
        etle_image1: ee.Image,
        ndvievi_image1: ee.Image,
        dsr_image1: ee.Image,
        lai_image2: ee.Image,
        lst_image2: ee.Image,
        albedo_image2: ee.Image,
        etle_image2: ee.Image,
        ndvievi_image2: ee.Image,
        dsr_image2: ee.Image,
        vegetation_image: ee.Image,
) -> ee.Image:
    """_合并、缩放和过滤植被区域，保留两期原始值_

    Args:
        *_image1: 第一期各类影像
        *_image2: 第二期各类影像
        vegetation_image: 植被掩膜影像

    Returns:
        ee.Image: 合并后的包含两期数据的植被区域影像
    """
    try:
        # 验证输入是否正确
        if not all([
            lai_image1, lst_image1, albedo_image1, etle_image1, ndvievi_image1, dsr_image1,
            lai_image2, lst_image2, albedo_image2, etle_image2, ndvievi_image2, dsr_image2,
            vegetation_image
        ]):
            raise ValueError("All input images are required")

        # 处理第一期影像
        try:
            # LAI T1
            lai_band_t1 = lai_image1.select(["Lai_500m"]).multiply(0.1).rename("LAI_T1")
            # LST T1
            lst_day_band_t1 = (
                lst_image1.select(["LST_Day_1km"]).multiply(0.02)
                .subtract(273.15).rename("LST_Day_T1")
            )
            lst_night_band_t1 = (
                lst_image1.select(["LST_Night_1km"]).multiply(0.02)
                .subtract(273.15).rename("LST_Night_T1")
            )
            lst_daily_band_t1 = (
                lst_day_band_t1.add(lst_night_band_t1).divide(2.0).rename("LST_Daily_T1")
            )
            # Albedo T1
            bsa_band_t1 = albedo_image1.select(["Albedo_BSA_shortwave"]).multiply(0.001)
            wsa_band_t1 = albedo_image1.select(["Albedo_WSA_shortwave"]).multiply(0.001)
            albedo_band_t1 = bsa_band_t1.add(wsa_band_t1).divide(2.0).rename("Albedo_T1")
            # ETLE T1
            et_band_t1 = etle_image1.select(["ET"]).multiply(0.1).rename("ET_T1")
            le_band_t1 = etle_image1.select(["LE"]).multiply(10000).rename("LE_T1")
            # NDVIEVI T1
            ndvi_band_t1 = ndvievi_image1.select(["NDVI"]).multiply(0.0001).rename("NDVI_T1")
            evi_band_t1 = ndvievi_image1.select(["EVI"]).multiply(0.0001).rename("EVI_T1")
            # DSR T1
            dsr_band_t1 = dsr_image1.select(["DSR"]).rename("DSR_T1")

        except ee.EEException as e:
            logger.error(f"Error processing T1 image bands: {str(e)}")
            raise e

        # 处理第二期影像
        try:
            # LAI T2
            lai_band_t2 = lai_image2.select(["Lai_500m"]).multiply(0.1).rename("LAI_T2")
            # LST T2
            lst_day_band_t2 = (
                lst_image2.select(["LST_Day_1km"]).multiply(0.02)
                .subtract(273.15).rename("LST_Day_T2")
            )
            lst_night_band_t2 = (
                lst_image2.select(["LST_Night_1km"]).multiply(0.02)
                .subtract(273.15).rename("LST_Night_T2")
            )
            lst_daily_band_t2 = (
                lst_day_band_t2.add(lst_night_band_t2).divide(2.0).rename("LST_Daily_T2")
            )
            # Albedo T2
            bsa_band_t2 = albedo_image2.select(["Albedo_BSA_shortwave"]).multiply(0.001)
            wsa_band_t2 = albedo_image2.select(["Albedo_WSA_shortwave"]).multiply(0.001)
            albedo_band_t2 = bsa_band_t2.add(wsa_band_t2).divide(2.0).rename("Albedo_T2")
            # ETLE T2
            et_band_t2 = etle_image2.select(["ET"]).multiply(0.1).rename("ET_T2")
            le_band_t2 = etle_image2.select(["LE"]).multiply(10000).rename("LE_T2")
            # NDVIEVI T2
            ndvi_band_t2 = ndvievi_image2.select(["NDVI"]).multiply(0.0001).rename("NDVI_T2")
            evi_band_t2 = ndvievi_image2.select(["EVI"]).multiply(0.0001).rename("EVI_T2")
            # DSR T2
            dsr_band_t2 = dsr_image2.select(["DSR"]).rename("DSR_T2")

        except ee.EEException as e:
            logger.error(f"Error processing T2 image bands: {str(e)}")
            raise e

        # 合并所有波段
        try:
            merged_image = ee.Image.cat([
                # 第一期
                lai_band_t1, lst_day_band_t1, lst_night_band_t1, lst_daily_band_t1,
                albedo_band_t1, et_band_t1, le_band_t1, ndvi_band_t1, evi_band_t1, dsr_band_t1,
                # 第二期
                lai_band_t2, lst_day_band_t2, lst_night_band_t2, lst_daily_band_t2,
                albedo_band_t2, et_band_t2, le_band_t2, ndvi_band_t2, evi_band_t2, dsr_band_t2
            ])

            # 筛选植被区域
            merged_veg_image = ee.Image(merged_image).updateMask(vegetation_image)

            return merged_veg_image
        except ee.EEException as e:
            logger.error(f"Error merging image bands: {str(e)}")
            raise e
    except Exception as e:
        logger.error(f"Error occurred while merging, scaling and filtering vegetation area: {e}")
        raise e


def calculate_differences_with_original(merged_image: ee.Image) -> ee.Image:
    """_计算两期影像之间的差值，并保留原始值_

    Args:
        merged_image: 包含两期数据的合并影像

    Returns:
        ee.Image: 包含原始值和差值的影像
    """
    try:
        # 计算各变量的差值 (T2 - T1)
        lai_diff = merged_image.select('LAI_T2').subtract(merged_image.select('LAI_T1')).rename('LAI_diff')
        ndvi_diff = merged_image.select('NDVI_T2').subtract(merged_image.select('NDVI_T1')).rename('NDVI_diff')
        evi_diff = merged_image.select('EVI_T2').subtract(merged_image.select('EVI_T1')).rename('EVI_diff')

        lst_day_diff = merged_image.select('LST_Day_T2').subtract(merged_image.select('LST_Day_T1')).rename('LST_Day_diff')
        lst_night_diff = merged_image.select('LST_Night_T2').subtract(merged_image.select('LST_Night_T1')).rename('LST_Night_diff')
        lst_daily_diff = merged_image.select('LST_Daily_T2').subtract(merged_image.select('LST_Daily_T1')).rename('LST_Daily_diff')

        albedo_diff = merged_image.select('Albedo_T2').subtract(merged_image.select('Albedo_T1')).rename('Albedo_diff')
        et_diff = merged_image.select('ET_T2').subtract(merged_image.select('ET_T1')).rename('ET_diff')
        le_diff = merged_image.select('LE_T2').subtract(merged_image.select('LE_T1')).rename('LE_diff')
        dsr_diff = merged_image.select('DSR_T2').subtract(merged_image.select('DSR_T1')).rename('DSR_diff')

        # 计算LAI相对变化率
        lai_change_rate = lai_diff.divide(merged_image.select('LAI_T1')).rename('LAI_Change_Rate')

        # 合并所有波段（原始值 + 差值）
        result_image = merged_image.addBands([
            lai_diff, ndvi_diff, evi_diff,
            lst_day_diff, lst_night_diff, lst_daily_diff,
            albedo_diff, et_diff, le_diff, dsr_diff,
            lai_change_rate
        ])

        return result_image
    except Exception as e:
        logger.error(f"Error calculating differences: {e}")
        raise


def category_lai_diff(image_with_diff: ee.Image) -> ee.Image:
    """_对差值影像中的LAI像元进行分类_

    Args:
        image_with_diff: 包含差值的影像

    Returns:
        ee.Image: 包含LAI类别属性的影像
    """
    try:
        # 获取变化阈值以及对应的类别编码
        increase_threshold = ee.Number(ChangeRate.LAI_INCREASE_RATE.value)
        decrease_threshold = ee.Number(ChangeRate.LAI_DECREASE_RATE.value)
        increase_code = ee.Number(ChangeType.INCREASE.value)
        decrease_code = ee.Number(ChangeType.DECREASE.value)
        stable_code = ee.Number(ChangeType.NO_CHANGE.value)

        # 验证输入参数是否正确
        if not isinstance(image_with_diff, ee.Image):
            raise TypeError("image_with_diff must be an instance of ee.Image")

        # 获取LAI变化率
        lai_change_rate = image_with_diff.select(["LAI_Change_Rate"])

        # 分类LAI相对变化率
        change_type = ee.Image(stable_code)
        change_type = change_type.where(
            lai_change_rate.gt(increase_threshold), increase_code
        )
        change_type = change_type.where(
            lai_change_rate.lt(decrease_threshold), decrease_code
        )
        change_type = change_type.rename(["LAI_Change_Type"])

        # 将变化类别添加到影像中
        result_image = image_with_diff.addBands(change_type)

        return result_image
    except Exception as e:
        logger.error(f"Error classifying LAI change rate: {e}")
        raise


# ==========帮助函数==========
def get_qa_bits(
        image: ee.Image,
        start: int,
        end: int,
        new_name: str,
) -> ee.Image:
    """_提取质量控制波段_

    Args:
        image: 包含QA波段的影像
        start: 起始位
        end: 结束位
        new_name: 新波段名称

    Returns:
        ee.Image: 提取的QA波段
    """
    pattern = 0
    for i in range(start, end + 1):
        pattern += pow(2, i)
    return image.select([0], [new_name]).bitwiseAnd(pattern).rightShift(start)


def apply_quality_mask(
        image: ee.Image,
        dataset_id: str
) -> ee.Image:
    """_应用质量掩膜_

    Args:
        image: 原始影像
        dataset_id: 数据集ID

    Returns:
        ee.Image: 应用质量掩膜后的影像
    """
    if dataset_id == DATASETS["LST"].id:
        # MOD11A2 quality control
        QC = image.select("QC_Day")

        # Extract quality bits
        QA_mandatory = get_qa_bits(QC, 0, 1, "mandatory_qa")
        QA_data = get_qa_bits(QC, 2, 3, "data_quality")
        QA_emis = get_qa_bits(QC, 4, 5, "emis_error")
        QA_LST = get_qa_bits(QC, 6, 7, "lst_error")

        # Create quality mask for LST
        mask = (
            QA_mandatory.eq(0)
            .And(QA_data.eq(0).Or(QA_data.eq(1)))
            .And(QA_emis.eq(0).Or(QA_emis.eq(1)))
            .And(QA_LST.eq(0).Or(QA_LST.eq(1)))
        )
        return image.updateMask(mask)

    elif dataset_id == DATASETS["LAI"].id:
        # MOD15A2H quality control
        QC = image.select("FparLai_QC")

        # Extract quality bits
        QA_MODLAND = get_qa_bits(QC, 0, 1, "MODLAND_QC")
        QA_sensor = get_qa_bits(QC, 2, 3, "SENSOR")
        QA_deaddetector = get_qa_bits(QC, 4, 4, "DEADDETECTOR")
        QA_cloud = get_qa_bits(QC, 5, 5, "CLOUD_STATE")
        QA_cloud_shadow = get_qa_bits(QC, 6, 6, "CLOUD_SHADOW")

        # Create quality mask for LAI
        mask = (
            QA_MODLAND.eq(0)  # Good quality
            .And(QA_sensor.eq(0))  # Main algorithm used
            .And(QA_deaddetector.eq(0))  # No dead detector
            .And(QA_cloud.eq(0))  # No cloud
            .And(QA_cloud_shadow.eq(0))  # No cloud shadow
        )
        return image.updateMask(mask)

    else:
        # 其他数据集暂不应用质量掩膜
        return image