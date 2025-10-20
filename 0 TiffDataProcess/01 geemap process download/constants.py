"""This script contains all constants used in all TIFF process"""

import logging
from enum import Enum


# Climate regions
class ClimateZone(Enum):
    BOREAL = "Boreal"
    TEMPERATE = "Temperate"
    TROPICAL = "Tropical"


# Seasons
# class Season(Enum):
#     SPRING = "Spring"
#     SUMMER = "Summer"
#     AUTUMN = "Autumn"
#     WINTER = "Winter"


# LAI Change Types
class ChangeType(Enum):
    INCREASE = 1
    DECREASE = -1
    NO_CHANGE = 0


# LAI Change threshold
class ChangeRate(Enum):
    LAI_INCREASE_RATE = 0.15
    LAI_DECREASE_RATE = -0.15


# LAI Change threshold in 5km grid
class ChangeThresholed5km(Enum):
    COUNT_RATE = 0.7


# Vegetation types
class VegetationType(Enum):
    FOREST = "Forest"
    SHRUB = "Shrub"
    GRASS = "Grass"
    CROP = "Crop"


# Datasets info class
class DatasetInfo:
    def __init__(self, id: str, temporal: str, resolution: str, date_range: str):
        self.id = id
        self.temporal = temporal
        self.resolution = resolution
        self.date_range = date_range


# Phenological season DOY range
PHENOLOGY_SEASON_DOY = {
    Season.SPRING: (99, 165),
    Season.SUMMER: (165, 253),
    Season.AUTUMN: (253, 313),
    Season.WINTER: (313, 99),
}

# Phenological season months range
PHENOLOGY_SEASON_MONTHS = {
    Season.SPRING: [4, 5, 6],
    Season.SUMMER: [6, 7, 8, 9],
    Season.AUTUMN: [9, 10, 11],
    Season.WINTER: [11, 12, 1, 2, 3, 4]
}

# Default use of phenological season definitions
SEASON_MONTHS = PHENOLOGY_SEASON_MONTHS

# Moth to season mapping
MONTH_SEASON = {
    1: Season.WINTER, 2: Season.WINTER, 3: Season.WINTER,
    4: Season.SPRING, 5: Season.SPRING, 6: Season.SUMMER,
    7: Season.SUMMER, 8: Season.SUMMER, 9: Season.AUTUMN,
    10: Season.AUTUMN, 11: Season.WINTER, 12: Season.WINTER
}

# Vegetation type to code mapping
VEGETATION_TYPE_CODE = {
    VegetationType.FOREST: [1, 2, 3, 4, 5],
    VegetationType.SHRUB: [6, 7],
    VegetationType.GRASS: [10],
    VegetationType.CROP: [12, 14]
}

# Datasets info
DATASETS = {
    "LAI": DatasetInfo("MODIS/061/MOD15A2H", "8d", "500m", "2000-2022"),
    "LST": DatasetInfo("MODIS/061/MOD11A2", "8d", "1km", "2000-2025"),
    "LC": DatasetInfo("MODIS/061/MCD12Q1", "yr", "500m", "2001-2023"),
    "Albedo": DatasetInfo("MODIS/061/MCD43A3", "d", "500m", "2000-2024"),
    "ETLE": DatasetInfo("MODIS/061/MOD16A2GF", "8d", "500m", "2000-2023"),
    "NDVIEVI": DatasetInfo("MODIS/061/MYD13A1", "16d", "500m", "2002-2024"),
    "DSR": DatasetInfo("MODIS/062/MCD18A1", "d", "1km", "2000-2025")
}

# The path of climate region SHP data (you need to switch to your own GEE asset path here)
CLIMATE_ZONES = {
    ClimateZone.BOREAL: "XXX",
    ClimateZone.TEMPERATE: "XXX",
    ClimateZone.TROPICAL: "XXX"
}

# Mapping of band names export ed from GEE to names used in local analysis
BAND_NAME_MAPPING = {
    # Vegetation bands
    'LAI_T1': 'LAI_Begin',      # T1 LAI as the initial value
    'LAI_T2': 'LAI_End',        # T2 LAI as the end value
    'LAI_diff': 'LAI_diff',
    'LAI_Change_Rate': 'LAI_Change_Rate',
    'NDVI_T1': 'NDVI_Begin',
    'NDVI_T2': 'NDVI_End',
    'NDVI_diff': 'NDVI_diff',
    'EVI_T1': 'EVI_Begin',
    'EVI_T2': 'EVI_End',
    'EVI_diff': 'EVI_diff',

    # Temperature bands
    'LST_Day_T1': 'LST_Day_Begin',
    'LST_Day_T2': 'LST_Day_End',
    'LST_Day_diff': 'LST_Day_diff',
    'LST_Night_T1': 'LST_Night_Begin',
    'LST_Night_T2': 'LST_Night_End',
    'LST_Night_diff': 'LST_Night_diff',
    'LST_Daily_T1': 'LST_Daily_Begin',
    'LST_Daily_T2': 'LST_Daily_End',
    'LST_Daily_diff': 'LST_Daily_diff',

    # Energy bands
    'Albedo_T1': 'Albedo_Begin',
    'Albedo_T2': 'Albedo_End',
    'Albedo_diff': 'Albedo_diff',
    'ET_T1': 'ET_Begin',
    'ET_T2': 'ET_End',
    'ET_diff': 'ET_diff',
    'LE_T1': 'LE_Begin',
    'LE_T2': 'LE_End',
    'LE_diff': 'LE_diff',
    'DSR_T1': 'DSR_Begin',
    'DSR_T2': 'DSR_End',
    'DSR_diff': 'DSR_diff',

    # Classification
    'LAI_Change_Type': 'LAI_Change_Type'
}

# Data category definition
CATEGORIES = {
    "vegetation": {
        "bands": [
            "LAI_T1", "NDVI_T1", "EVI_T1",
            "LAI_T2", "NDVI_T2", "EVI_T2",
            "LAI_diff", "NDVI_diff", "EVI_diff",
            "LAI_Change_Rate"
        ],
        "scale_factors": [
            1000, 10000, 10000,
            1000, 10000, 10000,
            1000, 10000, 10000,
            10000
        ],
        "data_type": "int16",
        "description": "植被指数原始值和变化"
    },
    "temperature": {
        "bands": [
            "LST_Day_T1", "LST_Night_T1", "LST_Daily_T1",
            "LST_Day_T2", "LST_Night_T2", "LST_Daily_T2",
            "LST_Day_diff", "LST_Night_diff", "LST_Daily_diff"
        ],
        "scale_factors": [
            100, 100, 100,
            100, 100, 100,
            100, 100, 100
        ],
        "data_type": "int16",
        "description": "地表温度原始值和变化"
    },
    "energy": {
        "bands": [
            "Albedo_T1", "ET_T1", "LE_T1", "DSR_T1",
            "Albedo_T2", "ET_T2", "LE_T2", "DSR_T2",
            "Albedo_diff", "ET_diff", "LE_diff", "DSR_diff"
        ],
        "scale_factors": [
            10000, 10, 1, 10,
            10000, 10, 1, 10,
            10000, 10, 1, 10
        ],
        "data_type": "int16",
        "description": "能量平衡组分原始值和变化"
    },
    "classification": {
        "bands": ["LAI_Change_Type"],
        "scale_factors": [1],
        "data_type": "int8",
        "description": "LAI变化类型分类"
    }
}

# Supported regions
REGIONS = ["Boreal", "Temperate", "Tropical"]

# Default region
DEFAULT_REGION = "Boreal"
DEFAULT_CLIMATE_ZONE = ClimateZone.BOREAL

# Supported seasons
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]

# Grid size configuration
DEFAULT_SMALL_GRID_SIZE_KM = 5
DEFAULT_LARGE_GRID_SIZE_KM = 50
PIXEL_SIZE_M = 500

# The paras for IDW
DEFAULT_THRESHOLD = 0.7
MIN_DISTANCE_M = 500.0  

# Time range
YEAR_RANGE = range(2005, 2020)
MONTH_RANGE = range(1, 13)

YEAR_PAIRS = [
    "2005-2006", "2006-2007", "2007-2008", "2008-2009", "2009-2010",
    "2010-2011", "2011-2012", "2012-2013", "2013-2014", "2014-2015",
    "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020"
]

# Chunk processing configuration
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_OVERLAP = 100
MAX_MEMORY_MB = 2048

# NoData值定义

NODATA_VALUE = -32768
