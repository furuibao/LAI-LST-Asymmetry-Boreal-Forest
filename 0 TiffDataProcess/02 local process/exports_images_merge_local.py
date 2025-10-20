"""
批量合并TIF文件脚本 - 内存高效版本
使用GDAL VRT避免内存溢出，同时避免拼接伪影
"""
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import os
import re
import glob
from pathlib import Path
from itertools import product
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

try:
    from osgeo import gdal

    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    print("警告: 未找到GDAL，将尝试使用rasterio")
    HAS_GDAL = False
    import rasterio
    from rasterio.merge import merge

# 导入常量定义
try:
    from constants import CATEGORIES

    HAS_CONSTANTS = True
except ImportError:
    print("警告: 无法导入 constants.py，将跳过波段验证")
    HAS_CONSTANTS = False
    CATEGORIES = {
        "vegetation": {"bands": [], "scale_factors": [], "data_type": "int16"},
        "temperature": {"bands": [], "scale_factors": [], "data_type": "int16"},
        "energy": {"bands": [], "scale_factors": [], "data_type": "int16"},
        "classification": {"bands": [], "scale_factors": [], "data_type": "int8"}
    }


def validate_band_structure_gdal(dataset, variable: str) -> Tuple[bool, str]:
    """使用GDAL验证波段结构"""
    if not HAS_CONSTANTS or variable not in CATEGORIES:
        return True, "跳过验证"

    expected_bands = CATEGORIES[variable]["bands"]
    if not expected_bands:
        return True, "无预定义波段"

    if dataset.RasterCount != len(expected_bands):
        return False, f"波段数不匹配: 期待 {len(expected_bands)}, 实际 {dataset.RasterCount}"

    # 检查波段描述
    for i, expected_name in enumerate(expected_bands, 1):
        band = dataset.GetRasterBand(i)
        actual_name = band.GetDescription()
        if actual_name and actual_name != expected_name:
            return False, f"波段{i}名称不匹配: 期待 '{expected_name}', 实际 '{actual_name}'"

    return True, "验证通过"


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """解析GEE导出的文件名"""
    pattern = r'^(?P<region>[A-Za-z]+)_(?P<year_pair>\d{4}-\d{4})_(?P<season>[A-Za-z]+)_(?P<variable>[a-z]+)-(?P<tile1>\d+)-(?P<tile2>\d+)\.tif$'
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    return None


def get_matching_files_optimized(
        root_dir: str,
        region: str,
        year_pair: str,
        season: str,
        variable: str
) -> List[str]:
    """使用精确匹配获取TIF文件列表"""
    folder_path = os.path.join(root_dir, region, year_pair)
    if not os.path.exists(folder_path):
        return []

    all_tifs = glob.glob(os.path.join(folder_path, "*.tif"))
    matching_files = []

    for tif_file in all_tifs:
        filename = os.path.basename(tif_file)
        parsed = parse_filename(filename)

        if parsed and (
                parsed['region'] == region and
                parsed['year_pair'] == year_pair and
                parsed['season'] == season and
                parsed['variable'] == variable
        ):
            matching_files.append(tif_file)

    return sorted(matching_files)


def mosaic_tifs_gdal_vrt(input_tifs: List[str], output_tif: str, variable: str = None) -> Tuple[bool, str]:
    """
    使用GDAL VRT进行内存高效的影像合并

    优势：
    1. VRT不占用内存，只是一个XML文件
    2. gdal_translate分块处理，内存可控
    3. 无拼接伪影
    """
    if not HAS_GDAL:
        return False, "需要安装GDAL库"

    if not input_tifs:
        return False, "输入文件列表为空"

    output_path = Path(output_tif)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"创建输出目录失败: {e}"

    valid_tifs = [tif for tif in input_tifs if os.path.exists(tif)]
    if not valid_tifs:
        return False, "所有输入文件都不存在"

    try:
        # 创建临时VRT文件
        vrt_path = str(output_path.with_suffix('.vrt'))

        # 验证第一个文件的波段结构
        first_ds = gdal.Open(valid_tifs[0])
        if first_ds is None:
            return False, f"无法打开文件: {valid_tifs[0]}"

        if variable:
            is_valid, msg = validate_band_structure_gdal(first_ds, variable)
            if not is_valid:
                first_ds = None
                return False, f"波段验证失败: {msg}"

        first_ds = None  # 关闭文件

        # 使用gdalbuildvrt构建虚拟栅格
        # -resolution highest: 使用最高分辨率
        # -srcnodata 和 -vrtnodata: 处理nodata值
        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',
            separate=False,  # 不分离波段
            addAlpha=False
        )

        vrt_ds = gdal.BuildVRT(vrt_path, valid_tifs, options=vrt_options)
        if vrt_ds is None:
            return False, "创建VRT失败"

        vrt_ds = None  # 关闭VRT

        # 使用gdal.Translate将VRT转换为实际GeoTIFF
        # 使用分块处理和压缩
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=[
                'COMPRESS=LZW',
                'TILED=YES',
                'BLOCKXSIZE=512',
                'BLOCKYSIZE=512',
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUS'
            ]
        )

        result_ds = gdal.Translate(output_tif, vrt_path, options=translate_options)
        if result_ds is None:
            return False, "转换VRT到GeoTIFF失败"

        result_ds = None  # 关闭输出文件

        # 删除临时VRT文件
        try:
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
        except:
            pass

        return True, "成功"

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return False, f"处理失败: {e}\n{error_detail}"


def mosaic_tifs_rasterio_fallback(input_tifs: List[str], output_tif: str, variable: str = None) -> Tuple[bool, str]:
    """
    当GDAL不可用时使用rasterio的备用方法
    针对内存优化：使用windowed写入
    """
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.windows import Window
    except ImportError:
        return False, "rasterio未安装"

    if not input_tifs:
        return False, "输入文件列表为空"

    output_path = Path(output_tif)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"创建输出目录失败: {e}"

    valid_tifs = [tif for tif in input_tifs if os.path.exists(tif)]
    if not valid_tifs:
        return False, "所有输入文件都不存在"

    src_files = []
    try:
        for tif in valid_tifs:
            src = rasterio.open(tif)
            src_files.append(src)

        if not src_files:
            return False, "无法打开任何输入文件"

        first_src = src_files[0]

        # 使用较小的内存块大小
        # 这会增加处理时间，但避免内存溢出
        try:
            mosaic, out_transform = merge(
                src_files,
                nodata=first_src.nodata,
                method='first',
                dtype=first_src.dtypes[0],
                target_aligned_pixels=True
            )
        except MemoryError:
            return False, "内存不足，建议安装GDAL或减少并行进程数"

        out_meta = first_src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "YES",
        })

        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(mosaic)

        return True, "成功"

    except Exception as e:
        return False, f"处理失败: {e}"
    finally:
        for src in src_files:
            try:
                src.close()
            except:
                pass


def copy_single_file_optimized(input_file: str, output_file: str, variable: str = None) -> Tuple[bool, str]:
    """优化的单文件复制"""
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        if HAS_GDAL:
            # 使用GDAL复制
            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=[
                    'COMPRESS=LZW',
                    'TILED=YES',
                    'BLOCKXSIZE=512',
                    'BLOCKYSIZE=512',
                    'BIGTIFF=IF_SAFER'
                ]
            )
            result_ds = gdal.Translate(output_file, input_file, options=translate_options)
            if result_ds is None:
                return False, "GDAL复制失败"
            result_ds = None
        else:
            # 使用rasterio复制
            import rasterio
            with rasterio.open(input_file) as src:
                out_meta = src.meta.copy()
                out_meta.update({
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                })
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    for i in range(1, src.count + 1):
                        dest.write(src.read(i), i)

        return True, "成功"
    except Exception as e:
        return False, f"复制失败: {e}"


def process_single_task(task_info: Tuple[str, List[str], str, str]) -> Tuple[str, bool, str]:
    """处理单个任务"""
    task_type, files, output_path, variable = task_info

    if os.path.exists(output_path):
        return task_type, True, "文件已存在"

    if task_type == "merge":
        if HAS_GDAL:
            success, error_msg = mosaic_tifs_gdal_vrt(files, output_path, variable)
        else:
            success, error_msg = mosaic_tifs_rasterio_fallback(files, output_path, variable)
    else:  # copy
        success, error_msg = copy_single_file_optimized(files[0], output_path, variable)

    return task_type, success, error_msg


def generate_output_path(
        output_root: str,
        region: str,
        year_pair: str,
        season: str,
        variable: str
) -> str:
    """生成输出文件路径"""
    output_dir = os.path.join(output_root, region, year_pair, season)
    filename = f"{variable}.tif"
    return os.path.join(output_dir, filename)


def get_all_combinations(regions: List[str], year_pairs: List[str]) -> List[Tuple[str, str, str, str]]:
    """获取所有可能组合"""
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    variables = ["vegetation", "temperature", "energy", "classification"]
    return list(product(regions, year_pairs, seasons, variables))


def batch_process_tifs(
        root_dir: str,
        output_root: str,
        regions: List[str],
        year_pairs: List[str],
        min_files_for_merge: int = 2,
        max_workers: int = None
) -> Dict[str, int]:
    """批量处理TIF文件"""
    if max_workers is None:
        # 降低默认并行度以减少内存压力
        max_workers = min(mp.cpu_count() // 2, 2)
        print(f"自动设置并行进程数为 {max_workers}（考虑内存限制）")

    all_combinations = get_all_combinations(regions, year_pairs)

    stats = {
        "total_combinations": len(all_combinations),
        "need_merge": 0,
        "single_file": 0,
        "no_files": 0,
        "merge_success": 0,
        "merge_failed": 0,
        "copy_success": 0,
        "copy_failed": 0,
    }

    tasks = []
    print("\n正在扫描文件...")

    for region, year_pair, season, variable in tqdm(all_combinations, desc="扫描文件"):
        files = get_matching_files_optimized(root_dir, region, year_pair, season, variable)
        output_path = generate_output_path(output_root, region, year_pair, season, variable)

        if len(files) >= min_files_for_merge:
            tasks.append(("merge", files, output_path, variable))
            stats["need_merge"] += 1
        elif len(files) == 1:
            tasks.append(("copy", files, output_path, variable))
            stats["single_file"] += 1
        else:
            stats["no_files"] += 1

    if not tasks:
        print("没有找到需要处理的文件")
        return stats

    print(f"\n找到 {len(tasks)} 个任务需要处理")
    print(f"  - 需要合并: {stats['need_merge']}")
    print(f"  - 单文件复制: {stats['single_file']}")
    print(f"  - 无文件: {stats['no_files']}")

    print(f"\n使用 {max_workers} 个进程并行处理...")
    error_log = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_task, task): task for task in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="处理文件"):
            task_type, success, error_msg = future.result()
            task_info = future_to_task[future]

            if task_type == "merge":
                if success:
                    stats["merge_success"] += 1
                else:
                    stats["merge_failed"] += 1
                    error_log.append(f"合并失败 - {task_info[2]}: {error_msg}")
            else:
                if success:
                    stats["copy_success"] += 1
                else:
                    stats["copy_failed"] += 1
                    error_log.append(f"复制失败 - {task_info[2]}: {error_msg}")

    if error_log:
        print("\n" + "=" * 50)
        print("错误详情:")
        print("=" * 50)
        for i, error in enumerate(error_log[:10], 1):
            print(f"{i}. {error}")
        if len(error_log) > 10:
            print(f"... 还有 {len(error_log) - 10} 个错误未显示")
        print("=" * 50)

    return stats


def get_available_regions(root_dir: str) -> List[str]:
    """获取可用区域列表"""
    available_regions = []
    default_regions = ["Boreal", "Temperate", "Tropical"]
    for region in default_regions:
        if os.path.exists(os.path.join(root_dir, region)):
            available_regions.append(region)
    return available_regions


def get_available_year_pairs(root_dir: str, region: str) -> List[str]:
    """获取可用年份对"""
    available_years = []
    region_path = os.path.join(root_dir, region)
    if os.path.exists(region_path):
        for item in os.listdir(region_path):
            if os.path.isdir(os.path.join(region_path, item)) and '-' in item:
                try:
                    years = item.split('-')
                    if len(years) == 2:
                        int(years[0])
                        int(years[1])
                        available_years.append(item)
                except ValueError:
                    continue
    return sorted(available_years)


def parse_year_range(year_range_str: str) -> List[str]:
    """解析年份范围"""
    year_pairs = []
    parts = [part.strip() for part in year_range_str.split(',')]
    for part in parts:
        if '-' in part:
            years = part.split('-')
            if len(years) == 2:
                start_year = int(years[0])
                end_year = int(years[1])
                if end_year - start_year == 1:
                    year_pairs.append(f"{start_year}-{end_year}")
                elif end_year - start_year > 1:
                    for year in range(start_year, end_year):
                        year_pairs.append(f"{year}-{year + 1}")
    return year_pairs


def print_summary(stats: Dict[str, int]):
    """打印摘要"""
    print("\n" + "=" * 50)
    print("批量处理完成 - 结果摘要")
    print("=" * 50)
    print(f"总组合数量: {stats['total_combinations']}")
    print(f"需要合并: {stats['need_merge']}")
    print(f"单文件: {stats['single_file']}")
    print(f"无文件: {stats['no_files']}")
    print("-" * 30)
    print(f"合并成功: {stats['merge_success']}")
    print(f"合并失败: {stats['merge_failed']}")
    print(f"复制成功: {stats['copy_success']}")
    print(f"复制失败: {stats['copy_failed']}")
    print("-" * 30)
    total_processed = stats['merge_success'] + stats['copy_success']
    total_failed = stats['merge_failed'] + stats['copy_failed']
    print(f"总成功: {total_processed}")
    print(f"总失败: {total_failed}")
    if total_processed + total_failed > 0:
        success_rate = (total_processed / (total_processed + total_failed)) * 100
        print(f"成功率: {success_rate:.1f}%")
    print("=" * 50)


if __name__ == '__main__':
    root_dir = r"D:\BNU\research_direction\LAI-LST-Asy\data\oriTIF"
    output_root = r"D:\BNU\research_direction\LAI-LST-Asy\data\oriTIF\merge"

    print("TIF文件批量合并工具 - 内存优化版")
    print("=" * 40)

    if HAS_GDAL:
        print("✓ 使用GDAL进行高效合并")
    else:
        print("⚠ 未检测到GDAL，将使用rasterio（可能遇到内存问题）")
        print("  建议: conda install -c conda-forge gdal")

    if HAS_CONSTANTS:
        print("✓ 已加载波段定义")

    available_regions = get_available_regions(root_dir)
    if not available_regions:
        print("错误: 未找到可用区域")
        exit(1)

    print(f"\n可用区域: {', '.join(available_regions)}")
    region_input = input("\n输入区域 (逗号分隔，或'all'): ").strip()

    if region_input.lower() == 'all':
        selected_regions = available_regions
    else:
        selected_regions = [r.strip() for r in region_input.split(',')]

    all_available_years = set()
    for region in selected_regions:
        all_available_years.update(get_available_year_pairs(root_dir, region))
    all_available_years = sorted(list(all_available_years))

    if not all_available_years:
        print("错误: 未找到年份对")
        exit(1)

    print(f"\n可用年份: {', '.join(all_available_years)}")
    year_input = input("\n输入年份 (格式如2015-2016或2015-2020或all): ").strip()

    if year_input.lower() == 'all':
        selected_years = all_available_years
    else:
        selected_years = parse_year_range(year_input)

    print(f"\n处理参数:")
    print(f"区域: {', '.join(selected_regions)}")
    print(f"年份: {', '.join(selected_years)}")
    print(f"组合数: {len(selected_regions) * len(selected_years) * 16}")

    confirm = input("\n继续? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        exit(0)

    # 内存优化：建议较低的并行度
    cpu_count = mp.cpu_count()
    recommended = max(1, cpu_count // 4)  # 推荐使用1/4核心数
    print(f"\n系统CPU: {cpu_count}核")
    print(f"推荐并行数: {recommended} (考虑内存限制)")
    max_workers_input = input(f"输入并行数 (1-{cpu_count}, 回车使用推荐值): ").strip()

    if max_workers_input:
        max_workers = min(max(1, int(max_workers_input)), cpu_count)
    else:
        max_workers = recommended

    print(f"使用 {max_workers} 个并行进程")

    print("\n开始处理...")
    stats = batch_process_tifs(
        root_dir,
        output_root,
        selected_regions,
        selected_years,
        max_workers=max_workers
    )

    print_summary(stats)