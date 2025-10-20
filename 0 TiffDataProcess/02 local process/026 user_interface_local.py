"""更新的交互式用户界面 - 修复版本"""

import argparse
import logging
import os
import sys
from typing import List, Optional
from pathlib import Path

# 设置GDAL环境变量
def setup_gdal_environment():
    """设置GDAL环境变量"""
    if 'GDAL_DATA' not in os.environ:
        try:
            import rasterio
            from rasterio._env import get_gdal_data_dir
            gdal_data_dir = get_gdal_data_dir()
            if gdal_data_dir and os.path.exists(gdal_data_dir):
                os.environ['GDAL_DATA'] = gdal_data_dir
                print(f"已自动设置GDAL_DATA: {gdal_data_dir}")
        except Exception:
            possible_paths = [
                '/usr/share/gdal',
                '/usr/local/share/gdal',
                os.path.join(sys.prefix, 'share', 'gdal'),
                os.path.join(sys.prefix, 'Library', 'share', 'gdal'),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    os.environ['GDAL_DATA'] = path
                    print(f"已自动设置GDAL_DATA: {path}")
                    break
            else:
                print("警告: 无法自动找到GDAL数据目录,可能会出现GDAL警告")

setup_gdal_environment()

from constants import (
    REGIONS, SEASONS, DEFAULT_LARGE_GRID_SIZE_KM, DEFAULT_SMALL_GRID_SIZE_KM,
    DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, YEAR_PAIRS
)
from main_analyzer_local import OptimizedVegetationClimateAnalyzer


def interactive_input():
    """交互式输入参数"""
    print("=" * 60)
    print("        植被-气候相互作用分析工具")
    print("        (支持智能分块处理)")
    print("=" * 60)
    print()

    # 选择分析模式
    print("请选择分析模式:")
    print("1. 传统模式 - 一次性加载所有数据 (适合小数据集)")
    print("2. 优化模式 - 增量加载 + 智能分块处理 (推荐,节省内存)")

    while True:
        mode_choice = input("请选择模式 (1/2) [默认: 2]: ").strip()
        if not mode_choice:
            use_optimized = True
            break
        elif mode_choice == '1':
            use_optimized = False
            print("注意: 传统模式已弃用,将使用基本优化策略")
            use_optimized = True  # 强制使用优化模式
            break
        elif mode_choice == '2':
            use_optimized = True
            break
        else:
            print("请输入 1 或 2")
            continue

    mode_name = "优化模式(智能分块)" if use_optimized else "传统模式"
    print(f"已选择: {mode_name}")

    # 分块处理参数设置(仅优化模式)
    chunk_size = DEFAULT_CHUNK_SIZE
    overlap = DEFAULT_OVERLAP

    if use_optimized:
        print(f"\n分块处理参数设置:")
        print(f"分块处理将在检测到大文件时自动启用")

        while True:
            chunk_input = input(f"分块大小(像素) [默认: {DEFAULT_CHUNK_SIZE}]: ").strip()
            if not chunk_input:
                chunk_size = DEFAULT_CHUNK_SIZE
                break
            try:
                chunk_size = int(chunk_input)
                if chunk_size < 256:
                    print("分块大小不能小于256像素")
                    continue
                if chunk_size > 4096:
                    print("警告: 分块大小过大可能导致内存不足")
                break
            except ValueError:
                print("请输入有效整数")
                continue

        while True:
            overlap_input = input(f"重叠像素数 [默认: {DEFAULT_OVERLAP}]: ").strip()
            if not overlap_input:
                overlap = DEFAULT_OVERLAP
                break
            try:
                overlap = int(overlap_input)
                if overlap < 0:
                    print("重叠像素数不能为负数")
                    continue
                if overlap >= chunk_size // 2:
                    print("重叠像素数不能超过分块大小的一半")
                    continue
                break
            except ValueError:
                print("请输入有效整数")
                continue

    # 输入数据目录
    while True:
        data_dir = input("\n请输入数据目录路径: ").strip()
        if not data_dir:
            print("数据目录不能为空,请重新输入")
            continue
        if not Path(data_dir).exists():
            print(f"目录不存在: {data_dir}")
            continue
        if not Path(data_dir).is_dir():
            print(f"不是有效目录: {data_dir}")
            continue
        break

    # 输入输出目录
    while True:
        output_dir = input("请输入输出目录路径: ").strip()
        if not output_dir:
            print("输出目录不能为空,请重新输入")
            continue
        break

    # 内存和分块估算
    if use_optimized:
        print(f"\n正在估算内存使用和分块情况...")
        try:
            temp_analyzer = OptimizedVegetationClimateAnalyzer(data_dir, output_dir, chunk_size, overlap)

            # 查找第一个可用的数据进行估算
            sample_found = False
            for region in REGIONS:
                if sample_found:
                    break
                for year_pair in YEAR_PAIRS[:3]:  # 只检查前3个年份对
                    if sample_found:
                        break
                    for season in SEASONS:
                        memory_info = temp_analyzer.estimate_memory_savings(region, year_pair, season)
                        processing_stats = temp_analyzer.get_processing_statistics(region, year_pair, season)

                        if memory_info and processing_stats:
                            print(f"\n内存使用估算 (基于样本数据: {region}/{year_pair}/{season}):")
                            print(f"  传统模式: {memory_info['traditional_memory']}")
                            print(f"  优化模式: {memory_info['optimized_memory']}")
                            print(f"  节省内存: {memory_info['memory_savings']} ({memory_info['savings_percent']})")
                            print(f"  分块处理: {memory_info['chunked_processing']}")
                            print(f"  估算分块数: {processing_stats.get('estimated_chunks', 0)}")
                            sample_found = True
                            break

            if not sample_found:
                print("未找到可用数据进行估算")

        except Exception as e:
            print(f"内存估算失败: {e}")

    # 选择区域
    print(f"\n可选区域: {', '.join(REGIONS)}")
    while True:
        regions_input = input("请选择区域 (用逗号分隔,或输入 'all'): ").strip()
        if not regions_input:
            print("区域不能为空,请重新输入")
            continue

        if regions_input.lower() == 'all':
            regions = REGIONS
            break
        else:
            regions = [r.strip() for r in regions_input.split(',')]
            invalid_regions = [r for r in regions if r not in REGIONS]
            if invalid_regions:
                print(f"无效区域: {invalid_regions}")
                continue
            break

    # 选择年份对
    print(f"\n年份对格式:")
    print(f"  - 单个: 2010-2011")
    print(f"  - 多个: 2005-2006,2010-2011")
    print(f"  - 范围: 2010-2011:2015-2016")
    print(f"  - 全部: all")

    while True:
        year_pairs_input = input("请输入年份对: ").strip()
        if not year_pairs_input:
            print("年份对不能为空,请重新输入")
            continue

        try:
            if year_pairs_input.lower() == 'all':
                year_pairs = YEAR_PAIRS
            else:
                year_pairs = parse_year_pairs(year_pairs_input)
            break
        except Exception as e:
            print(f"年份对格式错误: {e}")
            continue

    # 选择季节
    print(f"\n可选季节: {', '.join(SEASONS)}")
    while True:
        seasons_input = input("请选择季节 (用逗号分隔,或输入 'all'): ").strip()
        if not seasons_input:
            print("季节不能为空,请重新输入")
            continue

        if seasons_input.lower() == 'all':
            seasons = SEASONS
            break
        else:
            seasons = [s.strip() for s in seasons_input.split(',')]
            invalid_seasons = [s for s in seasons if s not in SEASONS]
            if invalid_seasons:
                print(f"无效季节: {invalid_seasons}")
                continue
            break

    # 网格参数设置
    print(f"\n网格参数设置 (按回车使用默认值):")

    # 大网格大小
    while True:
        large_grid_input = input(f"大网格大小(km) [默认: {DEFAULT_LARGE_GRID_SIZE_KM}]: ").strip()
        if not large_grid_input:
            large_grid_size_km = DEFAULT_LARGE_GRID_SIZE_KM
            break
        try:
            large_grid_size_km = int(large_grid_input)
            if large_grid_size_km <= 0:
                print("大网格大小必须大于0")
                continue
            break
        except ValueError:
            print("请输入有效整数")
            continue

    # 小网格大小
    while True:
        small_grid_input = input(f"小网格大小(km) [默认: {DEFAULT_SMALL_GRID_SIZE_KM}]: ").strip()
        if not small_grid_input:
            small_grid_size_km = DEFAULT_SMALL_GRID_SIZE_KM
            break
        try:
            small_grid_size_km = int(small_grid_input)
            if small_grid_size_km <= 0:
                print("小网格大小必须大于0")
                continue
            if small_grid_size_km >= large_grid_size_km:
                print(f"小网格大小必须小于大网格大小({large_grid_size_km}km)")
                continue
            break
        except ValueError:
            print("请输入有效整数")
            continue

    # 高级参数
    print(f"\n高级参数设置 (按回车使用默认值):")

    # 分类阈值
    while True:
        threshold_input = input("网格分类阈值 [默认: 0.7]: ").strip()
        if not threshold_input:
            threshold = 0.7
            break
        try:
            threshold = float(threshold_input)
            if not 0 < threshold <= 1:
                print("阈值必须在0-1之间")
                continue
            break
        except ValueError:
            print("请输入有效数字")
            continue

    # 排除零值
    exclude_zero_input = input("是否排除零值网格? (y/N) [默认: N]: ").strip().lower()
    exclude_zero = exclude_zero_input in ['y', 'yes', '是']

    # 日志级别
    log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    print(f"可选日志级别: {', '.join(log_levels)}")
    while True:
        log_level_input = input("日志级别 [默认: INFO]: ").strip().upper()
        if not log_level_input:
            log_level = 'INFO'
            break
        if log_level_input in log_levels:
            log_level = log_level_input
            break
        else:
            print(f"无效日志级别,请选择: {log_levels}")
            continue

    # 显示配置摘要
    print(f"\n" + "=" * 60)
    print(f"配置摘要:")
    print(f"=" * 60)
    print(f"分析模式: {mode_name}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"区域: {regions}")
    print(f"年份对: {year_pairs[:3]}{'...' if len(year_pairs) > 3 else ''} (共{len(year_pairs)}个)")
    print(f"季节: {seasons}")
    print(f"大网格大小: {large_grid_size_km}km")
    print(f"小网格大小: {small_grid_size_km}km")
    print(f"分类阈值: {threshold}")
    print(f"排除零值: {'是' if exclude_zero else '否'}")
    print(f"日志级别: {log_level}")

    if use_optimized:
        print(f"分块大小: {chunk_size}像素")
        print(f"重叠像素: {overlap}像素")
        print(f"内存优化: 启用")
        print(f"智能分块: 自适应启用")
    print(f"=" * 60)

    # 确认开始
    while True:
        confirm = input("\n是否开始分析? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes', '是']:
            break
        elif confirm in ['n', 'no', '否']:
            print("分析已取消")
            return None
        else:
            print("请输入 y/yes 或 n/no")
            continue

    return {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'regions': regions,
        'year_pairs': year_pairs,
        'seasons': seasons,
        'large_grid_size_km': large_grid_size_km,
        'small_grid_size_km': small_grid_size_km,
        'threshold': threshold,
        'exclude_zero': exclude_zero,
        'log_level': log_level,
        'use_optimized': use_optimized,
        'chunk_size': chunk_size,
        'overlap': overlap
    }


def parse_year_pairs(year_str: str) -> List[str]:
    """解析年份对字符串"""
    year_pairs = []

    for part in year_str.split(','):
        part = part.strip()

        if ':' in part:
            # 范围格式: 2010-2011:2015-2016
            start_pair, end_pair = part.split(':')
            start_year = int(start_pair.split('-')[0])
            end_year = int(end_pair.split('-')[0])

            for year in range(start_year, end_year + 1):
                year_pairs.append(f"{year}-{year + 1}")
        else:
            # 单个年份对: 2010-2011
            if '-' in part and len(part.split('-')) == 2:
                try:
                    start_year, end_year = map(int, part.split('-'))
                    if end_year == start_year + 1:
                        year_pairs.append(part)
                    else:
                        raise ValueError(f"年份对格式错误: {part},应为连续年份如 '2005-2006'")
                except ValueError as e:
                    raise ValueError(f"年份对解析错误: {part}。{str(e)}")
            else:
                raise ValueError(f"年份对格式错误: {part},应为 'YYYY-YYYY' 格式")

    return sorted(list(set(year_pairs)))


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="植被-气候相互作用分析工具 (支持智能分块处理)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式:
  1. 交互式模式: python user_interface_local.py
  2. 命令行模式: python user_interface_local.py --data-dir /path/to/data --output-dir /path/to/output [其他参数]

命令行参数示例:
  python user_interface_local.py --data-dir /data --output-dir /results --regions Boreal --chunk-size 2048
        """
    )

    parser.add_argument('--data-dir', type=str, help='输入数据根目录路径')
    parser.add_argument('--output-dir', type=str, help='输出结果根目录路径')
    parser.add_argument('--regions', type=str, default='all', help=f'要分析的区域 (默认: all)')
    parser.add_argument('--year-pairs', type=str, default='all', help='要分析的年份对 (默认: all)')
    parser.add_argument('--seasons', type=str, default='all', help='要分析的季节 (默认: all)')
    parser.add_argument('--large-grid', type=int, default=DEFAULT_LARGE_GRID_SIZE_KM,
                       help=f'大网格大小(km) (默认: {DEFAULT_LARGE_GRID_SIZE_KM})')
    parser.add_argument('--small-grid', type=int, default=DEFAULT_SMALL_GRID_SIZE_KM,
                       help=f'小网格大小(km) (默认: {DEFAULT_SMALL_GRID_SIZE_KM})')
    parser.add_argument('--threshold', type=float, default=0.7, help='网格分类阈值 (默认: 0.7)')
    parser.add_argument('--exclude-zero', action='store_true', help='排除零值网格')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别 (默认: INFO)')
    parser.add_argument('--interactive', action='store_true', help='强制进入交互模式')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f'分块大小(像素) (默认: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                       help=f'重叠像素数 (默认: {DEFAULT_OVERLAP})')

    return parser


def validate_paths(data_dir: str, output_dir: str):
    """验证路径有效性"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"数据路径不是目录: {data_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


def parse_regions(region_str: str) -> List[str]:
    """解析区域字符串"""
    if region_str.lower() == 'all':
        return REGIONS

    regions = [r.strip() for r in region_str.split(',')]
    for region in regions:
        if region not in REGIONS:
            raise ValueError(f"不支持的区域: {region}。支持的区域: {REGIONS}")

    return regions


def parse_seasons(season_str: str) -> List[str]:
    """解析季节字符串"""
    if season_str.lower() == 'all':
        return SEASONS

    seasons = [s.strip() for s in season_str.split(',')]
    for season in seasons:
        if season not in SEASONS:
            raise ValueError(f"不支持的季节: {season}。支持的季节: {SEASONS}")

    return seasons


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 判断是否进入交互模式
    if (not args.data_dir or not args.output_dir) or args.interactive:
        print("进入交互模式...")
        config = interactive_input()
        if config is None:
            return 0

        # 使用交互输入的配置
        data_dir = config['data_dir']
        output_dir = config['output_dir']
        regions = config['regions']
        year_pairs = config['year_pairs']
        seasons = config['seasons']
        large_grid_size_km = config['large_grid_size_km']
        small_grid_size_km = config['small_grid_size_km']
        threshold = config['threshold']
        exclude_zero = config['exclude_zero']
        log_level = config['log_level']
        use_optimized = config['use_optimized']
        chunk_size = config['chunk_size']
        overlap = config['overlap']
    else:
        # 使用命令行参数
        data_dir = args.data_dir
        output_dir = args.output_dir
        regions = parse_regions(args.regions)

        if args.year_pairs.lower() == 'all':
            year_pairs = YEAR_PAIRS
        else:
            year_pairs = parse_year_pairs(args.year_pairs)

        seasons = parse_seasons(args.seasons)
        large_grid_size_km = args.large_grid
        small_grid_size_km = args.small_grid
        threshold = args.threshold
        exclude_zero = args.exclude_zero
        log_level = args.log_level
        use_optimized = True  # 始终使用优化模式
        chunk_size = args.chunk_size
        overlap = args.overlap

    # 验证参数
    if large_grid_size_km <= small_grid_size_km:
        print(f"错误: 大网格大小({large_grid_size_km}km)必须大于小网格大小({small_grid_size_km}km)")
        return 1

    if chunk_size < 256:
        print(f"错误: 分块大小({chunk_size})不能小于256像素")
        return 1

    if overlap >= chunk_size // 2:
        print(f"错误: 重叠像素数({overlap})不能超过分块大小({chunk_size})的一半")
        return 1

    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                f"analysis_{Path(output_dir).name}.log",
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)

    try:
        # 验证路径
        validate_paths(data_dir, output_dir)

        mode_name = "优化模式(智能分块)"
        logger.info(f"开始分析 ({mode_name})...")
        logger.info(f"  数据目录: {data_dir}")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  区域: {regions}")
        logger.info(f"  年份对数量: {len(year_pairs)}")
        logger.info(f"  季节: {seasons}")
        logger.info(f"  大网格大小: {large_grid_size_km}km")
        logger.info(f"  小网格大小: {small_grid_size_km}km")
        logger.info(f"  分块大小: {chunk_size}像素")
        logger.info(f"  重叠像素: {overlap}像素")
        logger.info(f"  智能分块: 自适应启用")

        # 创建分析器
        analyzer = OptimizedVegetationClimateAnalyzer(data_dir, output_dir, chunk_size, overlap)

        # 运行分析
        analyzer.run_optimized_analysis(
            regions=regions,
            year_pairs=year_pairs,
            seasons=seasons,
            large_grid_size_km=large_grid_size_km,
            small_grid_size_km=small_grid_size_km,
            threshold=threshold,
            exclude_zero=exclude_zero
        )

        print(f"\n{mode_name}分析完成！结果保存在: {output_dir}")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"\n错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
