import sys
from common.logger import logger
from data_pipeline.scanner import FileScanner


def scanner(root_path: str):
    logger.info(f"开始扫描目录: {root_path}")
    
    scanner = FileScanner()
    files = scanner.scan_directory(root_path)
    
    if not files:
        logger.warning("未找到符合要求的文件")
        return
    
    logger.info(f"扫描完成，共发现 {len(files)} 个文件")
    
    print("\n" + "=" * 80)
    print("发现的文件列表:")
    print("=" * 80)
    
    for i, filepath in enumerate(files, 1):
        print(f"{i:3d}. {filepath}")
    
    print("=" * 80)
    
    stats = scanner.get_file_stats(files)
    print(f"\n文件统计信息:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  总大小: {stats['total_size'] / (1024 * 1024):.2f} MB")
    print(f"  按扩展名分布:")
    for ext, count in sorted(stats['by_extension'].items()):
        print(f"    {ext}: {count} 个")


if __name__ == "__main__":
    scanner("D://Codes//Rag_Project//data//Smart-Emergency-Response//baseline//datas")
    ...
