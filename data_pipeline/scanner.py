import os
from pathlib import Path
from typing import List, Optional, Set, Iterator
from common import logger


class FileScanner:
    DEFAULT_SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".json", ".csv",
        ".pdf", ".doc", ".docx", ".rtf",
        ".xls", ".xlsx", ".ppt", ".pptx",
        ".html", ".xml", ".yaml", ".yml"
    }

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
        excluded_dirs: Optional[Set[str]] = None,
        excluded_files: Optional[Set[str]] = None
    ):
        self.supported_extensions = supported_extensions or self.DEFAULT_SUPPORTED_EXTENSIONS
        self.excluded_dirs = excluded_dirs or {".git", "__pycache__", "node_modules", ".idea", ".pytest_cache"}
        self.excluded_files = excluded_files or {"Thumbs.db", ".DS_Store"}

    def is_supported_file(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in self.supported_extensions

    def is_excluded_dir(self, dirname: str) -> bool:
        return dirname in self.excluded_dirs

    def is_excluded_file(self, filename: str) -> bool:
        return filename in self.excluded_files

    def scan_directory(self, root_path: str) -> List[str]:
        """扫描指定目录，返回所有符合条件的文件路径列表"""
        if not os.path.exists(root_path):
            logger.error(f"路径不存在: {root_path}")
            return []

        if not os.path.isdir(root_path):
            logger.error(f"不是有效目录: {root_path}")
            return []

        file_paths: List[str] = []
        
        try:
            for dirpath, dirnames, filenames in os.walk(root_path):
                dirnames[:] = [d for d in dirnames if not self.is_excluded_dir(d)]
                
                for filename in filenames:
                    if self.is_excluded_file(filename):
                        continue
                    
                    if self.is_supported_file(filename):
                        full_path = os.path.join(dirpath, filename)
                        file_paths.append(full_path)
                        logger.debug(f"发现文件: {full_path}")

            logger.info(f"扫描完成，共发现 {len(file_paths)} 个文件")
        except PermissionError:
            logger.error(f"无权访问目录: {root_path}")
        except Exception as e:
            logger.error(f"扫描目录时发生错误: {str(e)}")

        return file_paths

    def scan_directories(self, root_paths: List[str]) -> List[str]:
        """扫描多个目录，返回所有符合条件的文件路径列表"""
        all_files: List[str] = []
        for root_path in root_paths:
            files = self.scan_directory(root_path)
            all_files.extend(files)
        return all_files

    def get_file_stats(self, file_paths: List[str]) -> dict:
        """获取文件统计信息"""
        stats = {
            "total_files": len(file_paths),
            "by_extension": {},
            "total_size": 0
        }

        for filepath in file_paths:
            ext = Path(filepath).suffix.lower()
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
            
            try:
                stats["total_size"] += os.path.getsize(filepath)
            except OSError:
                pass

        return stats

    def iter_files(self, root_path: str) -> Iterator[str]:
        """返回文件路径迭代器"""
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if not self.is_excluded_dir(d)]
            
            for filename in filenames:
                if self.is_excluded_file(filename):
                    continue
                
                if self.is_supported_file(filename):
                    yield os.path.join(dirpath, filename)


def scan_files(
    root_path: str,
    supported_extensions: Optional[Set[str]] = None
) -> List[str]:
    """便捷函数：扫描目录获取文件列表"""
    scanner = FileScanner(supported_extensions=supported_extensions)
    return scanner.scan_directory(root_path)
