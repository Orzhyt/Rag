"""
文档解析模块 - 解析 Word/Excel/PDF/Markdown 等文档，输出可入库 Milvus 的元数据数组。

架构设计:
    BaseParser (抽象基类)
    ├── PDFParser       - PyMuPDF
    ├── DocParser       - .doc 二进制启发式提取 (中文友好)
    ├── DocxParser      - python-docx
    ├── ExcelParser     - pandas + openpyxl
    ├── MarkdownParser  - 纯文本 + frontmatter
    ├── PlainTextParser - txt/json/csv/yaml/xml
    └── UnstructuredParser - .ppt/.pptx/.html/.rtf 等 (unstructured 兜底, Linux/Mac)

    ParserRegistry: 扩展名 → 解析器映射，支持运行时注册/注销
    DocumentParser: 编排器，组合 scanner 结果 + 解析 + 分块

扩展方式:
    1. 新增文件类型: 继承 BaseParser，实现 parse()，注册到 ParserRegistry
    2. 替换分块策略: 实现 chunk_text 签名相同的方法，传给 DocumentParser
    3. 替换解析器: 在 ParserRegistry 中覆盖特定扩展名的解析器
"""

import os
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from common import logger


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ParsedChunk:
    """解析后的文本块，可直接入库 Milvus"""
    chunk_id: str
    content: str
    source_file: str
    file_name: str
    file_type: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    title: Optional[str] = None
    file_size: int = 0
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_milvus_dict(self) -> Dict[str, Any]:
        """转为可直接插入 Milvus 的字典"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number,
            "sheet_name": self.sheet_name,
            "title": self.title,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "metadata": self.metadata,
        }


# ============================================================================
# 分块策略
# ============================================================================

class RecursiveTextSplitter:
    """递归文本分块器，按分隔符优先级递归切分"""

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", "；", ";", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str) -> List[str]:
        """将文本切分为块"""
        if not text or not text.strip():
            return []
        return self._split_text(text.strip(), self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        result: List[str] = []
        separator = separators[0]
        next_separators = separators[1:] if len(separators) > 1 else [""]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        current_chunk = ""
        for part in splits:
            if not part and separator:
                continue
            piece = part if not separator else part + separator

            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    result.append(current_chunk.strip())
                if len(piece) > self.chunk_size:
                    sub_splits = self._split_text(piece, next_separators)
                    if result and sub_splits:
                        overlap_text = result[-1][-self.chunk_overlap:]
                        sub_splits[0] = overlap_text + sub_splits[0]
                    result.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            result.append(current_chunk.strip())

        return result


def _default_chunk_strategy() -> RecursiveTextSplitter:
    return RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)


# ============================================================================
# 解析器基类
# ============================================================================

class BaseParser(ABC):
    """文档解析器抽象基类。
    子类只需实现 parse()，返回纯文本字符串。
    """

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """解析文件，返回提取的文本内容"""
        ...

    def parse_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """解析文件，返回 (文本内容, 附加元数据)。
        默认实现调用 parse()，子类可覆盖以提供更多元数据。
        """
        return self.parse(file_path), {}


# ============================================================================
# 具体解析器
# ============================================================================

class PDFParser(BaseParser):
    """PDF 解析器，基于 PyMuPDF (fitz)"""

    def parse(self, file_path: str) -> str:
        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF 未安装，请执行: pip install PyMuPDF")
            return ""

        try:
            doc = fitz.open(file_path)
            texts: List[str] = []
            for page in doc:
                text = page.get_text()
                if text:
                    texts.append(text)
            doc.close()
            return "\n\n".join(texts)
        except Exception as e:
            logger.error(f"PDF 解析失败 [{file_path}]: {e}")
            return ""

    def parse_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            import fitz
        except ImportError:
            return "", {}

        try:
            doc = fitz.open(file_path)
            texts: List[str] = []
            title = None

            if doc.metadata:
                title = doc.metadata.get("title")

            for page in doc:
                text = page.get_text()
                if text:
                    texts.append(text)

            metadata = {
                "page_count": doc.page_count,
                "title": title,
            }
            doc.close()
            return "\n\n".join(texts), metadata
        except Exception as e:
            logger.error(f"PDF 解析失败 [{file_path}]: {e}")
            return "", {}


class DocxParser(BaseParser):
    """DOCX 解析器，基于 python-docx"""

    def parse(self, file_path: str) -> str:
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx 未安装，请执行: pip install python-docx")
            return ""

        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"DOCX 解析失败 [{file_path}]: {e}")
            return ""

    def parse_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            from docx import Document
        except ImportError:
            return "", {}

        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            title = None
            if doc.paragraphs and doc.paragraphs[0].text.strip():
                title = doc.paragraphs[0].text.strip()

            metadata = {
                "paragraph_count": len(doc.paragraphs),
                "title": title,
            }
            return "\n".join(paragraphs), metadata
        except Exception as e:
            logger.error(f"DOCX 解析失败 [{file_path}]: {e}")
            return "", {}


class ExcelParser(BaseParser):
    """Excel 解析器，基于 pandas + openpyxl。
    每个 sheet 的内容会用 sheet 名称作为标题分隔。
    """

    def parse(self, file_path: str) -> str:
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas 未安装，请执行: pip install pandas openpyxl")
            return ""

        try:
            ext = Path(file_path).suffix.lower()
            engine = "openpyxl" if ext == ".xlsx" else "xlrd" if ext == ".xls" else None
            xls = pd.ExcelFile(file_path, engine=engine)
            parts: List[str] = []

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                parts.append(f"[Sheet: {sheet_name}]")
                parts.append(df.to_string(index=False))
                parts.append("")

            xls.close()
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Excel 解析失败 [{file_path}]: {e}")
            return ""

    def parse_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError:
            return "", {}

        try:
            ext = Path(file_path).suffix.lower()
            engine = "openpyxl" if ext == ".xlsx" else "xlrd" if ext == ".xls" else None
            xls = pd.ExcelFile(file_path, engine=engine)
            parts: List[str] = []
            sheet_metadata: List[str] = []

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue
                sheet_metadata.append(sheet_name)
                parts.append(f"[Sheet: {sheet_name}]")
                parts.append(df.to_string(index=False))
                parts.append("")

            xls.close()
            return "\n".join(parts), {"sheet_names": sheet_metadata}
        except Exception as e:
            logger.error(f"Excel 解析失败 [{file_path}]: {e}")
            return "", {}


class MarkdownParser(BaseParser):
    """Markdown 解析器。
    保留原始 markdown 格式，YAML frontmatter 会被提取为元数据。
    """

    def parse(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self._strip_frontmatter(content)
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Markdown 读取失败 [{file_path}]: {e}")
                return ""
        except Exception as e:
            logger.error(f"Markdown 解析失败 [{file_path}]: {e}")
            return ""

    def _strip_frontmatter(self, content: str) -> str:
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content.strip()

    def parse_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
            except Exception:
                return "", {}
        except Exception:
            return "", {}

        frontmatter: Dict[str, Any] = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = self._parse_frontmatter(parts[1])
                body = parts[2].strip()

        title = frontmatter.get("title")
        return body, {"title": title, "frontmatter": frontmatter}

    def _parse_frontmatter(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for line in text.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip().strip("\"'")
        return result


class PlainTextParser(BaseParser):
    """纯文本解析器，用于 txt/json/csv/yaml/xml 等格式"""

    def parse(self, file_path: str) -> str:
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"文本读取失败 [{file_path}]: {e}")
                return ""
        logger.error(f"无法解码文件 [{file_path}]")
        return ""


class DocParser(BaseParser):
    """.doc (OLE2) 解析器，通过二进制启发式提取文本。
    适用于中文 .doc 文件。对于西文 .doc 文件，建议使用 UnstructuredParser。
    """

    def parse(self, file_path: str) -> str:
        return self._binary_extract(file_path)

    def _binary_extract(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except Exception as e:
            logger.error(f"读取 .doc 文件失败 [{file_path}]: {e}")
            return ""

        try:
            text = data.decode("utf-16-le", errors="ignore")
        except Exception:
            return ""

        # 提取可读文本段
        _cjk = (
            "一-鿿"      # CJK统一表意文字
            "　-〿"      # CJK标点符号
            "＀-￯"      # 全角形式
            " -⁯"      # 通用标点
            "a-zA-Z0-9"
            "“”"       # 弯双引号
            "‘’"       # 弯单引号
            "「」"       # 「」
            "『』"       # 『』
            "【】"       # 【】
            "《》"       # 《》
            "（）"       # （）
            "…—"       # …—
            "·%"            # ·%
            r",\.!\?;:'\(\)\[\]\{\}<>/\|@#$%^&\*\+=\-_~`"
            " -~"
        )
        pattern = re.compile("[" + _cjk + "]+")
        matches = pattern.findall(text)

        # 过滤噪声: 去除过短片段及 Word 域代码
        word_field_pattern = re.compile(
            r"^(TOC|HYPERLINK|PAGEREF|PAGE|MERGEFORMAT|FORMTEXT|SEQ|STYLEREF)"
        )
        meaningful = []
        for m in matches:
            stripped = m.strip()
            if len(stripped) < 10:
                continue
            if word_field_pattern.search(stripped):
                continue
            # 去除单字符重复组成的噪声
            if len(set(stripped)) < 3:
                continue
            meaningful.append(stripped)

        return "\n".join(meaningful)

    def parse_with_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        content = self.parse(file_path)
        title = None
        if content:
            first_line = content.split("\n")[0].strip()
            if len(first_line) < 100:
                title = first_line
        return content, {"title": title}


class UnstructuredParser(BaseParser):
    """兜底解析器，基于 unstructured 库。
    用于 .ppt / .pptx / .html / .rtf 等格式。

    注意: unstructured 在 Windows 上可能因原生依赖导致崩溃 (segfault)。
    在 Windows 上会自动降级为空结果，不会导致进程崩溃。
    """

    def parse(self, file_path: str) -> str:
        import platform

        if platform.system() == "Windows":
            logger.warning(
                f"unstructured 在 Windows 上可能不稳定，跳过解析 [{file_path}]。"
                "建议在 Linux/Mac 环境下使用，或手动安装 LibreOffice 进行格式转换。"
            )
            return ""

        try:
            from unstructured.partition.auto import partition
        except ImportError:
            logger.error("unstructured 未安装，请执行: pip install unstructured")
            return ""

        try:
            elements = partition(filename=file_path)
            return "\n".join(str(el) for el in elements if str(el).strip())
        except Exception as e:
            logger.error(f"unstructured 解析失败 [{file_path}]: {e}")
            return ""


# ============================================================================
# 解析器注册表
# ============================================================================

class ParserRegistry:
    """解析器注册表，管理扩展名 → 解析器的映射。

    扩展方式:
        registry = ParserRegistry()
        registry.register({".custom"}, CustomParser())
        registry.deregister(".custom")
    """

    def __init__(self):
        self._parsers: Dict[str, BaseParser] = {}
        self._fallback: Optional[BaseParser] = None

    def register(self, extensions: set, parser: BaseParser) -> None:
        """为指定扩展名注册解析器"""
        for ext in extensions:
            normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            self._parsers[normalized] = parser
            logger.debug(f"注册解析器: {normalized} → {type(parser).__name__}")

    def deregister(self, extension: str) -> None:
        """注销指定扩展名的解析器"""
        normalized = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self._parsers.pop(normalized, None)
        logger.debug(f"注销解析器: {normalized}")

    def set_fallback(self, parser: BaseParser) -> None:
        """设置兜底解析器（当没有匹配的扩展名解析器时使用）"""
        self._fallback = parser

    def get(self, extension: str) -> Optional[BaseParser]:
        """获取扩展名对应的解析器"""
        normalized = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        parser = self._parsers.get(normalized)
        if parser:
            return parser
        if self._fallback:
            return self._fallback
        return None

    def list_registered(self) -> Dict[str, str]:
        """列出所有已注册的扩展名 → 解析器名映射"""
        return {ext: type(p).__name__ for ext, p in sorted(self._parsers.items())}


def create_default_registry() -> ParserRegistry:
    """创建包含默认解析器的注册表"""
    registry = ParserRegistry()

    registry.register({".pdf"}, PDFParser())
    registry.register({".docx"}, DocxParser())
    registry.register({".xls", ".xlsx"}, ExcelParser())
    registry.register({".md"}, MarkdownParser())
    registry.register({".txt", ".json", ".csv", ".yaml", ".yml", ".xml"}, PlainTextParser())

    registry.register({".doc"}, DocParser())

    # .ppt / .pptx / .html / .rtf 等用 unstructured 兜底
    # 注意: Windows 上 unstructured 会自动降级，如需解析这些格式请使用 Linux/Mac
    fallback_formats = {".ppt", ".pptx", ".html", ".htm", ".rtf"}
    registry.register(fallback_formats, UnstructuredParser())

    return registry


# ============================================================================
# 文档解析编排器
# ============================================================================

class DocumentParser:
    """文档解析编排器。

    用法:
        from data_pipeline.scanner import FileScanner
        from data_pipeline.parser import DocumentParser

        scanner = FileScanner()
        files = scanner.scan_directory("./data")
        parser = DocumentParser()
        chunks = parser.parse_files(files)

        for chunk in chunks:
            print(chunk.to_milvus_dict())
    """

    def __init__(
        self,
        registry: Optional[ParserRegistry] = None,
        chunk_strategy: Optional[RecursiveTextSplitter] = None,
    ):
        self.registry = registry or create_default_registry()
        self.chunk_strategy = chunk_strategy or _default_chunk_strategy()

    def parse_files(self, file_paths: List[str]) -> List[ParsedChunk]:
        """批量解析文件，返回所有 ParsedChunk 的列表"""
        all_chunks: List[ParsedChunk] = []
        for file_path in file_paths:
            chunks = self.parse_file(file_path)
            all_chunks.extend(chunks)
        logger.info(f"文档解析完成: {len(file_paths)} 个文件 → {len(all_chunks)} 个文本块")
        return all_chunks

    def parse_file(self, file_path: str) -> List[ParsedChunk]:
        """解析单个文件，返回其 ParsedChunk 列表"""
        ext = Path(file_path).suffix.lower()
        parser = self.registry.get(ext)

        if parser is None:
            logger.warning(f"未找到支持解析器: {ext} [{file_path}]，跳过")
            return []

        logger.debug(f"解析文件 [{ext}]: {file_path}")

        try:
            content, extra_metadata = parser.parse_with_metadata(file_path)
        except Exception as e:
            logger.error(f"解析异常 [{file_path}]: {e}")
            return []

        if not content or not content.strip():
            logger.warning(f"文件内容为空: {file_path}")
            return []

        text_chunks = self.chunk_strategy.split(content)
        if not text_chunks:
            return []

        file_stat = self._get_file_stat(file_path)
        file_name = Path(file_path).name
        total_chunks = len(text_chunks)

        parsed_chunks: List[ParsedChunk] = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(file_path, i, chunk_text)
            chunk = ParsedChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                source_file=file_path,
                file_name=file_name,
                file_type=ext.lstrip("."),
                chunk_index=i,
                total_chunks=total_chunks,
                title=extra_metadata.get("title"),
                file_size=file_stat.get("size", 0),
                created_at=file_stat.get("created_at"),
                modified_at=file_stat.get("modified_at"),
                metadata=extra_metadata,
            )
            parsed_chunks.append(chunk)

        return parsed_chunks

    def iter_parse_files(self, file_paths: List[str]) -> Iterator[ParsedChunk]:
        """迭代器版本：逐个产出 ParsedChunk，适合大文件集"""
        for file_path in file_paths:
            chunks = self.parse_file(file_path)
            yield from chunks

    def _generate_chunk_id(self, file_path: str, index: int, content: str) -> str:
        raw = f"{file_path}:{index}:{content[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _get_file_stat(self, file_path: str) -> Dict[str, Any]:
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except OSError:
            return {}

    # --- 扩展点 ---

    def set_chunk_strategy(self, strategy: RecursiveTextSplitter) -> None:
        """替换分块策略"""
        self.chunk_strategy = strategy

    def register_parser(self, extensions: set, parser: BaseParser) -> None:
        """注册新的解析器"""
        self.registry.register(extensions, parser)


# ============================================================================
# 便捷函数
# ============================================================================

def parse_files(
    file_paths: List[str],
    registry: Optional[ParserRegistry] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[ParsedChunk]:
    """便捷函数：解析文件列表，返回 ParsedChunk 列表"""
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    parser = DocumentParser(registry=registry, chunk_strategy=splitter)
    return parser.parse_files(file_paths)


def parse_directory(
    root_path: str,
    registry: Optional[ParserRegistry] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[ParsedChunk]:
    """便捷函数：扫描并解析目录下所有文件"""
    from data_pipeline.scanner import FileScanner

    scanner = FileScanner()
    files = scanner.scan_directory(root_path)
    return parse_files(files, registry=registry, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
