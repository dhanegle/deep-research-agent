"""本地文件夹检索：把指定目录下的文档当作"搜索语料"。

- search(query)：按关键词（空格分词）给文档打分，返回文件路径作为链接；
- read_page(路径)：由 pages.fetch_page 调用 read_local_document 读取正文。
- 支持 md / txt / pdf / docx；txt/md 自动兼容 UTF-8 与 GBK 编码。
- 每次搜索重新扫描目录，运行中新增/修改的文档即刻可见。

用途：资料不在公网而在本地（内部文档、笔记、论文）时的离线调研。
"""
from pathlib import Path

from .. import config
from .base import SearchProvider, SearchResult

SUPPORTED_EXTS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def read_local_document(path: str | Path, max_chars: int = 6000) -> str:
    """读取本地文档正文，统一返回纯文本并截断。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # Windows 中文 txt 常见 GBK 编码
            text = path.read_bytes().decode("gbk", errors="ignore")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "（未安装 pypdf，无法读取 PDF：pip install pypdf）"
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    elif suffix == ".docx":
        try:
            import docx
        except ImportError:
            return "（未安装 python-docx，无法读取 docx：pip install python-docx）"
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    else:
        return f"（不支持的文件类型：{suffix}）"
    return text[:max_chars]


class LocalDocsProvider(SearchProvider):
    """文件夹即语料库：文件名与正文的关键词命中数排序。"""

    def __init__(self, docs_dir: str | Path | None = None):
        self.docs_dir = Path(docs_dir or config.LOCAL_DOCS_DIR)
        if not self.docs_dir.is_dir():
            raise RuntimeError(
                f"本地文档目录不存在：{self.docs_dir}\n"
                "请在 .env 中设置 LOCAL_DOCS_DIR 指向你的文档文件夹，或改用 tavily 搜索源。"
            )

    def _score(self, path: Path, terms: list[str]) -> int:
        name = path.stem
        try:
            head = read_local_document(path, max_chars=3000)
        except Exception:
            head = ""
        score = 0
        for term in terms:
            if term in name:
                score += 5          # 文件名命中的权重更高
            if term in head:
                score += 2
        return score

    def _snippet(self, text: str, terms: list[str], width: int = 120) -> str:
        for term in terms:
            pos = text.find(term)
            if pos != -1:
                start = max(0, pos - width // 2)
                return text[start:start + width].replace("\n", " ")
        return text[:width].replace("\n", " ")

    @staticmethod
    def _expand_terms(terms: list[str]) -> list[str]:
        """超过 6 字的长词（中文自然语句）按整串匹配几乎必落空，
        退化为滑动 4 元组：相关文档因共享片段多而排在前面。"""
        out = []
        for t in terms:
            if len(t) > 6:
                out.extend(t[i:i + 4] for i in range(0, len(t) - 3, 2))
            else:
                out.append(t)
        return out

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        terms = [t for t in query.split() if len(t) >= 2]
        if not terms:
            terms = [query.strip()]
        terms = self._expand_terms(terms)
        files = [p for p in self.docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()]
        scored = []
        for path in files:
            score = self._score(path, terms)
            if score > 0:
                scored.append((score, path))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        results = []
        for score, path in scored[:max_results]:
            text = read_local_document(path, max_chars=4000)
            # 返回相对 docs_dir 的短路径：模型回传长绝对路径（含空格/中文）易抄错
            rel = path.relative_to(self.docs_dir).as_posix()
            results.append(SearchResult(
                title=path.stem,
                url=rel,
                snippet=self._snippet(text, terms),
            ))
        return results
