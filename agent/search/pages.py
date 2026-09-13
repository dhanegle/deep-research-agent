"""网页/文档正文提取。

优先级：本地文件（local 搜索源返回的路径）→ 磁盘缓存 → 真实抓取。
正文截断到 MAX_CHARS，控制后续摘要与上下文的体量。
"""
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx

try:  # trafilatura 依赖 lxml；环境装不上时自动退化为正则剥离
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

from .. import config
from ..dates import resolve_published
from ..evidence import clean_page_text, page_problem
from .cache import DiskCache
from .local import read_local_document

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
MAX_CHARS = 24000

_cache = DiskCache("pages")


@dataclass
class PageDocument:
    text: str
    published_at: str = ""
    publisher: str = ""
    retrieved_at: str = ""


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "nav", "footer", "aside"}:
            self.hidden.append(tag)
        if not self.hidden and tag in {"p", "div", "br", "tr", "article", "h1", "h2"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.hidden and self.hidden[-1] == tag:
            self.hidden.pop()
        if not self.hidden and tag in {"p", "div", "tr", "h1", "h2"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _strip_tags(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    return clean_page_text("".join(parser.parts))


def _checked(document: PageDocument) -> PageDocument:
    document.text = clean_page_text(document.text)[:MAX_CHARS]
    problem = page_problem(document.text)
    if problem:
        raise ValueError(f"页面无有效正文：{problem}")
    return document


def extract_document(html: str, url: str = "") -> PageDocument:
    extracted = trafilatura.extract(
        html, url=url, favor_precision=True, include_comments=False,
        include_tables=True, deduplicate=True,
    ) if trafilatura else None
    document = PageDocument(
        text=extracted or _strip_tags(html), publisher=urlparse(url).hostname or "",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    meta_date = ""
    if trafilatura:
        metadata = trafilatura.extract_metadata(html, default_url=url)
        if metadata:
            meta_date = metadata.date or ""
            document.publisher = metadata.sitename or document.publisher
    # meta 日期会被页脚版权年份污染（实测光明日报电子版 URL 是 /202606/30/，
    # 却解析出 2025-01-01），因此与 URL 内嵌日期交叉校验后再采信。
    document.published_at = resolve_published(meta_date, url)
    return _checked(document)


def fetch_document(url: str) -> PageDocument:
    if not url.startswith(("http://", "https://")):
        # local 搜索源：url 是相对 LOCAL_DOCS_DIR 的文档路径
        path = Path(url)
        if not path.is_absolute():
            path = config.LOCAL_DOCS_DIR / path
        if path.is_file():
            return _checked(PageDocument(
                read_local_document(path, max_chars=MAX_CHARS), publisher="本地资料",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            ))
    cached = _cache.get(f"page:{url}")
    if cached is not None:
        document = _checked(PageDocument(
            **{k: cached[k] for k in PageDocument.__dataclass_fields__ if k in cached}))
        # 历史缓存里可能存着被页脚版权污染的日期（实测光明日报电子版
        # 解析出 2025-01-01，实际 2026-06-30）。用 URL 重新校验一次——
        # resolve_published 以 URL 为准，对已正确的值幂等，无需重新请求。
        document.published_at = resolve_published(document.published_at, url)
        return document
    # 挂死站点最多等 10 秒；HTML 先截断再抽正文，trafilatura 对超大页面很慢
    resp = httpx.get(url, headers={"User-Agent": UA},
                     timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=True,
                     trust_env=config.USE_SYSTEM_PROXY)
    resp.raise_for_status()
    if len(resp.content) > 2_000_000:
        raise ValueError("页面过大，无法可靠提取正文")
    document = extract_document(resp.text, url)
    _cache.put(f"page:{url}", asdict(document))
    return document


def fetch_page(url: str) -> str:
    return fetch_document(url).text
