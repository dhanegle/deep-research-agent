"""网页/文档正文提取。

优先级：本地文件（local 搜索源返回的路径）→ 磁盘缓存 → 真实抓取。
正文截断到 MAX_CHARS，控制后续摘要与上下文的体量。
"""
import re
from pathlib import Path

import httpx

try:  # trafilatura 依赖 lxml；环境装不上时自动退化为正则剥离
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

from .. import config
from .cache import DiskCache
from .local import read_local_document

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
MAX_CHARS = 6000

_cache = DiskCache("pages")


def _strip_tags(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        # local 搜索源：url 是相对 LOCAL_DOCS_DIR 的文档路径
        path = Path(url)
        if not path.is_absolute():
            path = config.LOCAL_DOCS_DIR / path
        if path.is_file():
            return read_local_document(path, max_chars=MAX_CHARS)
    cached = _cache.get(f"page:{url}")
    if cached is not None:
        return cached["text"]
    # 挂死站点最多等 10 秒；HTML 先截断再抽正文，trafilatura 对超大页面很慢
    resp = httpx.get(url, headers={"User-Agent": UA},
                     timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=True,
                     trust_env=config.USE_SYSTEM_PROXY)
    resp.raise_for_status()
    html = resp.text[:60000]
    extracted = trafilatura.extract(html) if trafilatura else None
    text = (extracted or _strip_tags(html))[:MAX_CHARS]
    _cache.put(f"page:{url}", {"text": text})
    return text
