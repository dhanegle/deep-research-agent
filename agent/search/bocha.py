"""博查搜索实现（国内 AI 搜索 API，中文召回优于 Tavily）。

与 TavilyProvider 实现同一个 SearchProvider 接口，仅请求/响应解析不同。
返回结果同样落盘缓存，评估回放零消耗。
"""
import httpx

from .. import config
from .base import SearchProvider, SearchResult
from .cache import DiskCache


class BochaProvider(SearchProvider):
    API = "https://api.bochaai.com/v1/web-search"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("BochaProvider 需要 BOCHA_API_KEY")
        self.api_key = api_key
        self.cache = DiskCache("search")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        key = f"bocha:v2:{query}:{max_results}"
        cached = self.cache.get(key)
        if cached is not None:
            return [SearchResult(**r) for r in cached]
        resp = httpx.post(
            self.API,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "count": max_results,
                "summary": True,
            },
            timeout=20,
            trust_env=config.USE_SYSTEM_PROXY,
        )
        resp.raise_for_status()
        data = resp.json()
        # 博查返回结构：顶层 code/log_id/data，搜索结果在 data.webPages.value[] 中
        web = (data.get("data") or {}).get("webPages") or {}
        raw = web.get("value", [])
        results = []
        for r in raw:
            if not r.get("url"):
                continue
            summary = str(r.get("summary") or "")
            snip = str(r.get("snippet") or "")
            text = summary if summary else snip
            if snip and snip not in text:
                text = f"{text} {snip}".strip()
            results.append(SearchResult(
                title=str(r.get("name", ""))[:120],
                url=r["url"],
                snippet=text[:800],
            ))
        self.cache.put(key, [vars(r) for r in results])
        return results
