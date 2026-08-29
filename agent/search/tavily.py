"""Tavily 搜索实现（免费额度 1000 次/月）。

更换博查/智谱等国内搜索只需实现 search.base.SearchProvider 同样的接口，
并在 search/__init__.py 的工厂里注册一行。
"""
import httpx

from .. import config
from .base import SearchProvider, SearchResult
from .cache import DiskCache


class TavilyProvider(SearchProvider):
    API = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("TavilyProvider 需要 TAVILY_API_KEY")
        self.api_key = api_key
        self.cache = DiskCache("search")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        key = f"tavily:{config.TAVILY_SEARCH_DEPTH}:{query}:{max_results}"
        cached = self.cache.get(key)
        if cached is not None:
            return [SearchResult(**r) for r in cached]
        resp = httpx.post(
            self.API,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"query": query, "max_results": max_results,
                  "search_depth": config.TAVILY_SEARCH_DEPTH},
            timeout=20,
            trust_env=config.USE_SYSTEM_PROXY,
        )
        resp.raise_for_status()
        results = [
            SearchResult(
                title=str(r.get("title", ""))[:120],
                url=r["url"],
                snippet=str(r.get("content", ""))[:800],
            )
            for r in resp.json().get("results", [])
        ]
        self.cache.put(key, [vars(r) for r in results])
        return results
