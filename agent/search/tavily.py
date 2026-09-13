"""Tavily 搜索实现（免费额度 1000 次/月）。

更换博查/智谱等国内搜索只需实现 search.base.SearchProvider 同样的接口，
并在 search/__init__.py 的工厂里注册一行。
"""
import httpx

from .. import config
from ..dates import normalize
from .base import SearchProvider, SearchResult
from .cache import DiskCache


class TavilyProvider(SearchProvider):
    API = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("TavilyProvider 需要 TAVILY_API_KEY")
        self.api_key = api_key
        self.cache = DiskCache("search")

    def search(self, query: str, max_results: int = 5,
               include_domains: list[str] | None = None) -> list[SearchResult]:
        key = f"tavily:{config.TAVILY_SEARCH_DEPTH}:{query}:{max_results}"
        # topic 只在非默认值时才进键，否则会让历史快照缓存整体失效
        if config.TAVILY_TOPIC != "general":
            key += f"|t={config.TAVILY_TOPIC}"
        if include_domains:  # 域名白名单进入缓存键，避免与普通检索互相覆盖
            key += "|d=" + ",".join(sorted(include_domains))
        cached = self.cache.get(key)
        if cached is not None:
            return [SearchResult(**r) for r in cached]
        payload: dict = {"query": query, "max_results": max_results,
                         "search_depth": config.TAVILY_SEARCH_DEPTH,
                         "topic": config.TAVILY_TOPIC}
        if include_domains:
            payload["include_domains"] = list(include_domains)
        resp = httpx.post(
            self.API,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=20,
            trust_env=config.USE_SYSTEM_PROXY,
        )
        resp.raise_for_status()
        results = [
            SearchResult(
                title=str(r.get("title", ""))[:120],
                url=r["url"],
                snippet=str(r.get("content", ""))[:800],
                # published_date 只在 topic=news 下返回（实测 general 为 0/32，
                # news 为 32/32）；默认 topic 保持 general 以不损失政府站点召回，
                # 此时落到 SearchResult 的 URL 内嵌日期兜底（权威域覆盖≈70%）。
                published_at=normalize(str(r.get("published_date") or "")),
            )
            for r in resp.json().get("results", [])
        ]
        self.cache.put(key, [vars(r) for r in results])
        return results
