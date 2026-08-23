"""search 包：按配置组装搜索提供方。"""
from .. import config
from .base import SearchProvider, SearchResult


def build_provider(name: str | None = None) -> SearchProvider:
    name = name or config.SEARCH_PROVIDER
    if name == "local":
        from .local import LocalDocsProvider
        return LocalDocsProvider(config.LOCAL_DOCS_DIR)
    if name == "tavily":
        from .tavily import TavilyProvider
        return TavilyProvider(config.TAVILY_API_KEY)
    raise ValueError(f"未知搜索提供方: {name}")
