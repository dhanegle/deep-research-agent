"""搜索提供方接口与数据结构。"""
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider:
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError
