"""搜索提供方接口与数据结构。"""
from dataclasses import dataclass

from ..dates import extract_url_date


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    # 发布日期（YYYY-MM-DD），取不到为空串。此前没有这个字段，导致排序与选材
    # 完全没有时间维度——2025-11 的旧稿与 2026-06 的新稿同权。见 agent/dates.py。
    published_at: str = ""

    def __post_init__(self):
        # 提供方没给日期时退回 URL 内嵌日期。放在这里而不是各 provider 里，
        # 是为了让**历史缓存**也受益——旧快照没有 published_at 字段，
        # 重建时同样能补上（URL 提取是确定性的，无需重新请求）。
        if not self.published_at:
            self.published_at = extract_url_date(self.url)


class SearchProvider:
    def search(self, query: str, max_results: int = 5,
               include_domains: list[str] | None = None) -> list[SearchResult]:
        """include_domains：只在该域名白名单内检索（权威定向补搜用）。

        Tavily 映射到 include_domains、博查映射到 include；不支持的提供方
        可以忽略该参数（调用方按 TypeError 兜底）。注意博查不支持泛域
        "gov.cn"，白名单须逐个枚举具体站点，见 agent/source_quality.py。
        """
        raise NotImplementedError
