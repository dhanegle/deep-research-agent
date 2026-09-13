"""时效偏好离线评估：排序加入时效奖励后，候选质量如何变化。

用法（零 API 成本，复用 cache/search 快照）：

    .venv/Scripts/python.exe -m eval.freshness

背景：搜索结果此前不带日期，排序只有权威层级与相关度，没有时间维度。
本脚本对照 FRESH_DAYS=0（关闭时效＝改造前行为）与 FRESH_DAYS=180（默认），
输出 top-5 候选的时效分布变化，以及**权威条目是否被时效挤出**。

最后一项是关键回归指标：早期版本把时效排在相关性之前，导致"日期未知"的
知乎/华尔街日报挤掉相关性更高但发布较早的央视/新华网（22 份快照出现）。
最终方案改为"只奖新鲜、不罚旧"，权威净变化转正。
"""
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from agent import config
from agent.dates import extract_url_date, is_recent
from agent.researcher import Researcher
from agent.search.base import SearchResult
from agent.source_quality import authority_tier

TODAY = date.today()
CACHE = Path(__file__).resolve().parent.parent / "cache" / "search"


class _Trace:
    def log(self, *a, **k):
        pass


def _run(results, query, fresh_days):
    """在给定 FRESH_DAYS 下跑一次候选过滤（直接改 config，脚本退出即失效）。"""
    config.FRESH_DAYS = fresh_days
    r = Researcher.__new__(Researcher)
    r._question = query
    r.trace = _Trace()
    return r._filter_results(results, query)


def _tier(url):
    return authority_tier((urlparse(url).hostname or "").lower())


def _bucket(item):
    if not item.published_at:
        return "日期未知"
    return "近 180 天" if is_recent(item.published_at, 180, TODAY) else "更早"


def main() -> int:
    snapshots = []
    for f in sorted(CACHE.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list) and data:
            snapshots.append(data)
    if not snapshots:
        print(f"没有可用的搜索快照：{CACHE}（先跑一次真实检索生成缓存）")
        return 1

    all_urls = [r["url"] for d in snapshots for r in d
                if isinstance(r, dict) and r.get("url")]
    auth_urls = [u for u in all_urls if _tier(u) > 0]
    wd = sum(1 for u in all_urls if extract_url_date(u))
    ad = sum(1 for u in auth_urls if extract_url_date(u))
    print(f"快照 {len(snapshots)} 份")
    print(f"URL 日期可得性：全量 {wd}/{len(all_urls)} = {wd / max(len(all_urls), 1):.1%}"
          f"；权威域 {ad}/{len(auth_urls)} = {ad / max(len(auth_urls), 1):.1%}")

    off_c, on_c = Counter(), Counter()
    changed = lost = gained = 0
    for data in snapshots:
        items = [SearchResult(title=r.get("title") or "", url=r["url"],
                              snippet=r.get("snippet") or "",
                              published_at=extract_url_date(r["url"]))
                 for r in data if isinstance(r, dict) and r.get("url")]
        if not items:
            continue
        query = (items[0].title or "测试")[:12]
        off = _run(list(items), query, 0)
        on = _run(list(items), query, config._int("FRESH_DAYS", 180))
        if [x.url for x in off] != [x.url for x in on]:
            changed += 1
        for x in off:
            off_c[_bucket(x)] += 1
        for x in on:
            on_c[_bucket(x)] += 1
        t_off = sum(1 for x in off if _tier(x.url) > 0)
        t_on = sum(1 for x in on if _tier(x.url) > 0)
        lost += max(t_off - t_on, 0)
        gained += max(t_on - t_off, 0)

    print()
    print("top-5 候选的时效分布：")
    for k in ("近 180 天", "日期未知", "更早"):
        print(f"  {k:8s}  关闭时效 {off_c[k]:5d}  →  开启时效 {on_c[k]:5d}")
    print()
    print(f"top-5 序列发生变化的快照：{changed}/{len(snapshots)}")
    print(f"权威条目被时效挤出：{lost} 次；被时效补入：{gained} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
