"""阶段 B：信息收集 ReAct 循环（项目的核心）。

针对 Qwen2.5-3B 的可靠性设计——三条防线：
1. 非法工具调用（未知工具/缺参数/编造链接）→ 错误信息以 tool 消息回喂，模型自修复；
2. 连续 3 次非法 → 该轮降级为 JSON-ReAct（提示词版动作选择，不依赖原生 tool call），
   两条路径的对比数据计入 stats，是评估实验之一；
3. 模型驱动整轮颗粒无收 → scripted_collect 规则兜底：按查询词直接搜索并阅读靠前结果。
其它约束：只暴露 2 个单参数工具；read_page 的 url 必须来自 web_search 结果（防幻觉链接）；
单轮最多 MAX_STEPS 步、最多收录 8 条来源。
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .search.base import SearchProvider
from .search.pages import fetch_document

from . import config
from .dates import is_recent
from .evidence import content_terms, evidence_blocks, select_evidence
from .knowledge import (
    KnowledgeBase, _bigrams, ascii_proper_tokens, ascii_token_hit,
    query_tokens, token_coverage,
)

from .llm import LLM, tool_call_args, tool_call_name
from .source_quality import (
    authority_domains, authority_tier, is_blocked, is_internal_source, is_low_quality,
)
from .stats import RunStats
from .trace import Trace

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索资料库，返回文档标题、链接和摘要的编号列表。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词，5~15 字，单一主题"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "读取指定文档正文，收录为报告资料。path 必须原样使用 web_search 返回的路径，不要添加任何网址前缀。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文档路径，原样复制 web_search 结果中的路径"}},
                "required": ["path"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
你是一个资料调研助手，任务是围绕给定调研问题收集资料。规则：
1. 用 web_search 搜索时，搜索词必须短而具体（5-15 字、单一主题），不要把多个主题拼在一个搜索词里；不同角度分开多次搜索。\
查统计总量、行业数据时，用官方口径词加范围限定词（如「全国普通高校毕业生人数」「中国新能源汽车出口量 海关总署」「全国居民人均消费支出 统计局」），\
不要用口语缩略（如「毕业总人数」「出口多少」）——口语词只会匹配到单个学校/公司的通知页面，官方数据另有固定说法。
2. 从搜索返回的编号列表中挑选与调研问题最相关的 2-3 个文档，用 read_page 阅读。
3. read_page 的 path 必须原样复制搜索结果中的路径，不能编造、修改或添加网址前缀。
4. 大纲的各个方面都要收集资料，不要只搜一个角度。
5. 资料足够覆盖问题后，停止调用工具，直接回复：资料收集完成
"""

# 消融实验的朴素基线：不含任何行为规则约束
NAIVE_SYSTEM_PROMPT = "你是资料调研助手，请使用可用工具收集资料来回答调研问题。"

NUDGE_TMPL = (
    "资料不足：当前仅收录 {n} 条（至少需要 {min} 条）。"
    "请立即用 read_page 阅读下列尚未阅读的路径（不要再搜索）：\n{urls}"
)

REACT_FALLBACK_PROMPT = """\
请以 JSON 决定下一步动作，只输出 JSON，三选一：
{"action": "web_search", "query": "搜索关键词"}
{"action": "read_page", "path": "之前搜索结果中的文档路径"}
{"action": "finish"}
path 必须原样来自之前的搜索结果，资料足够时选 finish。
"""

# 「调研一下2026年的毕业情况」→「2026年的毕业情况」：剥掉请求动词留核心
_QUESTION_PREFIX = re.compile(
    r"^(请|帮我|我想|给我|麻烦)?(深度|全面|快速)?"
    r"(调研|研究|分析|了解|查一下|查查|看看|搜集|调查|统计)(一下|一番)?"
)


def _strip_question_verbs(question: str) -> str:
    return _QUESTION_PREFIX.sub("", question).strip() or question


def _date_tag(published: str) -> str:
    """候选列表里标注发布日期，让模型能判断新旧。

    此前候选只有标题/路径/摘要，模型看不到任何时间信号——2025-11 的旧稿与
    2026-06 的新稿在它眼里完全一样，于是照排序读到哪条用哪条。日期未知时
    不标注（宁可缺失也不编造）。
    """
    return f"（{published}）" if published else ""


# 泛词二元组：出现在查询与核心的交集里不代表有主题锚定（「情况」「分析」任何题都带）
_GENERIC_BIGRAMS = {
    "情况", "分析", "统计", "特点", "分布", "趋势", "影响", "因素",
    "政策", "数据", "报告", "现状", "概述", "概况", "相关", "研究", "介绍",
}


class Act(BaseModel):
    action: Literal["web_search", "read_page", "finish"]
    query: str = Field(default="", max_length=60)
    path: str = Field(default="")


class Researcher:
    def __init__(
        self,
        llm: LLM,
        provider: SearchProvider,
        kb: KnowledgeBase,
        stats: RunStats,
        trace: Trace,
        on_step: Callable[[str, dict], None] | None = None,
    ):
        self.llm = llm
        self.provider = provider
        self.kb = kb
        self.stats = stats
        self.trace = trace
        self.emit = on_step or (lambda event, data: None)
        self._searched: set[str] = set()  # 本轮已执行过的搜索词（去重防空转）
        self._failed_urls: set[str] = set()  # 抓取失败/被拒收的 URL，跨轮不再重试
        self._lock = threading.Lock()     # 并行执行工具时保护共享状态
        self._core = ""                   # 问题核心词（剥掉请求动词），大纲空词改写用
        self._authority_passes = 0        # 已执行的权威定向检索次数（受预算约束）

    # ---- 主循环 ----------------------------------------------------------
    def run(self, question: str, outline: list[str], queries: list[str],
            max_steps: int | None = None, min_sources: int | None = None) -> None:
        max_steps = max_steps or config.MAX_STEPS
        if config.NAIVE:
            min_sources = 1        # 朴素基线：模型说完成就完成
        min_sources = min_sources if min_sources is not None else 3
        self._question = question  # 搜索结果相关性过滤要用
        self._core = _strip_question_verbs(question)[:14]  # 拼接改写防查询过长
        query_text = "、".join(queries)
        collected = ""
        if self.kb.sources:  # 补搜轮：告知已收录资料，避免重复阅读
            collected = (
                "\n已收录资料（不要重复阅读，只补充缺口）：\n"
                + "\n".join(f"[{s.id}] {s.title}" for s in self.kb.sources)
            )
        system = NAIVE_SYSTEM_PROMPT if config.NAIVE else SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"调研问题：{question}\n报告大纲：{'；'.join(outline)}\n"
                f"建议的搜索词（可自行补充）：{query_text}{collected}"
            )},
        ]
        seen_urls: dict[str, str] = {}
        consecutive_invalid = 0
        nudged = 0

        for step in range(max_steps):
            if len(self.kb.sources) >= config.MAX_SOURCES:
                break
            if config.REACT_MODE == "prompt":
                # 消融路径：全程提示词 JSON-ReAct，不使用原生 tool calling
                if self._json_react_step(messages, seen_urls):
                    break
                continue
            if consecutive_invalid >= 3:
                # 防线 2：原生 tool call 连续失败，降级为 JSON-ReAct
                self.stats.react_fallbacks += 1
                self.trace.log("react_fallback", step=step)
                done = self._json_react_step(messages, seen_urls)
                consecutive_invalid = 0
                if done:
                    break
                continue

            msg = self.llm.chat(messages, tools=TOOLS)
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                content = (msg.get("content") or "").strip()
                if config.NAIVE:
                    messages.append(msg)
                    break
                # 3B 常见行为：搜而不读、或读 1 条就宣布完成。
                # 约束：资料不足 min_sources 条前不接受自然结束；
                # nudge 直接给出未读链接清单，把"接下来做什么"降级为受限选择。
                enough = len(self.kb.sources) >= min_sources
                if ("完成" in content and enough) or enough or nudged >= 2:
                    messages.append(msg)
                    break
                unread = [u for u in seen_urls if u not in self.kb._by_url]
                urls = "\n".join(f"- {u}" for u in unread[:4]) or "（无未读链接，请先 web_search）"
                nudged += 1
                messages.append(msg)
                messages.append({"role": "user", "content": NUDGE_TMPL.format(
                    n=len(self.kb.sources), min=min_sources, urls=urls)})
                continue

            messages.append(msg)
            calls = [(tool_call_name(tc), tool_call_args(tc)) for tc in tool_calls]
            if len(calls) > 1:
                # 并行只用于 read_page（网页抓取是 IO 密集，串行是耗时大头）。
                # web_search 必须串行：实测 5 路并发触发 Tavily 限流（403/SSL 断连）。
                # 同一批 read_page 的路径只能来自更早回合的搜索结果，
                # 所以先提交阅读任务、主线程串行跑搜索是安全的。
                outcomes: list[tuple[bool, str] | None] = [None] * len(calls)
                with ThreadPoolExecutor(max_workers=3) as ex:
                    futs = {i: ex.submit(self._dispatch_safe, name, args, seen_urls)
                            for i, (name, args) in enumerate(calls) if name == "read_page"}
                    for i, (name, args) in enumerate(calls):
                        if i not in futs:
                            outcomes[i] = self._dispatch_safe(name, args, seen_urls)
                    for i, fut in futs.items():
                        outcomes[i] = fut.result()
            else:
                outcomes = [self._dispatch_safe(calls[0][0], calls[0][1], seen_urls)]
            for (name, _), (valid, result) in zip(calls, outcomes):
                self.stats.tool_calls += 1
                if not valid:
                    self.stats.tool_invalid += 1
                    consecutive_invalid += 1
                else:
                    consecutive_invalid = 0
                name = name or "unknown"
                messages.append({"role": "tool", "tool_name": name, "name": name, "content": result})

        self.trace.log(
            "research_round_end", sources=len(self.kb.sources), steps=max_steps,
        )

    # ---- JSON-ReAct 降级步 ------------------------------------------------
    def _json_react_step(self, messages: list[dict], seen_urls: dict[str, str]) -> bool:
        """返回 True 表示模型选择 finish。结果以 user 消息回传，绕开 tool 消息模板。"""
        try:
            act = self.llm.chat_json(
                messages + [{"role": "user", "content": REACT_FALLBACK_PROMPT}], Act, tag="react_fallback"
            )
        except ValueError:
            return True  # JSON 模式也失败，结束本轮交给上层兜底
        if act.action == "finish":
            messages.append({"role": "assistant", "content": act.model_dump_json()})
            return True
        args = {"query": act.query} if act.action == "web_search" else {"path": act.path}
        valid, result = self._dispatch(act.action, args, seen_urls)
        self.stats.tool_calls += 1
        if not valid:
            self.stats.tool_invalid += 1
            result = "错误：" + result
        messages.append({"role": "assistant", "content": act.model_dump_json()})
        messages.append({"role": "user", "content": "动作执行结果：" + result})
        return False

    def _anchor_query(self, query: str) -> str:
        """大纲空词兜底改写。

        3B 常把小节标题（「政策影响因素」「地域分布特点」）原样当搜索词；
        题库/答疑站的题面恰好字面包含这些词，任何搜索引擎都会把它们排在
        权威媒体前面——权威报道不会用这种标题。查询与问题核心词没有共享
        实义二元组时，把问题核心拼在前面再搜，补上主题限定。
        「情况/分析/分布」等泛词不算锚定——「专业分布情况」只与问题共享
        泛词「情况」，仍是需要改写的大纲空词。
        """
        core = self._core
        if not core or core in query or query in core:
            return query
        shared = content_terms(query) & content_terms(core)
        if shared:
            return query
        return f"{core} {query}"

    def _filter_results(self, results: list, query: str) -> list:
        """搜索结果相关性过滤与权威重排。

        实测问题：查专有名词（如 "linuxsb"）时搜索引擎常返回泛主题页面
        （如"什么是Linux"），3B 模型照单全收导致报告跑题。
        规则：
        - 查询含 ASCII 专有名词 → 必须命中该词（linux.sb 等分隔符已归一）；
          命中后不再用 0.35 覆盖率否决——否则 "linuxsb 主要 讨论 领域"
          会把真正的 linux.sb 主页丢掉（专有名词权重 1.5 / 总分 4.5 < 0.35）。
        - 纯中文（或只有年份）→ 覆盖率 ≥ 0.35，挡住只命中单个泛词的弱相关。
        - 权威分层优先（本次改造）：此前权威只按单一 +0.15 加权，央视网与
          县级政府网站同权，实测收录来源里权威域名仅占 4.6%。改为分层排序——
          中央媒体(2) > 政府/统计机构(1) > 普通站点(0)，让 3B 模型优先读到
          权威来源（模型只从候选里挑 2-3 条阅读，排序即决定它看不看得到）。
          分层不是无条件顶置：权威项的相关性需与最佳项相差不超过 RELEVANCE_FLOOR，
          避免把一个只沾边的政府页排到明显更对题的报道前面。
        - 单位内部页拒收（本次改造）：非权威域下标题命中"公示/通知/资格审查"
          的页面整条丢弃，不再降权。理由见 is_internal_source。
        """
        tokens = query_tokens(query)
        has_proper = bool(ascii_proper_tokens(tokens))
        q_tokens = query_tokens(getattr(self, "_question", query))
        q_terms = _bigrams(query) | _bigrams(getattr(self, "_question", query))
        failed = getattr(self, "_failed_urls", set())
        scored = []
        for r in results:
            if r.url in failed:  # 反爬 412/无关正文拒收过的页面，别再喂给模型
                continue
            host = (urlparse(r.url).hostname or "").lower()
            if is_blocked(host, r.url):
                continue
            # 非权威域的单位内部页（公示/通知/资格审查…）整条拒收。此前只做 -0.15
            # 降权，实测拦不住：北京体育大学就业指导中心的《关于开展2026届毕业生
            # 就业意向和进展调查的通知》在 51 份报告里被收录 3 次，正文全是调查安排、
            # 没有一个统计数据，却因字面密度高挤进候选前 5。这类页面任何话题下都不含
            # 一手数据，属于"收录了也用不上"——直接拒收比降权更省一轮抓取与摘要。
            # 权威域的「通知」是部委文件，恰是统计类问题的最佳来源，由
            # is_internal_source 内部豁免（它还会放行无域名的本地语料）。
            if is_internal_source(host, r.title or ""):
                self.trace.log("internal_page_dropped", url=r.url, title=r.title)
                continue
            text = f"{r.title} {r.snippet} {r.url}"
            cov = token_coverage(text, tokens)
            qcov = token_coverage(text, q_tokens)
            if has_proper:
                if not ascii_token_hit(text, tokens):
                    continue
            elif cov < 0.25 and qcov < 0.35:
                # 0.25：长中文搜索词 4 元组只命中一半（贸易伙伴 0.25）仍该留
                # qcov：搜索词对不上但页面明显对题（「出口高增长能延续吗」对「未来展望」）
                continue
            overlap = sum(1 for t in q_terms if t in text)
            tier = authority_tier(host)
            score = cov
            if tier == 0:
                # UGC/自媒体平台质量方差大，小幅降权
                # （单位内部页不在此处——上面已整条拒收）
                if is_low_quality(host):
                    score -= 0.10
            # 时效只做**正向**奖励，不惩罚旧内容。离线回放发现：一旦惩罚旧的，
            # "日期未知"（多数政府一手来源的 URL 不含日期）就会相对占优，
            # 把确知较旧的央视/新华网挤出 top-5（22 份快照出现，权威占比倒退）。
            # 只奖励新鲜则不会制造这种悖论——旧文只是不占便宜，不吃亏。
            if config.FRESH_DAYS and is_recent(r.published_at, config.FRESH_DAYS):
                score += config.FRESH_BONUS
            scored.append((tier, score, qcov, overlap, cov, r))
        if not scored:
            return []
        # 排序优先级：权威层级（受相关性下限约束）→ 排序分（相关性±质量/时效调整）
        # → 问题词覆盖 → 二元组重叠。floor 用**纯相关性** cov 计算，
        # 不让时效奖励抬高下限。
        best = max(x[4] for x in scored)
        floor = best - 0.3
        scored.sort(key=lambda x: (-(x[0] if x[4] >= floor else 0), -x[1], -x[2], -x[3]))
        return [r for *_, r in scored[:5]]

    def _authority_pass(self, query: str, kept: list, seen_urls: dict[str, str]) -> list:
        """常规搜索没有权威来源时，对权威域名做一次定向检索并前置。

        实测依据：原始搜索结果里央媒只占 2.8%，仅 12.9% 的查询能搜到任何央媒，
        权威结果平均排在原始列表第 4.3 位——被动等待排序把央视顶上来不够，
        必须主动去取。Tavily 用 include_domains、博查用 include（实测均生效，
        且博查不支持泛域 "gov.cn"，白名单已逐个枚举具体站点）。

        预算受 AUTHORITY_PASS_BUDGET 约束，避免每轮搜索都翻倍消耗额度。
        """
        if not config.AUTHORITY_PASS:
            return kept
        with self._lock:
            if self._authority_passes >= config.AUTHORITY_PASS_BUDGET:
                return kept
            self._authority_passes += 1
        try:
            extra = self.provider.search(query, max_results=8,
                                         include_domains=list(authority_domains()))
        except TypeError:
            # 提供方未实现 include_domains（自定义 SearchProvider）→ 静默跳过
            self.trace.log("authority_search_unsupported", query=query)
            return kept
        except Exception as e:
            self.trace.log("authority_search_fail", query=query, error=str(e)[:200])
            return kept
        self.stats.searches += 1
        self.stats.authority_searches += 1
        picked = self._filter_results(extra, query)
        self.trace.log("authority_search", query=query, results=len(extra), kept=len(picked))
        self.emit("search", {"query": query, "results": len(picked), "dropped": 0,
                             "authority": True})
        if not picked:
            return kept
        with self._lock:
            for r in picked:
                seen_urls.setdefault(r.url, r.title)
        # 权威结果前置，普通结果顺延并去重
        top_urls = {r.url for r in picked}
        return (picked + [r for r in kept if r.url not in top_urls])[:8]


    # ---- 工具执行 ----------------------------------------------------------
    def _dispatch_safe(self, name: str, args: dict, seen_urls: dict[str, str]) -> tuple[bool, str]:
        """带计时与异常兜底的执行入口（并行线程内跑的也是它）。"""
        t0 = time.time()
        try:
            outcome = self._dispatch(name, args, seen_urls)
        except Exception as e:
            outcome = (True, f"工具执行异常（{e}），请换一个。")
        self.trace.log("tool_done", name=name or "unknown", seconds=round(time.time() - t0, 1))
        return outcome

    def _dispatch(self, name: str, args: dict, seen_urls: dict[str, str]) -> tuple[bool, str]:
        """执行一次工具调用。返回 (是否模型合法调用, 回给模型的结果文本)。

        网络抓取失败不算模型的非法调用（valid=True），只有模型自身产生的
        未知工具/缺参数/编造链接才计为 invalid。
        """
        if name == "web_search" and isinstance(args.get("query"), str) and args["query"].strip():
            query = self._anchor_query(args["query"].strip())
            q_key = query.lower()
            if config.NAIVE:  # 朴素基线：无去重、无预算约束
                dup = over_budget = False
                self._searched.add(q_key)
            else:
                with self._lock:
                    dup = q_key in self._searched
                    over_budget = len(self._searched) >= config.MAX_SEARCHES
                    self._searched.add(q_key)
            if dup:
                # 3B 典型空转：反复搜同一关键词而不去阅读。直接顶回去读未读链接。
                unread = [u for u in seen_urls if u not in self.kb._by_url]
                hint = "\n".join(f"- {u}" for u in unread[:4])
                return True, ("该关键词已搜索过，不要重复搜索。请用 read_page 阅读以下未读路径：\n"
                              + (hint or "（已有结果都读过了，请换一个明显不同的关键词）"))
            if over_budget:
                return True, "本轮搜索次数已达上限，请只用 read_page 阅读已有搜索结果。"
            try:
                results = self.provider.search(query, max_results=8)
            except Exception as e:  # 搜索 API 故障
                return True, f"搜索请求失败（{e}），请稍后重试或换一个关键词。"
            self.stats.searches += 1
            kept = self._filter_results(results, query)
            raw_n = len(results)
            # 长查询把专有名词淹没时，用专有名词单独再搜一次（只在本轮颗粒无收时）
            if not kept:
                core = " ".join(ascii_proper_tokens(query_tokens(query)))
                if core and core.lower() != query.lower():
                    try:
                        extra = self.provider.search(core, max_results=8)
                    except Exception:
                        extra = []
                    else:
                        self.stats.searches += 1
                    kept = self._filter_results(extra, core)
                    raw_n += len(extra)
                    self.trace.log("tool_search_retry", query=core, results=len(extra), kept=len(kept))
            # 常规召回里没有权威来源 → 做一次权威域名定向检索并前置，
            # 否则 3B 只能在小站/聚合站里挑，报告来源档次被搜索源决定
            if kept and not any(authority_tier((urlparse(r.url).hostname or "").lower()) > 0
                                for r in kept):
                kept = self._authority_pass(query, kept, seen_urls)
            self.trace.log("tool_search", query=query, results=len(results), kept=len(kept))
            self.emit("search", {
                "query": query, "results": len(kept),
                "dropped": max(raw_n - len(kept), 0),
            })
            with self._lock:
                for r in kept:
                    seen_urls.setdefault(r.url, r.title)
            lines = [
                f"{i}. {r.title}{_date_tag(r.published_at)}\n"
                f"   路径: {r.url}\n   {r.snippet[:120]}"
                for i, r in enumerate(kept, 1)
            ]
            if not lines:
                return True, (
                    f"搜到 {raw_n} 条但均未命中关键词，已丢弃。"
                    "请换一个更短的专有名词再搜（不要把多个主题拼在一个搜索词里）。"
                )
            return True, "\n".join(lines)

        if name == "read_page" and isinstance(args.get("path") or args.get("url"), str):
            raw = (args.get("path") or args.get("url")).strip()
            url = self._resolve_path(raw, seen_urls)
            if url is None:
                return False, (
                    f"路径 {raw} 不在搜索结果中。只能阅读 web_search 返回的路径，"
                    "请原样复制，不要添加网址前缀。"
                )
            try:
                digest, sid = self._read_and_digest(url, seen_urls[url])
            except Exception as e:
                with self._lock:
                    self._failed_urls.add(url)
                return True, f"文档读取失败（{e}），请换一个阅读。"
            return True, f"已收录资料 [{sid}]《{seen_urls[url]}》\n摘要：{digest[:200]}"

        return False, (
            f"无法识别的工具调用：{name} {json.dumps(args, ensure_ascii=False)[:200]}。"
            "可用工具只有 web_search(query) 和 read_page(path)。"
        )

    @staticmethod
    def _resolve_path(raw: str, seen_urls: dict[str, str]) -> str | None:
        """路径容错：3B 模型偶尔给文件路径编造 https:// 前缀或截断。
        剥掉协议与域名后做后缀匹配，唯一命中即视为该文档。"""
        if raw in seen_urls:
            return raw
        cleaned = re.sub(r"^https?://[^/]+/", "", raw)
        if cleaned != raw and cleaned in seen_urls:
            return cleaned
        candidates = [u for u in seen_urls if u.endswith(cleaned) or cleaned.endswith(u)]
        return candidates[0] if len(candidates) == 1 else None

    def _read_and_digest(self, url: str, title: str) -> tuple[str, int]:
        with self._lock:
            if url in self.kb._by_url:  # 补搜轮常重复阅读已收录页面，跳过重复抓取与摘要
                return "（该资料此前已收录，无需重复阅读）", self.kb._by_url[url]
        try:
            document = fetch_document(url)
            text = document.text
            terms = content_terms(getattr(self, "_question", ""))
            evidence = "\n".join(evidence_blocks(text)).lower()
            if terms and sum(term in evidence for term in terms) < min(2, len(terms)):
                raise ValueError("正文与调研主题无关")
        except ValueError as exc:
            self.stats.pages_rejected += 1
            self.trace.log("source_rejected", url=url, reason=str(exc))
            self.emit("research", {"status": f"未收录《{title}》：{exc}"})
            raise
        digest = self._digest(text)
        with self._lock:
            if len(self.kb) >= config.MAX_SOURCES:
                raise ValueError("来源预算已用完")
            sid = self.kb.add(url, title, digest, text=text,
                              published_at=document.published_at, publisher=document.publisher,
                              retrieved_at=document.retrieved_at)
        self.stats.pages_fetched += 1
        self.trace.log("tool_read", url=url, title=title, source_id=sid, digest=digest[:300])
        self.emit("read", {"source_id": sid, "title": title, "url": url})
        return digest, sid

    def _digest(self, text: str) -> str:
        return select_evidence(text, getattr(self, "_question", ""), max_chars=900)

    # ---- 防线 3：规则兜底采集 ------------------------------------------------
    def scripted_collect(self, queries: list[str], per_query_pages: int = 2) -> None:
        """不经模型决策的确定性采集：每个查询词阅读靠前的 N 个结果。

        与模型路径同样过相关性过滤——兜底采集宁缺毋滥，
        不能成为垃圾结果的旁门（实测曾从这里混入无关页面）。
        """
        self.trace.log("scripted_collect_start", queries=queries)
        for query in queries:
            if len(self.kb) >= config.MAX_SOURCES:
                break
            query = self._anchor_query(query)
            try:
                results = self.provider.search(query, max_results=per_query_pages + 3)
                self.stats.searches += 1
            except Exception as e:
                self.trace.log("scripted_search_fail", query=query, error=str(e))
                continue
            kept = self._filter_results(results, query)
            # 兜底采集同样争取权威来源：无权威时做一次定向检索并前置
            if kept and not any(authority_tier((urlparse(r.url).hostname or "").lower()) > 0
                                for r in kept):
                kept = self._authority_pass(query, kept, {})
            self.trace.log("scripted_search", query=query, kept=len(kept))
            # 失败不占配额、不重试：第 1 名被反爬挡住时顺延读下一条，
            # 否则海关官网一个 412 就能让整个查询颗粒无收（实测踩过）。
            collected = 0
            for r in kept:
                if collected >= per_query_pages or len(self.kb) >= config.MAX_SOURCES:
                    break
                try:
                    self._read_and_digest(r.url, r.title)
                    collected += 1
                except Exception as e:
                    self._failed_urls.add(r.url)
                    self.trace.log("scripted_read_fail", url=r.url, error=str(e))
        self.trace.log("scripted_collect_end", sources=len(self.kb.sources))
