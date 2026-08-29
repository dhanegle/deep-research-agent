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

from pydantic import BaseModel, Field

from .search.base import SearchProvider
from .search.pages import fetch_page

from . import config
from .knowledge import (
    KnowledgeBase, _bigrams, ascii_proper_tokens, ascii_token_hit,
    query_tokens, token_coverage,
)

# 文库/课件站几乎没有可用正文，录用后摘要是空壳，报告会被带跑；
# 题库/答疑站是大纲空词的字面命中大户——「政策影响因素」这类查询词
# 完美命中选择题题面，任何搜索引擎都会把它们排在权威媒体前面。
_BLOCKED_HOSTS = (
    "docin.com", "doc88.com", "taodocs.com", "book118.com",
    "360doc.com", "wenku.baidu.com", "max.book118.com",
    "jutiku.cn", "xilvlaw.com", "wkda.cn", "027art.com",
    "renrendoc.com", "cooco.net.cn", "eepw.com/shiti",
)
# 中央权威媒体与政府站点：相关性打分小幅加权，措辞不同也能排到普通站前面
_AUTHORITY_HOSTS = (
    ".gov.cn", "cctv.com", "cctv.cn", "news.cn", "xinhuanet.com",
    "people.com.cn", "china.com.cn", "cetv.cn", "cnr.cn",
)
# 单位内部页面标题特征（公示/资格审查/放假通知……任何话题都成立）：
# 这类页面字面密度高、更新勤，总量统计类查询下常压过权威报道，非权威域名一律降权
_INTERNAL_PAGE_RE = re.compile(
    r"公示|教务处|资格审查|录取名单|成绩查询|放假安排|校历|返校"
    r"|关于做好.{0,12}的通知"
)
_JUNK_DIGEST = re.compile(
    r"appkey|cf[_-]?app[_-]?waf|ac_opt|enablejavascript|just a moment|请开启\s*javascript",
    re.I,
)
from .llm import LLM, tool_call_args, tool_call_name
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

DIGEST_PROMPT = (
    "你是一个严谨的资料整理员。用不超过 250 字概括给定网页正文的关键事实，"
    "保留具体数字、时间、机构名与结论，不要添加评论。只输出摘要正文。"
)

# 「调研一下2026年的毕业情况」→「2026年的毕业情况」：剥掉请求动词留核心
_QUESTION_PREFIX = re.compile(
    r"^(请|帮我|我想|给我|麻烦)?(深度|全面|快速)?"
    r"(调研|研究|分析|了解|查一下|查查|看看|搜集|调查|统计)(一下|一番)?"
)


def _strip_question_verbs(question: str) -> str:
    return _QUESTION_PREFIX.sub("", question).strip() or question


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
        self._lock = threading.Lock()     # 并行执行工具时保护共享状态
        self._core = ""                   # 问题核心词（剥掉请求动词），大纲空词改写用

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
            if len(self.kb.sources) >= 8:
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
        shared = (_bigrams(query) & _bigrams(core)) - _GENERIC_BIGRAMS
        if shared:
            return query
        return f"{core} {query}"

    def _filter_results(self, results: list, query: str) -> list:
        """搜索结果相关性过滤与重排。

        实测问题：查专有名词（如 "linuxsb"）时搜索引擎常返回泛主题页面
        （如"什么是Linux"），3B 模型照单全收导致报告跑题。
        规则：
        - 查询含 ASCII 专有名词 → 必须命中该词（linux.sb 等分隔符已归一）；
          命中后不再用 0.35 覆盖率否决——否则 "linuxsb 主要 讨论 领域"
          会把真正的 linux.sb 主页丢掉（专有名词权重 1.5 / 总分 4.5 < 0.35）。
        - 纯中文（或只有年份）→ 覆盖率 ≥ 0.35，挡住只命中单个泛词的弱相关。
        """
        tokens = query_tokens(query)
        has_proper = bool(ascii_proper_tokens(tokens))
        q_tokens = query_tokens(getattr(self, "_question", query))
        q_terms = _bigrams(query) | _bigrams(getattr(self, "_question", query))
        scored = []
        for r in results:
            host = (r.url or "").lower()
            if any(b in host for b in _BLOCKED_HOSTS):
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
            auth = 0.15 if any(a in host for a in _AUTHORITY_HOSTS) else 0.0
            cov_eff = cov + auth
            # 非权威域名的单位内部页面（公示/通知/教务处…）降权：
            # 权威域不受影响——部委通知恰恰是统计类问题的最佳来源
            if auth == 0.0 and _INTERNAL_PAGE_RE.search(r.title or ""):
                cov_eff -= 0.15
            scored.append((cov_eff, qcov, overlap, r))
        scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        return [r for _, _, _, r in scored[:5]]

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
                    over_budget = len(self._searched) >= 12
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
            self.trace.log("tool_search", query=query, results=len(results), kept=len(kept))
            self.emit("search", {
                "query": query, "results": len(kept),
                "dropped": max(raw_n - len(kept), 0),
            })
            with self._lock:
                for r in kept:
                    seen_urls.setdefault(r.url, r.title)
            lines = [f"{i}. {r.title}\n   路径: {r.url}\n   {r.snippet[:120]}" for i, r in enumerate(kept, 1)]
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
        text = fetch_page(url)
        digest = self._digest(text)
        if _JUNK_DIGEST.search(digest) or len(re.sub(r"\s+", "", digest)) < 40:
            raise ValueError("页面无有效正文（验证页或空壳）")
        with self._lock:
            sid = self.kb.add(url, title, digest)
        self.stats.pages_fetched += 1
        self.trace.log("tool_read", url=url, title=title, source_id=sid, digest=digest[:300])
        self.emit("read", {"source_id": sid, "title": title, "url": url})
        return digest, sid

    def _digest(self, text: str) -> str:
        text = (text or "").strip()[:6000]
        if not text:
            return "（页面无有效正文）"
        self.stats.digest_calls += 1
        return self.llm.chat_text([
            {"role": "system", "content": DIGEST_PROMPT},
            {"role": "user", "content": text},
        ]) or "（摘要生成失败）"

    # ---- 防线 3：规则兜底采集 ------------------------------------------------
    def scripted_collect(self, queries: list[str], per_query_pages: int = 2) -> None:
        """不经模型决策的确定性采集：每个查询词阅读靠前的 N 个结果。

        与模型路径同样过相关性过滤——兜底采集宁缺毋滥，
        不能成为垃圾结果的旁门（实测曾从这里混入无关页面）。
        """
        self.trace.log("scripted_collect_start", queries=queries)
        for query in queries:
            query = self._anchor_query(query)
            try:
                results = self.provider.search(query, max_results=per_query_pages + 3)
            except Exception as e:
                self.trace.log("scripted_search_fail", query=query, error=str(e))
                continue
            kept = self._filter_results(results, query)
            self.trace.log("scripted_search", query=query, kept=len(kept))
            for r in kept[:per_query_pages]:
                try:
                    self._read_and_digest(r.url, r.title)
                except Exception as e:
                    self.trace.log("scripted_read_fail", url=r.url, error=str(e))
        self.trace.log("scripted_collect_end", sources=len(self.kb.sources))
