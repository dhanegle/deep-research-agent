"""Extractive evidence selection; numeric table values never pass through an LLM."""
import csv
import hashlib
import io
import re
from decimal import Decimal, InvalidOperation


_FILLER = re.compile(r"调研|研究|分析|情况|现状|相关|报告|总体|主要|最新|概述|概况|方面|一下")
# 虚词/助词：作为二元组的切分点，不让二元组跨越它们。
# 只收「的/地/得/之/了/着/在」这类纯功能字——「上/下/中」等可能构成实义词
# （上半年、下半年），不切分，避免误伤。
_PARTICLES = re.compile(r"[的地得之了着在]")
_TAIL = re.compile(r"^\s*(?:为你推荐|相关推荐|推荐阅读|相关阅读|热门推荐|猜你喜欢)\s*[:：]?\s*$")
_SCRIPT = re.compile(r"\b(?:var|function|window|document|prototype|Reflect|__proto__)\b|=>")
_ASPECTS = (
    (r"技术|实现|路径", ("量化", "算力", "架构", "调度", "部署", "训练", "工艺", "路线")),
    (r"性能|能耗|功耗", ("内存", "功耗", "算力", "温控", "降频", "时延", "精度", "能耗")),
    (r"企业|车企|品牌", ("企业", "车企", "品牌", "厂商", "公司", "领跑")),
    (r"政策|监管|环境|风险|挑战", ("关税", "合规", "监管", "政策", "壁垒", "压力", "风险", "瓶颈")),
    # 前瞻段落常不含"未来/预计"，而写"下半年形势判断""接下来的核心方向"
    # （实测光明网展望专稿：整章展望正文一个旧词都不命中，只有小标题进了片段）。
    # "预测"必须入表：「毕业总人数预测」这类小节标题只含"预测"，
    # 不入表则 aspect_terms 为空，align_feedback 给不出任何纠偏提示。
    (r"趋势|未来|前景|展望|预测", ("未来", "预计", "预测", "倾向", "趋势", "有望", "策略",
                            "展望", "前景", "下半年", "中长期", "接下来", "后续")),
    (r"应用|场景", ("应用", "场景", "用于", "落地", "内置")),
    (r"市场|伙伴|区域|分布", ("市场", "地区", "区域", "分布", "伙伴", "国别")),
    # "人数"必须入表：「毕业总人数预测」只含"总人数"，旧表只认"总量/数量"，
    # 导致该节 aspect_terms 为空——既给不出纠偏提示，也让扣题检查无从判别。
    # 权威报道写"规模预计1270万人"（含规模/万人）而聚合站复述"毕业生人数"，
    # 两者都要能对上，否则会把权威正文误判为跑题。
    (r"总量|规模|增速|数量|人数", ("总量", "规模", "增速", "同比", "增长", "总值", "人数", "万人")),
    # 不收"占比"：任何带百分比的句子都含它，与商品无关。实测贸易伙伴
    # 排名句（"美国排名第一，占比7.88%"）靠它通过了「主要商品类别」的扣题检查。
    (r"商品|产品|品类|类别|结构|构成", ("商品", "产品", "品类", "类别", "结构", "构成")),
)


def aspect_terms(section: str) -> set[str]:
    """Small, inspectable synonym groups for generic research headings."""
    return {term for pattern, terms in _ASPECTS if re.search(pattern, section) for term in terms}


def content_terms(text: str) -> set[str]:
    # 数字连同其后的单位字一起剥掉：「2026年毕业」只留「毕业」。
    # 只去数字会残留「年」，与实义词拼出「年毕」这类跨界二元组——
    # 讲「2026届毕业生」的对题页面永远不含「年毕」，会被相关性门槛误杀。
    text = re.sub(r"\d+(?:\.\d+)?[年月日届万亿千百人名次个元辆吨%％]*", " ", text.lower())
    text = _FILLER.sub(" ", text)
    terms = set(re.findall(r"[a-z][a-z0-9_.-]+", text))
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        # 虚词作切分点，避免「的毕业」拼出「的毕」这类跨虚词伪二元组。
        # 实测：问题「2026年的毕业情况」原先生成 {毕业, 的毕}，而页面质检
        # 要求两条全中——人民网/新华网正文（只含"毕业"）被拒收，照抄问题
        # 措辞的 SEO 聚合站反而命中"的毕"被收录。相关性门槛恰好把权威报道
        # 挡在门外，是"总收录小网站"的核心成因之一。
        for seg in _PARTICLES.split(part):
            terms.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return terms


def clean_page_text(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if _TAIL.match(line):
            break
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def page_problem(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 40:
        return "正文过短"
    script_hits = len(_SCRIPT.findall(text))
    prose = len(re.findall(r"[\u4e00-\u9fff]", text))
    if script_hits >= 8 and (prose < 80 or script_hits > prose / 5):
        return "脚本或验证页面"
    if re.search(r"[0-9a-fA-F]{300,}", text) and script_hits >= 3:
        return "混淆脚本页面"
    if re.search(r"captcha|verify you are human|just a moment|cf[_-]?app[_-]?waf|请开启\s*javascript", text, re.I):
        return "访问验证页面"
    return ""


def content_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()


def _trade_rows(text: str) -> list[str]:
    """Decode the explicit two-level Quantity/Value customs table schema only."""
    lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    if not lines or not all(s in text for s in ("Quantity Unit", "Percentage Change", "Commodity")):
        return []
    currency = re.search(r"Unit\s*:\s*US\$\s*([\d,]+)", text, re.I)
    if not currency:
        return []
    scale = Decimal(currency.group(1).replace(",", ""))
    period = re.search(r"Value\s*,\s*(\d{1,2})\.(20\d{2})", text)
    context = f"{period[2]}年{int(period[1])}月" if period else "期间未标明"
    context += "出口商品" if "Export" in text else "进口商品" if "Import" in text else "商品"
    rows = []
    for parsed in csv.reader(io.StringIO("\n".join(lines)), delimiter="|"):
        cells = [cell.strip() for cell in parsed[1:-1]]
        if len(cells) != 6:
            continue
        name, unit, qty, value, qty_change, value_change = cells
        try:
            dollars = Decimal(value.replace(",", "")) * scale
        except InvalidOperation:
            continue
        quantity = ""
        if unit in {"T", "10000T", "KG", "10000L", "10000CR"}:
            try:
                units = {"T": (1, "吨"), "10000T": (10000, "吨"), "KG": (1, "千克"),
                         "10000L": (10000, "升"), "10000CR": (10000, "条")}
                multiplier, label = units[unit]
                quantity = f"数量{Decimal(qty.replace(',', '')) * multiplier:f}{label}；"
            except InvalidOperation:
                pass
        changes = []
        for label, cell in (("数量同比", qty_change), ("金额同比", value_change)):
            try:
                changes.append(f"{label}{Decimal(cell.replace(',', '')):f}%")
            except InvalidOperation:
                pass
        rows.append(f"{context}，{name}：{quantity}金额{dollars:f}美元；{'；'.join(changes)}。")
    return rows


def evidence_blocks(text: str) -> list[str]:
    normalized = _trade_rows(text)
    if normalized:
        return normalized
    blocks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Keep complete sentences and table rows; a character slice can detach a unit.
        if len(line) > 700 and not line.startswith("|"):
            blocks.extend(s.strip() for s in re.split(r"(?<=[。！？])", line) if s.strip())
        else:
            blocks.append(line)
    return blocks


def select_evidence(text: str, query: str, max_chars: int = 2400,
                    boost: set[str] | None = None) -> str:
    """boost 是小节特征词，权重高于问题词。

    问题词（"进出""出口""我国"）在长文的几乎每个段落都命中，与小节特征词
    同权时段落排序实际由问题词决定：实测「未来展望」从光明网展望专稿里
    选出的全是总量段落，文末的下半年展望部分被 max_chars 截掉，模型看不到
    任何前瞻内容，两次重试都只能重发总量数字，整节因此占位。
    """
    blocks = evidence_blocks(text)
    terms = content_terms(query)
    boost = boost or set()
    numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
    scored = sorted(enumerate(blocks), key=lambda row: (
        -sum(term in row[1].lower() for term in terms)
        -3 * sum(term in row[1].lower() for term in boost)
        -0.2 * len(numbers.intersection(re.findall(r"\d+(?:\.\d+)?", row[1]))), row[0]))
    selected = []
    size = 0
    for index, block in scored:
        if size + len(block) > max_chars:
            continue
        selected.append((index, block))
        size += len(block) + 1
    # A table excerpt needs its header even when its rows rank independently.
    if not _trade_rows(text) and any(block.startswith("|") for _, block in selected):
        headers = [(i, block) for i, block in enumerate(blocks[:5]) if block.startswith("|")]
        selected = list(dict(headers + selected).items())
    return "\n".join(block for _, block in sorted(selected))
