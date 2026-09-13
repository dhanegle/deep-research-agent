"""引用忠实度核查原语：从 eval/faithfulness.py 抽出的可复用层。

两层校验：
- 规则层（零成本）：抽取句子中的数字（年份/百分比/数量），与被引来源摘要比对，
  找不到依据的记为 missing_numbers。型号编号（Qwen2.5、GPT-4、5G）与题目自带的
  数字（如年份）不算；允许四舍五入容差（来源 106.9 万 vs 句子"约 107 万"）。
- LLM 层（低成本）：对带引用的句子让模型判 支持/部分支持/不支持。

eval/faithfulness.py（评估编排，全报告抽样裁决）和 agent/reviewer.py（自审闭环，
按节精准裁决）共用这些原语，避免 agent→eval 的反向依赖。
"""
import re
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel
from typing import Literal

from .evidence import evidence_blocks

CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
MAX_JUDGES = 12  # 每份报告最多送 LLM 裁决的句子数，控制评估耗时


def _norm(s: str) -> str:
    return re.sub(r"[,\s]", "", s)


def extract_numbers(sentence: str) -> list[str]:
    """抽取数量类数字。紧邻 ASCII 字母的（Qwen2.5、GPT-4、5G）是型号编号，跳过。"""
    text = CITE_RE.sub("", sentence)  # 引用标记 [3] 本身不算数字
    nums = []
    for m in NUM_RE.finditer(text):
        start, end = m.span()
        before, after = text[:start], text[end:end + 1]
        if re.search(r"[A-Za-z][\-_]?$", before) or re.match(r"[A-Za-z]", after):
            continue
        nums.append(m.group(0))
    return nums


_SCALES = {"": 1, "百": 100, "千": 1000, "万": 10000, "百万": 1000000,
           "千万": 10000000, "亿": 100000000, "十亿": 1000000000, "万亿": 1000000000000}
_UNIT = re.compile(
    r"\s*(万亿|十亿|千万|百万|亿|万|千|百)?\s*"
    r"(个百分点|人民币|美元|千克|公斤|元|人|名|辆|吨|升|条|%|％|年|届|月|日|GB|MB|秒|次|个)?"
)
_YEARS = re.compile(r"((?:19|20)\d{2})\s*(年|届)")


@dataclass
class _Number:
    raw: str
    value: Decimal
    unit: str
    scale: Decimal
    year: str
    metric: str
    approximate: bool
    date: bool = False


def _numbers(text: str) -> list[_Number]:
    text = CITE_RE.sub("", re.sub(r"https?://\S+", "", text))
    out = []
    for match in NUM_RE.finditer(text):
        start, end = match.span()
        before, after = text[:start], text[end:]
        if re.search(r"[A-Za-z][\-_]?$", before) or re.match(r"[A-Za-z]", after):
            continue
        scale_name, unit = _UNIT.match(after).groups()
        scale = Decimal(_SCALES[scale_name or ""])
        unit = {"人民币": "元", "％": "%", "公斤": "千克", "届": "年", "名": "人"}.get(unit, unit or "")
        value = Decimal(match.group().replace(",", "")) * scale
        context = re.split(r"[。！？\n]", before)[-1]
        clause = re.split(r"[，,；;]", context)[-1]
        if unit in {"%", "个百分点"} and not match.group().startswith(("-", "+")):
            if re.search(r"(?:下降|减少|降低|下滑|负增长)[^\d]{0,5}$", clause):
                value = -value
        date = unit == "年" and bool(re.match(r"\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", after))
        years = [m[1] for m in _YEARS.finditer(context)
                 if not re.match(r"\s*\d{1,2}\s*月\s*\d{1,2}\s*日", context[m.end():])]
        metrics = re.findall(r"金额|价值|出口额|进口额|数量|出口量|进口量", clause)
        metric = ""
        if metrics:
            metric = "value" if metrics[-1] in {"金额", "价值", "出口额", "进口额"} else "quantity"
        out.append(_Number(match.group(), value, unit, scale, years[-1] if years else "", metric,
                           bool(re.search(r"约|大约|近|左右", clause[-6:])), date))
    return out


def missing_numbers(sentence: str, digest: str, question: str) -> list[str]:
    """按数值、单位、数据年份和已知指标核对；问题里的数字不是证据。"""
    source = _numbers("\n".join(evidence_blocks(digest)))
    miss = []
    for item in _numbers(sentence):
        candidates = [s for s in source if s.unit == item.unit
                      and not (item.unit == "年" and not item.date and s.date)
                      and not (item.year and s.year and item.year != s.year)
                      and not (item.metric and s.metric and item.metric != s.metric)]
        supported = any(s.value == item.value or (
            item.approximate and item.unit not in {"年", "月", "日"}
            and item.value / item.scale == round(item.value / item.scale)
            and round(s.value / item.scale) == item.value / item.scale
        ) for s in candidates)
        if not supported and item.raw not in miss:
            miss.append(item.raw)
    return miss


def unsupported_citations(sentence: str, evidence_by_id: dict[int, str],
                          question: str) -> list[int]:
    """多引用句里哪些编号对不上本句的任何数字。

    核验把被引来源拼成一份证据整体判定，只要其中一条支持整句就通过，
    3B 顺手多挂的编号查不出来——实测"2026年上半年机电产品出口占63.5%"
    引了 [3][5][6]，[6] 是 2022 年发布的 2020 年报告，判定仍是"支持"。

    只在另有编号能独立支持整句时才判多余：否则各条证据"各支持一部分"的
    合法多引用会被误拆。非数字句没有零成本判据，交给 LLM 层。

    年月日不算判据：几乎每条来源都写着同一个年份，把它算作命中会让本规则
    对含年份的句子（也就是绝大多数句子）永远不触发。
    """
    nums = {n.raw for n in _numbers(CITE_RE.sub("", sentence))
            if n.unit not in {"年", "月", "日"}}
    if len(evidence_by_id) < 2 or not nums:
        return []
    missing = {i: set(missing_numbers(sentence, ev, question)) & nums
               for i, ev in evidence_by_id.items()}
    if not any(not miss for miss in missing.values()):
        return []
    return sorted(i for i, miss in missing.items() if miss == nums)


def _iter_sentences(report: str):
    """逐句产出正文（跳过标题、meta 行，到"参考来源"页脚为止）。

    只剩引用标记的"空句"（3B 写作时偶尔把 [1] 单独成行）没有可核验
    的论断，一并跳过，不进密度分母也不进判定集。
    """
    for line in report.splitlines():
        if re.match(r"^##\s*参考来源", line.strip()):
            break
        if line.lstrip().startswith(("#", ">")):
            continue
        text = line.strip().lstrip("-* ").strip()
        if not text:
            continue
        text = re.sub(r"([。！？；])\s*((?:\[\d+\][ \t]*)+)",
                      lambda m: m[2].strip() + m[1], text)
        for sent in re.split(r"(?<=[。！？；])", text):
            sent = sent.strip()
            if sent and re.search(r"[\u4e00-\u9fffA-Za-z]", CITE_RE.sub("", sent)):
                yield sent


def cited_sentences(report: str) -> list[tuple[str, list[int]]]:
    """返回带引用标记的 (句子, 来源编号)。"""
    out: list[tuple[str, list[int]]] = []
    for sent in _iter_sentences(report):
        ids = sorted({int(m) for m in CITE_RE.findall(sent)})
        if ids:
            out.append((sent, ids))
    return out


class Judgment(BaseModel):
    verdict: Literal["支持", "部分支持", "不支持"]
    reason: str = ""


JUDGE_PROMPT = """\
你是事实核查员。判断"来源资料"能否支持"报告句子"中的事实性内容。

来源资料：
{digest}

报告句子：{sentence}

要求：
1. 只依据来源资料判断，不要用自己的知识补全来源里没有的信息；
2. 数字允许合理的四舍五入与单位换算（如 106.9万 写成 约107万），不算错误；
3. 句子只有一部分被支持判"部分支持"；核心事实与来源矛盾、或来源完全未提及时判"不支持"。
只输出 JSON：{{"verdict": "支持/部分支持/不支持", "reason": "15字以内理由"}}
"""


def judge(llm, sentence: str, digest: str) -> str | None:
    """LLM 裁决单句。判定失败返回 None（模型故障不应记在报告头上）。"""
    prompt = JUDGE_PROMPT.format(digest=digest, sentence=sentence)
    try:
        return llm.chat_json([{"role": "user", "content": prompt}], Judgment, tag="faith").verdict
    except ValueError:
        return None
