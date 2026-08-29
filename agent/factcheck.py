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

from pydantic import BaseModel
from typing import Literal

CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
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


def _rounded_hit(n: str, src_norm: str) -> bool:
    """整数容差：句子写"约107万"而来源是 106.9 万，视为有依据。"""
    val = float(n)
    if val != int(val):
        return False  # 句子自己给了小数还匹配不上，不再宽容
    val = int(val)
    for m in NUM_RE.finditer(src_norm):
        s = float(m.group(0))
        if s == val or (s != int(s) and round(s) == val):
            return True
    return False


def missing_numbers(sentence: str, digest: str, question: str) -> list[str]:
    """规则层：句子中出现、但既不在被引摘要也不在题目里、也无容差命中的数字。"""
    src, q = _norm(digest), _norm(question)
    miss = []
    for raw in extract_numbers(sentence):
        n = _norm(raw)
        if n in q or n in src or _rounded_hit(n, src):
            continue
        miss.append(raw)
    return miss


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
