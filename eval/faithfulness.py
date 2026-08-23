"""引用忠实度校验：带引用标记的句子是否真的被它引用的来源支持。

与 writer._validate_citations 的分工：后者只保证引用编号指向真实来源（格式层），
本模块校验内容层——被引来源的摘要是否支撑句子里的论断。两层校验：

- 规则层（零成本）：抽取句子中的数字（年份/百分比/数量），与被引来源摘要比对，
  找不到依据的记为 num_flags。型号编号（Qwen2.5、GPT-4、5G）与题目自带的
  数字（如年份）不算；允许四舍五入容差（来源 106.9 万 vs 句子"约 107 万"）。
- LLM 层（低成本）：对带引用的句子让模型判 支持/部分支持/不支持。

  faithfulness = (支持 + 0.5×部分支持) / 已判定句数

判定集偏向可疑句：规则层标记过数字存疑的句子优先送 LLM，其余按序补足上限。
因此该分数是对报告的保守估计，而非无偏抽样。仅在评估时运行，不拖慢日常生成。
"""
import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

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


def check_report(llm, report: str, question: str, sources: list[dict],
                 max_judges: int = MAX_JUDGES) -> dict:
    """入口：report 为生成的 Markdown，sources 为 [{id, digest}, ...]。"""
    by_id = {s["id"]: s["digest"] for s in sources}
    items = []
    for sent, ids in cited_sentences(report):
        digests = [by_id[i] for i in ids if i in by_id]
        if not digests:
            continue
        items.append({
            "sentence": sent, "ids": ids,
            "num_flags": missing_numbers(sent, " ".join(digests), question),
        })
    items.sort(key=lambda x: -len(x["num_flags"]))  # 可疑句优先裁决
    chosen = items[:max_judges]

    counts = {"支持": 0, "部分支持": 0, "不支持": 0}
    unjudged = 0
    for it in chosen:
        digest = " ".join(by_id[i] for i in it["ids"] if i in by_id)
        verdict = judge(llm, it["sentence"], digest)
        it["verdict"] = verdict
        if verdict is None:
            unjudged += 1
        else:
            counts[verdict] += 1
    judged = sum(counts.values())
    total = len(list(_iter_sentences(report)))
    return {
        "sentences": total,
        "citation_density": round(len(items) / total, 3) if total else None,
        "cited_sentences": len(items),
        "judged": judged,
        "unjudged": unjudged,
        "supported": counts["支持"],
        "partial": counts["部分支持"],
        "refuted": counts["不支持"],
        "num_flagged": sum(1 for it in items if it["num_flags"]),
        "faithfulness": round((counts["支持"] + 0.5 * counts["部分支持"]) / judged, 3) if judged else None,
        "details": chosen,
    }
