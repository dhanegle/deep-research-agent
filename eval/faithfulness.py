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

底层原语（extract_numbers/missing_numbers/judge 等）已移入 agent/factcheck.py，
本文件保留评估编排器 check_report——用 MAX_JUDGES 做全报告抽样裁决。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.factcheck import (
    CITE_RE, NUM_RE, MAX_JUDGES,
    _iter_sentences, cited_sentences, missing_numbers, judge,
)
from agent.evidence import select_evidence


def check_report(llm, report: str, question: str, sources: list[dict],
                 max_judges: int = MAX_JUDGES) -> dict:
    """入口：report 为生成的 Markdown，sources 为 [{id, digest}, ...]。"""
    by_id = {s["id"]: s.get("text") or s["digest"] for s in sources}
    items = []
    seen = set()
    for sent, ids in cited_sentences(report):
        if sent in seen:
            continue
        seen.add(sent)
        digests = [select_evidence(by_id[i], sent, 1800) for i in ids if i in by_id]
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
        digest = " ".join(select_evidence(by_id[i], it["sentence"], 1800) for i in it["ids"] if i in by_id)
        verdict = "不支持" if it["num_flags"] else judge(llm, it["sentence"], digest)
        it["verdict"] = verdict
        if verdict is None:
            unjudged += 1
        else:
            counts[verdict] += 1
    judged = sum(counts.values())
    total = len(set(_iter_sentences(report)))
    return {
        "sentences": total,
        "citation_density": round(len(items) / total, 3) if total else None,
        "cited_sentences": len(items),
        "judged": judged,
        "judgment_coverage": round(judged / total, 3) if total else None,
        "supported_sentence_rate": round(counts["支持"] / total, 3) if total else None,
        "unjudged": unjudged,
        "supported": counts["支持"],
        "partial": counts["部分支持"],
        "refuted": counts["不支持"],
        "num_flagged": sum(1 for it in items if it["num_flags"]),
        "faithfulness": round((counts["支持"] + 0.5 * counts["部分支持"]) / judged, 3) if judged else None,
        "details": chosen,
    }
