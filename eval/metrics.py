"""评估指标计算。"""
from agent.factcheck import _iter_sentences


def coverage(report: str, must_cover: list[str]) -> dict:
    """要点覆盖率：评分表关键词在报告正文中的字面命中比例。

    rubric 设计原则：关键词选为报告中大概率原样出现的术语，
    因此字面匹配即可作为低成本、可复现的代理指标。
    """
    body = "\n".join(_iter_sentences(report))
    hits = [kw for kw in must_cover if kw in body]
    return {
        "hits": hits,
        "missed": [kw for kw in must_cover if kw not in body],
        "rate": round(len(hits) / len(must_cover), 3) if must_cover else 1.0,
        "kind": "body_keyword_coverage",
    }


def aggregate(results: list[dict]) -> dict:
    """汇总单任务结果为全局指标。失败任务按零覆盖计入分母。"""
    n = len(results)
    if n == 0:
        return {}
    ok = [r for r in results if r.get("ok")]
    faith_scores = [r["faith"]["faithfulness"] for r in ok
                    if r.get("faith", {}).get("faithfulness") is not None]
    densities = [r["faith"]["citation_density"] for r in ok
                 if r.get("faith", {}).get("citation_density") is not None]
    judged = sum(r.get("faith", {}).get("judged", 0) for r in ok)
    sentences = sum(r.get("faith", {}).get("sentences", 0) for r in ok)
    return {
        "tasks": n,
        "success": len(ok),
        "coverage_avg": round(sum(r["coverage"]["rate"] for r in ok) / n, 3),
        "faithfulness_avg": round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None,
        "faithfulness_reports": len(faith_scores),
        "judged_sentences": judged,
        "total_sentences": sentences,
        "judgment_coverage": round(judged / sentences, 3) if sentences else None,
        "reports_without_flags": sum(r.get("quality", {}).get("status") == "no_flags" for r in ok),
        "citation_density_avg": round(sum(densities) / len(densities), 3) if densities else None,
        "json_first_try_rate_avg": round(
            sum(r["stats"].get("json_first_try_rate", 0) for r in ok) / max(len(ok), 1), 3),
        "tool_valid_rate_avg": round(
            sum(r["stats"].get("tool_valid_rate", 0) for r in ok) / max(len(ok), 1), 3),
        "llm_calls_avg": round(sum(r["stats"].get("llm_calls", 0) for r in ok) / max(len(ok), 1), 1),
        "seconds_avg": round(sum(r["stats"].get("total_seconds", 0) for r in ok) / max(len(ok), 1), 1),
    }
