"""评测入口：对任务集全量运行管线并输出指标。

可复现评估工作流（磁盘快照回放）：
1. 首次：CACHE_MODE=record SEARCH_PROVIDER=tavily python eval/run_eval.py
   → 真实搜索并把结果快照落盘（消耗 API 额度，一次性）；
2. 之后：CACHE_MODE=replay python eval/run_eval.py
   → 全部读缓存，零 API 消耗、不依赖网络，结果可复现，适合改提示词后做对比实验。

用法：
    python eval/run_eval.py [--provider local|tavily] [--limit N] [--no-reflect] [--tag 名称]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from agent import config
from agent.pipeline import ResearchPipeline
from eval.faithfulness import check_report
from eval.metrics import aggregate, coverage

TASKS_PATH = Path(__file__).resolve().parent / "tasks.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_tasks(limit: int | None = None) -> list[dict]:
    tasks = [json.loads(line) for line in TASKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tasks[:limit] if limit else tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval")
    parser.add_argument("--provider", choices=["local", "tavily", "bocha"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个任务")
    parser.add_argument("--no-reflect", action="store_true")
    parser.add_argument("--react-mode", choices=["native", "prompt"], default=None)
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--no-faith", action="store_true", help="跳过引用忠实度校验（省 10~20 次 LLM 调用）")
    parser.add_argument("--tag", default=None, help="实验标签，用于区分对比实验（如 baseline）")
    args = parser.parse_args(argv)

    if args.no_reflect:
        config.MAX_REFLECT_ROUNDS = 0
    if args.react_mode:
        config.REACT_MODE = args.react_mode
    if args.naive:
        config.NAIVE = True

    console = Console()
    tasks = load_tasks(args.limit)
    results = []
    for i, task in enumerate(tasks, 1):
        console.print(f"[cyan][{i}/{len(tasks)}][/cyan] {task['question']}")
        try:
            pipeline = ResearchPipeline(args.provider, on_step=lambda e, d: None)
            run = pipeline.run(task["question"])
            row = {
                "id": task["id"], "ok": True,
                "coverage": coverage(run.report, task["must_cover"]),
                "stats": run.stats, "report_path": run.report_path,
            }
            if not args.no_faith:
                # 校验用的 LLM 调用发生在 stats 快照之后，不计入该任务的统计
                row["faith"] = check_report(pipeline.llm, run.report, task["question"], run.sources)
        except Exception as e:
            console.print(f"  [red]失败：{e}[/red]")
            row = {"id": task["id"], "ok": False, "error": str(e)[:200],
                   "coverage": {"rate": 0.0, "hits": [], "missed": task["must_cover"]},
                   "stats": {}}
        mark = "✓" if row["ok"] else "✗"
        console.print(f"  {mark} 覆盖率 {row['coverage']['rate']*100:.0f}%"
                      f"（命中 {len(row['coverage']['hits'])}/{len(task['must_cover'])}）")
        faith = row.get("faith")
        if faith:
            score = faith["faithfulness"]
            line = f"  引用忠实度 {score*100:.0f}%（支持 {faith['supported']}/{faith['judged']}" if score is not None \
                else f"  引用忠实度 无可判定句（引用句 {faith['cited_sentences']} 条"
            console.print(line + f"，数字存疑 {faith['num_flagged']} 句）"
                          + f"  引用密度 {faith['citation_density']*100:.0f}%"
                          f"（{faith['cited_sentences']}/{faith['sentences']} 句带引用）")
        results.append(row)

    summary = aggregate(results)
    if args.tag:
        summary["tag"] = args.tag
    else:
        tag_parts = [args.provider or config.SEARCH_PROVIDER]
        if args.no_reflect:
            tag_parts.append("no-reflect")
        if args.react_mode == "prompt":
            tag_parts.append("prompt-react")
        if args.naive:
            tag_parts.append("naive")
        summary["tag"] = "+".join(tag_parts)
    summary["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{summary['tag']}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    table = Table(title=f"评估汇总（{summary['tag']}）", show_header=True, header_style="bold")
    for col in ["任务", "成功", "覆盖率", "JSON首试", "工具有效", "忠实度", "LLM调用", "耗时s"]:
        table.add_column(col, justify="right")

    def _faith_cell(r: dict) -> str:
        score = r.get("faith", {}).get("faithfulness")
        return f"{score*100:.0f}%" if score is not None else "-"

    for r in results:
        s = r.get("stats", {})
        table.add_row(
            r["id"], "✓" if r["ok"] else "✗",
            f"{r['coverage']['rate']*100:.0f}%",
            f"{s.get('json_first_try_rate', 0)*100:.0f}%" if s else "-",
            f"{s.get('tool_valid_rate', 0)*100:.0f}%" if s else "-",
            _faith_cell(r),
            str(s.get('llm_calls', "-")) if s else "-",
            str(s.get('total_seconds', "-")) if s else "-",
        )
    table.add_row("— 平均 —", f"{summary['success']}/{summary['tasks']}",
                  f"{summary['coverage_avg']*100:.0f}%",
                  f"{summary['json_first_try_rate_avg']*100:.0f}%",
                  f"{summary['tool_valid_rate_avg']*100:.0f}%",
                  f"{summary['faithfulness_avg']*100:.0f}%" if summary.get("faithfulness_avg") is not None else "-",
                  str(summary["llm_calls_avg"]), str(summary["seconds_avg"]))
    console.print(table)
    console.print(f"[dim]明细已写入：{out}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
