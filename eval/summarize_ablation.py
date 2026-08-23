"""汇总消融实验结果：把 eval/results/ 下的 <config>-r<N>.json 按配置聚合取均值。

用法：python eval/summarize_ablation.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

RESULTS = Path(__file__).resolve().parent / "results"
CONFIG_ORDER = ["full", "no-reflect", "prompt-react", "naive"]


def main() -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in RESULTS.glob("*-r[0-9].json"):
        m = re.match(r"(?:\d{8}-\d{6}-)?(.+)-r\d+$", f.stem)
        if not m:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        s = data["summary"]
        s["_tasks_ok"] = sum(1 for r in data["results"] if r.get("ok"))
        s["_runs_tasks"] = len(data["results"])
        groups[m.group(1)].append(s)

    table = Table(title="消融实验（local 语料 2 任务 × N 次重复，均值）", show_header=True, header_style="bold")
    for col in ["配置", "重复", "成功率", "覆盖率", "JSON首试", "工具有效", "LLM调用", "耗时s"]:
        table.add_column(col, justify="right")

    for cfg in CONFIG_ORDER:
        if cfg not in groups:
            continue
        runs = groups[cfg]
        n = len(runs)
        task_total = sum(r["_runs_tasks"] for r in runs)
        ok_total = sum(r["_tasks_ok"] for r in runs)

        def avg(key: str) -> float:
            return sum(r.get(key, 0) for r in runs) / n

        table.add_row(
            cfg, str(n),
            f"{ok_total}/{task_total}",
            f"{avg('coverage_avg')*100:.0f}%",
            f"{avg('json_first_try_rate_avg')*100:.0f}%",
            f"{avg('tool_valid_rate_avg')*100:.0f}%",
            f"{avg('llm_calls_avg'):.1f}",
            f"{avg('seconds_avg'):.0f}",
        )

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
