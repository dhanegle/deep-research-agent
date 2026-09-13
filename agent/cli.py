"""命令行入口。

用法：
    python -m agent "2025 年国产新能源汽车出口情况" --provider local
"""
import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config
from .pipeline import ResearchPipeline, RunResult


def _on_step(console: Console):
    def emit(event: str, data: dict) -> None:
        if event == "start":
            console.print(Panel(f"[bold]{data['question']}[/bold]", title="🔍 深度调研 Agent", border_style="cyan"))
        elif event == "plan":
            outline = "\n".join(f"  {i}. {s}" for i, s in enumerate(data["outline"], 1))
            console.print(f"[cyan]📋 规划完成[/cyan]\n{outline}\n[dim]初始搜索词：{'、'.join(data['queries'])}[/dim]")
        elif event == "search":
            console.print(f"[yellow]🔍 搜索[/yellow] {data['query']}  →  {data['results']} 条结果")
        elif event == "read":
            console.print(f"[green]📄 收录[{data['source_id']}][/green] {data['title']}")
        elif event == "research":
            console.print(f"[blue]⚙️  {data['status']}[/blue]")
        elif event == "reflect":
            state = "资料充分" if data["sufficient"] else f"存在缺口 → 补搜 {data['gap_queries']}"
            console.print(f"[magenta]🤔 反思(第{data['round']}轮)[/magenta] {state}")
        elif event == "write":
            # 小节列表是可选字段：写作阶段还会推"跨节去重"这类无小节的状态事件
            sections = data.get("sections")
            console.print(f"[blue]✍️  {data['status']}[/blue]"
                          + (f" 小节：{'、'.join(sections)}" if sections else ""))
        elif event == "finish":
            console.print(f"[bold green]✅ 报告已生成：{data['report_path']}[/bold green]（来源 {data['sources']} 条）")
    return emit


def _print_stats(console: Console, result: RunResult) -> None:
    s = result.stats
    table = Table(title="运行指标", show_header=True, header_style="bold")
    table.add_column("指标", style="dim")
    table.add_column("值", justify="right")
    table.add_row("LLM 调用 / 纯推理耗时", f"{s['llm_calls']} 次 / {s['llm_seconds']:.0f}s")
    table.add_row("结构化输出首试成功率", f"{s['json_first_try_rate']*100:.0f}%（{s['json_first_try_ok']}/{s['json_calls']}）")
    table.add_row("工具调用有效率", f"{s['tool_valid_rate']*100:.0f}%（非法 {s['tool_invalid']}/{s['tool_calls']}）")
    table.add_row("JSON-ReAct 降级步数", str(s["react_fallbacks"]))
    table.add_row("搜索 / 阅读 / 摘要调用", f"{s['searches']} / {s['pages_fetched']} / {s['digest_calls']}")
    table.add_row("反思轮次", str(s["reflect_rounds"]))
    table.add_row("总耗时", f"{s['total_seconds']:.0f}s")
    console.print(table)
    console.print(f"[dim]全链路 trace: {result.trace_path}[/dim]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent", description="基于 Ollama + Qwen2.5-3B 的深度调研 Agent")
    parser.add_argument("question", help="调研问题")
    parser.add_argument("--provider", choices=["local", "tavily", "bocha"], default=None,
                        help="搜索提供方（默认取 .env 的 SEARCH_PROVIDER）")
    parser.add_argument("--max-steps", type=int, default=None, help="ReAct 单轮最大步数")
    parser.add_argument("--no-reflect", action="store_true", help="跳过反思补搜阶段")
    parser.add_argument("--react-mode", choices=["native", "prompt"], default=None,
                        help="工具调用路径：native=原生 tool calling | prompt=提示词 JSON-ReAct（消融实验用）")
    parser.add_argument("--naive", action="store_true",
                        help="朴素基线：最简提示词 + 关闭行为约束（消融实验用）")
    args = parser.parse_args(argv)

    if args.max_steps:
        config.MAX_STEPS = args.max_steps
    if args.no_reflect:
        config.MAX_REFLECT_ROUNDS = 0
    if args.react_mode:
        config.REACT_MODE = args.react_mode
    if args.naive:
        config.NAIVE = True

    console = Console()
    try:
        pipeline = ResearchPipeline(args.provider, on_step=_on_step(console))
        result = pipeline.run(args.question)
    except KeyboardInterrupt:
        console.print("[red]已中断[/red]")
        return 130
    except Exception as e:
        console.print(f"[bold red]运行失败：{e}[/bold red]")
        return 1

    _print_stats(console, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
