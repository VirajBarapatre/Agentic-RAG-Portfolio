"""Summarise one eval run: a table per category and a bar chart.

Run from the project root:
    python -m src.eval.summarize                          # latest run in results/
    python -m src.eval.summarize results/run_<time>.csv   # a specific run
"""

import csv
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

from src.config import RESULTS_DIR  # noqa: E402

CATEGORIES = ["easy", "multi", "refusal", "adversarial"]


def load_rows(path: Path) -> list[dict]:
    """Read a run CSV. Note: every value comes back as text, e.g. 'True', '12.5'."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rate(flags: list[bool]) -> str:
    """Format a list of True/False as '50% (7/14)', so small groups stay visible."""
    if not flags:
        return "n/a"
    return f"{sum(flags) / len(flags):.0%} ({sum(flags)}/{len(flags)})"


def summarize(rows: list[dict]) -> dict:
    """Metrics for a group of rows. Errors count as failures; averages skip error rows."""
    ok_rows = [r for r in rows if not r["error"]]
    checked = [r for r in ok_rows if r["grounded"] != ""]  # answers that contained numbers
    return {
        "n": len(rows),
        "tool": [r["tool_ok"] == "True" for r in rows],
        "answer": [r["answer_ok"] == "True" for r in rows],
        "grounded": [r["grounded"] == "True" for r in checked],
        "calls": mean(len(r["tools_used"].split("|")) if r["tools_used"] else 0 for r in ok_rows) if ok_rows else 0.0,
        "latency": mean(float(r["latency_s"]) for r in ok_rows) if ok_rows else 0.0,
        "tokens": mean(int(r["input_tokens"]) + int(r["output_tokens"]) for r in ok_rows) if ok_rows else 0.0,
        "errors": len(rows) - len(ok_rows),
    }


def markdown_table(groups: dict[str, dict]) -> str:
    """One markdown table row per group."""
    lines = [
        "| Group | n | Tool selection | Answer correct | Grounded (numbers) | Avg tool calls | Avg latency (s) | Avg tokens | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in groups.items():
        lines.append(
            f"| {name} | {s['n']} | {rate(s['tool'])} | {rate(s['answer'])} | {rate(s['grounded'])} "
            f"| {s['calls']:.1f} | {s['latency']:.1f} | {s['tokens']:.0f} | {s['errors']} |"
        )
    return "\n".join(lines)


def save_chart(groups: dict[str, dict], title: str, path: Path) -> None:
    """Grouped bar chart: tool selection and answer correctness per group."""
    names = list(groups)
    tool = [100 * sum(g["tool"]) / g["n"] for g in groups.values()]
    answer = [100 * sum(g["answer"]) / g["n"] for g in groups.values()]
    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], tool, width, label="Tool selection")
    ax.bar([i + width / 2 for i in x], answer, width, label="Answer correct")
    ax.set_xticks(list(x), [f"{n}\n(n={g['n']})" for n, g in groups.items()])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of questions")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) > 1:
        run_file = Path(sys.argv[1])
    else:
        run_file = max(RESULTS_DIR.glob("run_*.csv"))  # timestamps sort, so max = newest
    rows = load_rows(run_file)

    groups = {c: summarize([r for r in rows if r["category"] == c]) for c in CATEGORIES}
    groups = {name: s for name, s in groups.items() if s["n"] > 0}
    groups["all"] = summarize(rows)

    table = markdown_table(groups)
    model = rows[0]["model"] if rows else "unknown model"
    title = f"{run_file.stem} ({model})"
    run_file.with_suffix(".md").write_text(f"# {title}\n\n{table}\n", encoding="utf-8")
    save_chart(groups, title, run_file.with_suffix(".png"))

    print(table)
    print(f"\nSaved {run_file.with_suffix('.md').name} and {run_file.with_suffix('.png').name}")


if __name__ == "__main__":
    main()