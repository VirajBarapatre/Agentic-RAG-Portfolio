"""Compare two eval runs: per-group changes and which questions flipped.

Run from the project root:
    python -m src.eval.compare results/run_<before>.csv results/run_<after>.csv
"""

import sys
from pathlib import Path

from src.config import RESULTS_DIR
from src.eval.summarize import group_summaries, load_rows, plt  # plt is already set up to draw to files


def share(flags: list[bool]) -> float:
    """Fraction of True values (0.0 for an empty list)."""
    return sum(flags) / len(flags) if flags else 0.0


def change_line(name: str, before: dict, after: dict) -> str:
    """One markdown row: before -> after for each metric. Rate changes are in percentage points."""
    def rate(key: str) -> str:
        b, a = share(before[key]), share(after[key])
        return f"{b:.0%} -> {a:.0%} ({(a - b) * 100:+.1f} pp)"

    def avg(key: str, digits: int) -> str:
        return f"{before[key]:.{digits}f} -> {after[key]:.{digits}f}"

    return (
        f"| {name} | {before['n']} | {rate('answer')} | {rate('tool')} "
        f"| {avg('calls', 1)} | {avg('latency', 1)} | {avg('tokens', 0)} |"
    )


def flipped(rows_before: list[dict], rows_after: list[dict]) -> list[str]:
    """Questions whose answer_ok changed between the runs."""
    after_by_id = {r["id"]: r for r in rows_after}
    lines = []
    for b in rows_before:
        a = after_by_id.get(b["id"])
        if a is None or a["answer_ok"] == b["answer_ok"]:
            continue
        change = "FIXED" if a["answer_ok"] == "True" else "BROKE"
        lines.append(
            f"- {change} {b['id']} ({b['category']}): tools "
            f"{b['tools_used'] or 'none'} -> {a['tools_used'] or 'none'}"
        )
    return lines


def save_chart(before: dict, after: dict, title: str, path: Path) -> None:
    """Bar chart of answer correctness per group, before vs after."""
    names = list(after)
    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], [100 * share(before[n]["answer"]) for n in names], width, label="Before")
    ax.bar([i + width / 2 for i in x], [100 * share(after[n]["answer"]) for n in names], width, label="After")
    ax.set_xticks(list(x), [f"{n}\n(n={after[n]['n']})" for n in names])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Answer correct (% of questions)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m src.eval.compare <before.csv> <after.csv>")
    before_file, after_file = Path(sys.argv[1]), Path(sys.argv[2])
    rows_before, rows_after = load_rows(before_file), load_rows(after_file)
    before, after = group_summaries(rows_before), group_summaries(rows_after)

    lines = [
        f"# {before_file.stem} -> {after_file.stem}",
        "",
        "| Group | n | Answer correct | Tool selection | Avg tool calls | Avg latency (s) | Avg tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [change_line(name, before[name], after[name]) for name in after if name in before]
    lines += ["", "Questions that changed:"]
    lines += flipped(rows_before, rows_after) or ["- none"]
    errors = (before["all"]["errors"], after["all"]["errors"])
    lines += ["", f"Errors: {errors[0]} -> {errors[1]}"]
    report = "\n".join(lines)

    stem = f"compare_{before_file.stem}_vs_{after_file.stem}"
    (RESULTS_DIR / f"{stem}.md").write_text(report + "\n", encoding="utf-8")
    save_chart(before, after, f"{before_file.stem} -> {after_file.stem}", RESULTS_DIR / f"{stem}.png")
    print(report)
    print(f"\nSaved {stem}.md and {stem}.png")


if __name__ == "__main__":
    main()