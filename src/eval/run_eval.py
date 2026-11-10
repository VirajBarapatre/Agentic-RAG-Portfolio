"""Run every eval question through the agent, score it, and save a CSV.

Run from the project root:
    python -m src.eval.run_eval        # all questions
    python -m src.eval.run_eval 3      # only the first 3 (quick, cheap check)
"""

import csv
import json
import sys
import time
from datetime import datetime

from src.agent import build_agent, run_question
from src.config import GROQ_MODEL, QUESTIONS_FILE, RESULTS_DIR
from src.eval.scoring import score_answer, score_grounded, score_tools

PAUSE_SECONDS = 2  # wait between questions to stay under Groq's free-tier rate limits

FIELDS = [
    "id", "category", "question", "answer", "tools_used", "expected_tools",
    "tool_ok", "answer_ok", "grounded", "latency_s", "input_tokens",
    "output_tokens", "model", "error",
]


def load_questions() -> list[dict]:
    """Read eval/questions.jsonl: one JSON object per line."""
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_one(agent, question: dict, run_id: str) -> dict:
    """Run one question (after its setup turns, if any) and return a scored CSV row."""
    row = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected_tools": json.dumps(question["expected_tools"]),
        "model": GROQ_MODEL,
        "error": "",
    }
    thread_id = f"{run_id}-{question['id']}"  # fresh memory for every question
    evidence = [question["question"]]
    try:
        for setup_question in question.get("setup", []):
            setup = run_question(agent, setup_question, thread_id)
            evidence += [setup_question] + [str(a) for a in setup["tool_args"]] + setup["tool_outputs"]
        start = time.perf_counter()
        result = run_question(agent, question["question"], thread_id)
        latency = time.perf_counter() - start
    except Exception as err:  # record the failure and move on to the next question
        row.update(tool_ok=False, answer_ok=False, error=f"{type(err).__name__}: {err}")
        return row

    evidence += [str(a) for a in result["tool_args"]] + result["tool_outputs"]
    grounded = score_grounded(result["answer"], evidence)
    row.update(
        answer=result["answer"],
        tools_used="|".join(result["tools_used"]),
        tool_ok=score_tools(result["tools_used"], question["expected_tools"]),
        answer_ok=score_answer(question, result["answer"]),
        grounded="" if grounded is None else grounded,  # empty = no numbers to check
        latency_s=round(latency, 2),
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )
    return row


def main() -> None:
    questions = load_questions()
    if len(sys.argv) > 1:
        questions = questions[: int(sys.argv[1])]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"run_{run_id}.csv"
    agent = build_agent()

    # utf-8-sig adds a marker so Excel shows curly quotes and symbols correctly.
    with open(out_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, question in enumerate(questions, start=1):
            row = evaluate_one(agent, question, run_id)
            writer.writerow(row)
            f.flush()  # each row is saved immediately, so a crash keeps earlier results
            if row["error"]:
                status = f"ERROR {row['error'][:80]}"
            else:
                status = f"tools={'ok' if row['tool_ok'] else 'FAIL'}  answer={'ok' if row['answer_ok'] else 'FAIL'}"
            print(f"[{i}/{len(questions)}] {question['id']}: {status}")
            time.sleep(PAUSE_SECONDS)

    print(f"\nSaved {len(questions)} rows to {out_file}")


if __name__ == "__main__":
    main()  