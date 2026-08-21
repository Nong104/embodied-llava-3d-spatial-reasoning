"""
Sanity-check the metrics module against hand-constructed prediction/reference
pairs, decoupled from the model. This verifies the metric computation logic
itself is correct before it is wired up to real model generations.

Run: python -m scripts.test_compute_metrics
"""

from src.evaluation.metrics import (
    compute_em,
    compute_bleu4,
    compute_cider,
    classify_question,
    compute_metrics,
)


def main():
    # --- 1. Unit-level sanity checks -------------------------------------
    print("=== EM ===")
    print("exact match:      ", compute_em("nightstand", ["nightstand"]))       # expect 1.0
    print("case/punct diff:  ", compute_em("Night-stand.", ["nightstand"]))     # expect 1.0
    print("wrong answer:     ", compute_em("lamp", ["nightstand"]))             # expect 0.0
    print("multi-ref match:  ", compute_em("table", ["chair", "table"]))        # expect 1.0

    print("\n=== BLEU-4 ===")
    print("identical:        ", round(compute_bleu4("a chair and a table", ["a chair and a table"]), 4))  # ~1.0
    print("partial overlap:  ", round(compute_bleu4("a chair", ["a chair and a table"]), 4))               # low but > 0
    print("no overlap:       ", round(compute_bleu4("completely unrelated words", ["a chair and a table"]), 4))  # ~0

    print("\n=== CIDEr (batch) ===")
    preds = ["nightstand", "table", "a red chair"]
    refs = [["nightstand"], ["table"], ["a blue chair"]]
    cider_scores = compute_cider(preds, refs)
    for p, r, s in zip(preds, refs, cider_scores):
        print(f"pred={p!r:20} ref={r!r:20} CIDEr={s:.3f}")

    print("\n=== Question classification ===")
    test_questions = [
        "What is next to the bed?",
        "How many chairs are in this room?",
        "Is there a lamp on the table?",
        "What color is the sofa?",
    ]
    for q in test_questions:
        print(f"{q!r:45} -> {classify_question(q)}")

    # --- 2. End-to-end on the toy ScanQA questions -------------------------
    print("\n=== compute_metrics() on toy ScanQA-style batch ===")
    questions = [
        "What is next to the bed?",
        "What object is in front of the sofa?",
    ]
    references_list = [["nightstand"], ["table"]]

    # Simulate an untrained model: one correct guess, one wrong guess.
    predictions = ["nightstand", "chair"]

    results = compute_metrics(predictions, references_list, questions)
    for key, value in results.items():
        print(f"{key:30}: {value}")

    print("\ncompute_metrics smoke test passed.")


if __name__ == "__main__":
    main()