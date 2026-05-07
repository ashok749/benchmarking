"""
Full report: test all 4 models × 3 strategies (base, best_of_n, gepa)
against the local DSPy OpenAI-compatible endpoint at http://127.0.0.1:8000.

GEPA aliases require artifacts; this script trains them first.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 300  # seconds per request

MODELS = ["glm-5", "glm-4.7", "minimax-m2.7", "minimax-m2.5"]
STRATEGIES = ["base", "bestofn", "gepa"]

TEST_CASES = [
    {
        "id": "code-factorial",
        "category": "Code Generation",
        "messages": [{"role": "user", "content": "Write a Python function that computes the factorial of a number recursively."}],
    },
    {
        "id": "code-palindrome",
        "category": "Code Generation",
        "messages": [{"role": "user", "content": "Write a Python function to check if a string is a palindrome."}],
    },
    {
        "id": "math-arithmetic",
        "category": "Math",
        "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    },
    {
        "id": "reasoning-logic",
        "category": "Reasoning",
        "messages": [{"role": "user", "content": "A farmer has chickens and cows. He counts 20 heads and 56 legs. How many chickens and cows does he have?"}],
    },
    {
        "id": "summarization",
        "category": "Summarization",
        "messages": [{"role": "user", "content": "Summarize in one sentence: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems improve their performance on tasks through experience."}],
    },
]

# ── helpers ──────────────────────────────────────────────────────────────
def alias(strategy: str, model: str) -> str:
    return f"dspy-{strategy}-{model}"

def call_chat(model_alias: str, messages: list[dict]) -> dict:
    payload = {"model": model_alias, "messages": messages, "stream": False, "max_tokens": 1024, "temperature": 0.7}
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    return {"status": r.status_code, "body": r.json(), "elapsed": r.elapsed.total_seconds()}

# ── train GEPA artifacts ─────────────────────────────────────────────────
def train_gepa_all():
    print("\n" + "=" * 70)
    print("PHASE 1: Training GEPA artifacts for all 4 models")
    print("=" * 70)
    venv_python = str(Path(__file__).resolve().parents[0].parent.parent / ".venv" / "bin" / "python")
    train_script = str(Path(__file__).resolve().parent / "train_gepa.py")
    for model in MODELS:
        a = alias("gepa", model)
        print(f"\n  Training {a} ...")
        t0 = time.time()
        rc = os.system(f'{venv_python} {train_script} --alias {a} --auto light 2>&1')
        elapsed = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {a}: {status}  ({elapsed:.1f}s)")

# ── test all combos ──────────────────────────────────────────────────────
def test_all() -> list[dict]:
    total_tests = len(MODELS) * len(STRATEGIES) * len(TEST_CASES)
    print("\n" + "=" * 70)
    print(f"PHASE 2: Testing {total_tests} combinations  ({len(MODELS)} models × {len(STRATEGIES)} strategies × {len(TEST_CASES)} prompts)")
    print("=" * 70)
    results = []
    test_num = 0
    for model in MODELS:
        for strategy in STRATEGIES:
            a = alias(strategy, model)
            for tc in TEST_CASES:
                test_num += 1
                label = f"[{test_num}/{total_tests}] {a} / {tc['id']}"
                print(f"\n  {label} ... ", end="", flush=True)
                t0 = time.time()
                try:
                    resp = call_chat(a, tc["messages"])
                    elapsed = time.time() - t0
                    ok = resp["status"] == 200
                    answer = ""
                    if ok:
                        answer = resp["body"]["choices"][0]["message"]["content"]
                    error = resp["body"].get("error", {}).get("message", "") if not ok else ""
                    results.append({
                        "model": model,
                        "strategy": strategy,
                        "alias": a,
                        "test_id": tc["id"],
                        "category": tc["category"],
                        "prompt": tc["messages"][0]["content"][:80],
                        "status": resp["status"],
                        "ok": ok,
                        "elapsed_s": round(elapsed, 2),
                        "answer_len": len(answer),
                        "answer_preview": (answer[:200] + "...") if len(answer) > 200 else answer,
                        "error": error,
                    })
                    print(f"{'PASS' if ok else 'FAIL'} ({elapsed:.1f}s, {len(answer)} chars)")
                except Exception as e:
                    elapsed = time.time() - t0
                    results.append({
                        "model": model,
                        "strategy": strategy,
                        "alias": a,
                        "test_id": tc["id"],
                        "category": tc["category"],
                        "prompt": tc["messages"][0]["content"][:80],
                        "status": 0,
                        "ok": False,
                        "elapsed_s": round(elapsed, 2),
                        "answer_len": 0,
                        "answer_preview": "",
                        "error": str(e),
                    })
                    print(f"ERROR ({elapsed:.1f}s): {e}")
    return results

# ── report ───────────────────────────────────────────────────────────────
def print_report(results: list[dict]):
    print("\n" + "=" * 70)
    print("REPORT: DSPy OpenAI Endpoint – All Models × Strategies × Prompts")
    print("=" * 70)
    print(f"\n{'#':>3} {'Alias':<30} {'Test':<20} {'Status':>6} {'Time':>7} {'Len':>5}  Preview")
    print("-" * 120)
    for i, r in enumerate(results, 1):
        preview = r["answer_preview"][:50].replace("\n", " ") if r["ok"] else r["error"][:50]
        print(f"{i:>3} {r['alias']:<30} {r['test_id']:<20} {r['status']:>6} {r['elapsed_s']:>6.1f}s {r['answer_len']:>5}  {preview}")

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")

    # Per-strategy summary
    print("\n  Per-strategy breakdown:")
    for strat in STRATEGIES:
        strat_results = [r for r in results if r["strategy"] == strat]
        sp = sum(1 for r in strat_results if r["ok"])
        avg_len = sum(r["answer_len"] for r in strat_results if r["ok"]) / max(sp, 1)
        avg_time = sum(r["elapsed_s"] for r in strat_results if r["ok"]) / max(sp, 1)
        print(f"    {strat:<10}: {sp}/{len(strat_results)} passed  |  avg length: {avg_len:.0f} chars  |  avg latency: {avg_time:.1f}s")

    # Per-category summary
    print("\n  Per-category breakdown:")
    categories = list(dict.fromkeys(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cp = sum(1 for r in cat_results if r["ok"])
        print(f"    {cat:<18}: {cp}/{len(cat_results)} passed")

    report_path = Path(__file__).resolve().parent / "report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Full report saved to {report_path}")

    print("\n  Endpoint URL for Harness Lab:")
    print("    Base URL: http://127.0.0.1:8000/v1")
    print("    (expose with: ngrok http 8000)")
    print("\n  Available model aliases:")
    for r in results:
        tag = "✓" if r["ok"] else "✗"
        print(f"    {tag} {r['alias']}")


if __name__ == "__main__":
    # Phase 1: Train GEPA
    train_gepa_all()
    # Phase 2: Test all
    results = test_all()
    # Phase 3: Report
    print_report(results)
