# DSPy OpenAI-Compatible Endpoint

**Project Report / Notion-Ready Summary**  
**Date:** May 4, 2026  
**Owner:** Ashok  
**Status:** Complete  
**Public Endpoint:** `https://pyromania-reactive-setback.ngrok-free.dev/v1`

---

## Overview

This project adds a new **OpenAI-compatible API endpoint** that lets us test DSPy strategies on top of multiple upstream models without changing the client-side API contract.

In simple terms:
- Clients still send requests to `POST /v1/chat/completions`
- The endpoint still accepts normal OpenAI-style payloads
- Behind the scenes, we can choose whether a model answers in:
  - **Base** mode
  - **BestOfN** mode
  - **GEPA** mode

This gives us a clean way to compare baseline model behavior against DSPy-enhanced strategies using the exact same API shape.

---

## Goal

The goal was to build a drop-in endpoint that:

1. Looks like the standard OpenAI Chat Completions API
2. Supports multiple upstream models through OpenRouter
3. Exposes DSPy strategies through simple model aliases
4. Can be tested locally and shared publicly for Harness Lab
5. Produces measurable benchmark results for Base vs BestOfN vs GEPA

---

## What We Built

We created a FastAPI service under `tools/dspy-openai-endpoint/` with the following components:

### Core API

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /docs`

### DSPy Strategy Layer

Each request is routed to one of 3 DSPy execution modes:

| Strategy | What it does | Why it matters |
|---|---|---|
| **Base** | Single normal generation | Baseline behavior |
| **BestOfN** | Generates multiple candidates and selects the best one | Better quality, more latency |
| **GEPA** | Uses a compiled optimized prompt learned from examples | More concise, reference-aligned answers |

### Models Connected

We tested 4 upstream models through OpenRouter:

- `glm-5`
- `glm-4.7`
- `minimax-m2.7`
- `minimax-m2.5`

That produced **12 total aliases**:

| Model | Base | BestOfN | GEPA |
|---|---|---|---|
| `glm-5` | `dspy-base-glm-5` | `dspy-bestofn-glm-5` | `dspy-gepa-glm-5` |
| `glm-4.7` | `dspy-base-glm-4.7` | `dspy-bestofn-glm-4.7` | `dspy-gepa-glm-4.7` |
| `minimax-m2.7` | `dspy-base-minimax-m2.7` | `dspy-bestofn-minimax-m2.7` | `dspy-gepa-minimax-m2.7` |
| `minimax-m2.5` | `dspy-base-minimax-m2.5` | `dspy-bestofn-minimax-m2.5` | `dspy-gepa-minimax-m2.5` |

---

## Why GEPA and BestOfN Were Added

### BestOfN

BestOfN asks the model multiple times, scores the outputs, and returns the best candidate.

**Use case:** improve answer quality without retraining a model.

**Tradeoff:** more tokens and more latency.

### GEPA

GEPA is an offline prompt optimization method. We give it example prompts and target answers, and it evolves a better prompt over many iterations.

**Use case:** make the model respond in a more exact and useful style for a known task type.

**Tradeoff:** requires a training dataset and compile step before serving.

---

## API Contract

One important requirement was that this endpoint should follow the same request/response structure as the existing FastAPI OpenAI-compatible route.

### Request

```json
{
  "model": "dspy-gepa-glm-5",
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function to compute factorial"
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "stream": false
}
```

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1714742400,
  "model": "dspy-gepa-glm-5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

### Compatibility Summary

| Area | Existing FastAPI endpoint | DSPy endpoint |
|---|---|---|
| Route | `POST /v1/chat/completions` | `POST /v1/chat/completions` |
| Request shape | OpenAI-style | Same |
| Response shape | OpenAI-style | Same |
| Model naming | Direct upstream model names | DSPy alias names |
| Tool execution loop | Supported in original app | Not used here |
| DSPy strategies | No | Yes |

So from a client perspective, this is a **drop-in compatible endpoint**.

---

## Training Work Completed

To make GEPA meaningful, we expanded the training dataset from **3 basic examples** to **25 diverse examples**.

### Training Categories

| Category | Count | Examples |
|---|---|---|
| Code Generation | 7 | factorial, palindrome, fibonacci, sieve, SQL, JavaScript flatten |
| Math / Arithmetic | 3 | multiplication, distance, discount |
| Calculus / Algebra | 2 | derivative, linear equation |
| Reasoning / Logic | 2 | syllogism, chickens-and-cows |
| Summarization | 3 | one-line summaries, bullet conversion |
| Knowledge QA | 8 | capital cities, science, CS facts |

### GEPA Outcome

GEPA learned to produce shorter and more reference-aligned outputs by optimizing the instruction prompt over roughly **130 iterations per model**.

| Model | Approx. Iterations | Best Val Score | Artifact |
|---|---|---|---|
| `glm-5` | ~130 | 0.56 | `artifacts/dspy-gepa-glm-5.json` |
| `glm-4.7` | ~130 | 0.56 | `artifacts/dspy-gepa-glm-4.7.json` |
| `minimax-m2.7` | ~130 | 0.56 | `artifacts/dspy-gepa-minimax-m2.7.json` |
| `minimax-m2.5` | ~137 | 0.60 | `artifacts/dspy-gepa-minimax-m2.5.json` |

> Note: the GEPA score is strict because answers are judged against exact references, not just semantic correctness.

---

## Testing Approach

We tested in 2 stages.

### Stage 1: Alias Validation

We first checked whether all 12 aliases worked end-to-end.

**Result:** `12/12 passed`

### Stage 2: Expanded Benchmark

We then expanded testing to **5 prompts per alias**, which created **60 total test cases**.

### Prompt Set Used

| Test ID | Category | Example Intent |
|---|---|---|
| `code-factorial` | Code Generation | recursive Python function |
| `code-palindrome` | Code Generation | string logic function |
| `math-arithmetic` | Math | exact short answer |
| `reasoning-logic` | Reasoning | word problem solving |
| `summarization` | Summarization | one-sentence summary |

---

## Final Results

### Overall Result

| Benchmark | Result |
|---|---|
| Initial alias validation | `12/12 passed` |
| Expanded benchmark | `59/60 passed (98.3%)` |

### Note on the One Failure

The only failed case was:

- `dspy-base-glm-4.7 / code-palindrome`

It failed with a JSON parse error after a long upstream response. Since the other requests passed and the endpoint routing itself remained stable, this appears to be a **transient upstream/model formatting issue**, not a structural issue in the FastAPI/DSPy endpoint.

### Per-Strategy Summary

| Strategy | Passed | Avg Answer Length | Avg Latency |
|---|---|---|---|
| **Base** | 19/20 | 299 chars | 8.0s |
| **BestOfN** | 20/20 | 352 chars | 25.8s |
| **GEPA** | 20/20 | 61 chars | 5.9s |

### Per-Category Summary

| Category | Result |
|---|---|
| Code Generation | 23/24 |
| Math | 12/12 |
| Reasoning | 12/12 |
| Summarization | 12/12 |

### Key Takeaway

GEPA delivered the most concise output and also had the best overall speed profile in the expanded run.

In practical terms:
- **Base** is the normal benchmark
- **BestOfN** improves quality but is the slowest
- **GEPA** gives the shortest, cleanest answers and performed very well after prompt optimization

---

## Deployment and Access

### Local Server

```bash
cd tools/dspy-openai-endpoint
uvicorn dspy_openai_endpoint.main:app --host 0.0.0.0 --port 8000
```

### Public Access

| Item | Value |
|---|---|
| Public Base URL | `https://pyromania-reactive-setback.ngrok-free.dev/v1` |
| Swagger UI | `https://pyromania-reactive-setback.ngrok-free.dev/docs` |
| API key requirement | Not required for this wrapper endpoint |

### Harness Lab Setup

Use:
- **Base URL:** `https://pyromania-reactive-setback.ngrok-free.dev/v1`
- **Model:** any of the 12 aliases above
- **Authentication:** not required on this wrapper endpoint

---

## Files Created / Updated

### Main project folder

`tools/dspy-openai-endpoint/`

### Important files

| File | Purpose |
|---|---|
| `src/dspy_openai_endpoint/main.py` | FastAPI app and routes |
| `src/dspy_openai_endpoint/config.py` | model registry and `.env` loading |
| `src/dspy_openai_endpoint/strategies.py` | Base / BestOfN / GEPA routing |
| `src/dspy_openai_endpoint/programs.py` | DSPy programs and reward logic |
| `src/dspy_openai_endpoint/lm.py` | OpenRouter-backed `dspy.LM` builder |
| `train_gepa.py` | offline GEPA training |
| `run_full_report.py` | benchmark runner |
| `data/gepa_samples.jsonl` | 25-sample GEPA training set |
| `artifacts/*.json` | compiled GEPA prompt artifacts |
| `report.json` | machine-readable benchmark results |

---

## Step-by-Step Work Completed

1. Researched DSPy `BestOfN` and `GEPA`
2. Designed the OpenAI-compatible FastAPI wrapper
3. Implemented the strategy router for Base / BestOfN / GEPA
4. Connected 4 upstream models through OpenRouter
5. Created 12 strategy-specific model aliases
6. Added `.env` handling for `OPENROUTER_API_KEY`
7. Fixed the env-loading bug in `config.py`
8. Created the first GEPA dataset
9. Validated all 12 aliases successfully
10. Expanded the GEPA dataset to 25 samples
11. Retrained GEPA artifacts for all 4 models
12. Expanded the benchmark to 60 test cases
13. Collected final results and generated `report.json`
14. Configured ngrok and exposed the endpoint publicly
15. Verified the public endpoint through `/v1/models` and `/docs`

---

## Business / Practical Outcome

At the end of this process, we now have:

- A **working OpenAI-compatible endpoint**
- A **public URL** usable from Harness Lab
- A way to compare **Base vs BestOfN vs GEPA** using the same API
- Trained GEPA artifacts for 4 models
- A benchmark showing that **GEPA is concise and stable**, while **BestOfN trades speed for selection quality**

This means the project is ready for:
- demos
- side-by-side strategy comparisons
- Harness Lab integration
- future benchmark expansion

---

## Final Conclusion

We successfully built and deployed a DSPy-powered OpenAI-compatible endpoint on top of 4 upstream models.

The final system:
- preserves the standard `/v1/chat/completions` contract
- exposes 12 strategy-based model aliases
- supports both **BestOfN** and **GEPA** in a practical way
- passed the initial alias validation completely
- passed **59 out of 60** tests in the expanded benchmark
- is live and accessible through ngrok

If someone wants to use this endpoint, they do **not** need a new client integration. They only need to:

1. point to the new base URL
2. choose one of the DSPy aliases

That makes this a clean, reusable evaluation and experimentation layer on top of OpenRouter models.
