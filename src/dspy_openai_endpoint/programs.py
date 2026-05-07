from __future__ import annotations

import re


def build_prompt(messages: list[dict[str, str]] | list[object]) -> str:
    normalized: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or message["role"]
        content = getattr(message, "content", None) or message["content"]
        normalized.append(f"[{role.upper()}]\n{content}")
    return "\n\n".join(normalized)


def looks_like_coding_prompt(prompt: str) -> bool:
    return bool(re.search(r"\b(code|python|javascript|typescript|bug|fix|function|class|test|compile)\b", prompt, re.IGNORECASE))


def reward_prediction(prompt: str, answer: str) -> float:
    score = 0.0
    stripped = answer.strip()
    if stripped:
        score += 0.35
    if len(stripped.split()) >= 20:
        score += 0.2
    if "I can't" in stripped or "I cannot" in stripped:
        score -= 0.4
    if looks_like_coding_prompt(prompt):
        if "```" in stripped:
            score += 0.25
        if re.search(r"\b(def|class|function|const|let|var|import|from)\b", stripped):
            score += 0.2
    else:
        if stripped.endswith((".", "!", "?")):
            score += 0.1
    return max(0.0, min(score, 1.0))


def build_gepa_feedback(expected: str, actual: str) -> str:
    return (
        "Expected the response to align more closely with the reference answer. "
        f"Reference: {expected[:500]}\n"
        f"Actual: {actual[:500]}\n"
        "Improve factual alignment, completeness, and formatting."
    )


class SimpleChatProgram:
    def __init__(self):
        import dspy

        self._dspy = dspy
        self.module = dspy.ChainOfThought("prompt -> answer")

    def set_lm(self, lm) -> None:
        self.module.set_lm(lm)

    def __call__(self, prompt: str):
        return self.module(prompt=prompt)

    def load(self, path: str) -> None:
        self.module.load(path)

    def save(self, path: str) -> None:
        self.module.save(path)
