from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import httpx

from .config import ModelAlias, Settings
from .lm import build_lm
from .programs import SimpleChatProgram, build_prompt, build_gepa_feedback, reward_prediction
from .schemas import ChatCompletionRequest


@dataclass
class StrategyResult:
    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"


class StrategyRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, model_alias: ModelAlias, request: ChatCompletionRequest) -> StrategyResult:
        if request.tools:
            return self._run_tool_calling(model_alias, request)

        prompt = build_prompt(request.messages)
        temperature = request.temperature if request.temperature is not None else 0.7
        max_tokens = request.max_tokens if request.max_tokens is not None else 1024

        if model_alias.strategy == "base":
            return self._run_base(model_alias.upstream_model, prompt, temperature, max_tokens)
        if model_alias.strategy == "best_of_n":
            return self._run_best_of_n(model_alias.upstream_model, prompt, temperature, max_tokens)
        if model_alias.strategy == "gepa":
            return self._run_gepa(model_alias, prompt, temperature, max_tokens)
        raise ValueError(f"Unsupported strategy: {model_alias.strategy}")

    def _run_tool_calling(self, model_alias: ModelAlias, request: ChatCompletionRequest) -> StrategyResult:
        if model_alias.strategy == "base":
            return self._run_tool_calling_once(model_alias.upstream_model, request)
        if model_alias.strategy == "best_of_n":
            return self._run_tool_calling_best_of_n(model_alias.upstream_model, request)
        if model_alias.strategy == "gepa":
            return self._run_tool_calling_once(model_alias.upstream_model, request)
        raise ValueError(f"Unsupported strategy: {model_alias.strategy}")

    def _run_base(self, upstream_model: str, prompt: str, temperature: float, max_tokens: int) -> StrategyResult:
        program = SimpleChatProgram()
        lm = build_lm(self.settings, upstream_model, temperature, max_tokens)
        program.set_lm(lm)
        prediction = program(prompt)
        return StrategyResult(content=prediction.answer)

    def _run_best_of_n(self, upstream_model: str, prompt: str, temperature: float, max_tokens: int) -> StrategyResult:
        import dspy

        program = SimpleChatProgram()
        lm = build_lm(self.settings, upstream_model, temperature, max_tokens)
        program.set_lm(lm)

        best_of_n = dspy.BestOfN(
            module=program.module,
            N=self.settings.best_of_n_count,
            reward_fn=lambda args, pred: reward_prediction(args["prompt"], pred.answer),
            threshold=self.settings.best_of_n_threshold,
        )
        best_of_n.set_lm(lm)
        prediction = best_of_n(prompt=prompt)
        return StrategyResult(content=prediction.answer)

    def _run_gepa(self, model_alias: ModelAlias, prompt: str, temperature: float, max_tokens: int) -> StrategyResult:
        artifact_path = self.settings.artifact_dir / f"{model_alias.alias}.json"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"GEPA artifact not found for {model_alias.alias}. Train it first with train_gepa.py."
            )

        program = SimpleChatProgram()
        program.load(str(artifact_path))
        lm = build_lm(self.settings, model_alias.upstream_model, temperature, max_tokens)
        program.set_lm(lm)
        prediction = program(prompt)
        return StrategyResult(content=prediction.answer)

    def _run_tool_calling_best_of_n(self, upstream_model: str, request: ChatCompletionRequest) -> StrategyResult:
        best_result: StrategyResult | None = None
        best_score = -1.0
        tries = max(1, self.settings.best_of_n_count)
        for _ in range(tries):
            result = self._run_tool_calling_once(upstream_model, request)
            score = self._score_tool_result(result)
            if score > best_score:
                best_score = score
                best_result = result
        if best_result is None:
            raise ValueError("BestOfN tool-calling produced no result")
        return best_result

    def _score_tool_result(self, result: StrategyResult) -> float:
        if result.tool_calls:
            return 1.0 + len(result.tool_calls) * 0.1
        if result.content and result.content.strip():
            return 0.2
        return 0.0

    def _run_tool_calling_once(self, upstream_model: str, request: ChatCompletionRequest) -> StrategyResult:
        if not self.settings.upstream_api_key:
            raise ValueError("Missing upstream API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

        payload: dict[str, object] = {
            "model": upstream_model,
            "messages": [self._serialize_message(message) for message in request.messages],
            "stream": False,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        headers = {
            "Authorization": f"Bearer {self.settings.upstream_api_key}",
            "Content-Type": "application/json",
        }

        response = httpx.post(
            f"{self.settings.upstream_api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}

        tool_calls = message.get("tool_calls")
        content = self._normalize_content(message.get("content"))
        return StrategyResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop"),
        )

    def _serialize_message(self, message) -> dict[str, object]:
        if hasattr(message, "model_dump"):
            data = message.model_dump(exclude_none=True)
        else:
            data = {key: value for key, value in dict(message).items() if value is not None}
        return data

    def _normalize_content(self, content) -> str | None:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts) if parts else None
        if content is None:
            return None
        return json.dumps(content)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    import dspy

    expected = getattr(gold, "answer", "")
    actual = getattr(pred, "answer", "")
    expected_normalized = expected.strip().lower()
    actual_normalized = actual.strip().lower()
    score = 1.0 if expected_normalized and expected_normalized in actual_normalized else 0.0
    feedback = build_gepa_feedback(expected, actual)
    return dspy.Prediction(score=score, feedback=feedback)


def artifact_path(settings: Settings, alias: str) -> Path:
    return settings.artifact_dir / f"{alias}.json"
