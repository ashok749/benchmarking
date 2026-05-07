from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from dspy_openai_endpoint.config import Settings
from dspy_openai_endpoint.lm import build_lm
from dspy_openai_endpoint.programs import SimpleChatProgram
from dspy_openai_endpoint.strategies import artifact_path, gepa_metric


def load_examples(dataset_path: Path):
    import dspy

    examples = []
    for line in dataset_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(dspy.Example(prompt=row["prompt"], answer=row["answer"]).with_inputs("prompt"))
    if not examples:
        raise ValueError(f"No examples found in {dataset_path}")
    return examples


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compile a GEPA-optimized DSPy program for a model alias.")
    parser.add_argument("--alias", required=True, help="Model alias to compile, e.g. dspy-gepa-glm-5")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data" / "gepa_samples.jsonl"),
        help="Path to JSONL dataset with prompt/answer fields",
    )
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument(
        "--reflection-model",
        help="Optional override for the reflection LM upstream model. Defaults to the alias upstream model.",
    )
    args = parser.parse_args()

    settings = Settings.load()
    alias = settings.model_aliases.get(args.alias)
    if alias is None:
        raise ValueError(f"Unknown alias: {args.alias}")
    if alias.strategy != "gepa":
        raise ValueError(f"Alias {args.alias} is not a GEPA alias.")

    dataset_path = Path(args.dataset)
    trainset = load_examples(dataset_path)
    valset = trainset

    import dspy

    student = SimpleChatProgram()
    student_lm = build_lm(settings, alias.upstream_model, temperature=0.7, max_tokens=1024)
    student.set_lm(student_lm)

    reflection_target = args.reflection_model or alias.upstream_model
    reflection_lm = build_lm(settings, reflection_target, temperature=1.0, max_tokens=2048)

    optimizer = dspy.GEPA(
        metric=gepa_metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        track_stats=True,
        log_dir=str(settings.artifact_dir / f"{args.alias}-logs"),
    )

    optimized = optimizer.compile(student=student.module, trainset=trainset, valset=valset)
    destination = artifact_path(settings, args.alias)
    destination.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(destination))
    print(f"Saved GEPA artifact to {destination}")


if __name__ == "__main__":
    main()
