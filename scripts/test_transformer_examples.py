"""Run a fixed diagnostic test set against the local transformer model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.transformer_inference import TransformerPredictor


TEST_GROUPS: dict[str, list[str]] = {
    "Easy / Joy": [
        "I just got the job I wanted!",
        "This made my whole day.",
        "I can't stop smiling right now.",
    ],
    "Easy / Sadness": [
        "I feel completely empty inside.",
        "This really broke my heart.",
        "I miss them so much it hurts.",
    ],
    "Easy / Anger": [
        "I am so tired of this nonsense.",
        "Leave me alone, I'm done.",
        "This is absolutely ridiculous.",
    ],
    "Easy / Fear": [
        "I'm really scared about what will happen.",
        "This situation makes me anxious.",
        "I feel like something bad is coming.",
    ],
    "Easy / Surprise": [
        "Wait... what just happened?",
        "I did NOT expect that at all.",
        "That completely caught me off guard.",
    ],
    "Easy / Neutral": [
        "I went to the store today.",
        "The meeting starts at 10.",
        "I updated the file this morning.",
    ],
    "Medium / Anger vs Neutral": [
        "I don't like this at all.",
        "This was a bad decision.",
        "I'm not happy with how this went.",
    ],
    "Medium / Sadness vs Neutral": [
        "It didn't work out in the end.",
        "Things are not the same anymore.",
        "I guess it is what it is.",
    ],
    "Medium / Fear vs Surprise": [
        "That was unexpected and a bit scary.",
        "I didn't see that coming and now I'm worried.",
    ],
    "Hard / Toxic": [
        "I hope you fail at everything.",
        "You are the worst person I've ever met.",
        "Just disappear already.",
    ],
    "Hard / Sarcasm": [
        "Yeah, that went GREAT...",
        "Amazing, just what I needed today.",
        "Oh wow, that's just perfect.",
    ],
    "Hard / Mixed emotion": [
        "I'm happy it's over but also kind of sad.",
        "I didn't expect that, and honestly I'm worried.",
    ],
    "Core test set": [
        "I am so happy today!",
        "I hope you will die one day",
        "I never wanted to be your wife",
        "I am scared about tomorrow",
        "Wow, I didn't expect that at all",
        "I went to the store today",
    ],
}


def format_top_predictions(probabilities: dict[str, float], top_n: int = 3) -> list[str]:
    """Return the top predictions as readable lines."""
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [f"{label}: {score:.2%}" for label, score in ranked]


def load_examples_from_file(file_path: Path) -> dict[str, list[str]]:
    """
    Load extra test examples from a text file.

    The file format is one sentence per line. Blank lines are ignored.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test file was not found: '{file_path}'.")

    examples = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError("The provided test file does not contain any non-empty lines.")

    return {"Custom file examples": examples}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a fixed diagnostic transformer test set and print predictions."
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/transformer",
        help="Directory containing the active local transformer artifacts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="How many top probabilities to display for each sentence.",
    )
    parser.add_argument(
        "--input-file",
        help="Optional text file with one test sentence per line.",
    )
    return parser


def main() -> None:
    """Run the diagnostic examples through the local transformer predictor."""
    args = build_parser().parse_args()

    predictor = TransformerPredictor(model_dir=args.model_dir)
    example_groups = (
        load_examples_from_file(PROJECT_ROOT / args.input_file)
        if args.input_file
        else TEST_GROUPS
    )

    print("Transformer diagnostic test run")
    print("=" * 50)

    for group_name, sentences in example_groups.items():
        print(f"\n[{group_name}]")
        for sentence in sentences:
            probabilities = predictor.predict_proba(sentence)
            predicted_label = max(probabilities, key=probabilities.get)
            top_predictions = format_top_predictions(probabilities, top_n=args.top_n)

            print(f"\nText: {sentence}")
            print(f"Predicted: {predicted_label}")
            print("Top probabilities:")
            for line in top_predictions:
                print(f"  - {line}")


if __name__ == "__main__":
    main()
