"""Print the saved evaluation metrics for the baseline model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.config import load_config
from emotion_detector.utils.io import load_json


def main() -> None:
    """Load metrics JSON and print a readable summary."""
    config = load_config(PROJECT_ROOT / "configs" / "baseline_config.json")
    metrics_path = PROJECT_ROOT / config.metrics_output_path

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. Run training first."
        )

    metrics = load_json(metrics_path)

    print("Baseline evaluation summary")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_f1']:.4f}")
    if "validation" in metrics:
        validation_metrics = metrics["validation"]
        print(f"Validation Macro F1-score: {validation_metrics['macro_f1']:.4f}")
    print(f"Detailed report path: {metrics_path}")


if __name__ == "__main__":
    main()
