"""Terminal entry point for training the baseline emotion classifier."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.training import run_training


if __name__ == "__main__":
    run_training(PROJECT_ROOT / "configs" / "baseline_config.json")
