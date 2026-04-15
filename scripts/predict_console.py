"""Console application for interactive emotion prediction."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.inference import run_console_inference


if __name__ == "__main__":
    run_console_inference(PROJECT_ROOT / "configs" / "baseline_config.json")
