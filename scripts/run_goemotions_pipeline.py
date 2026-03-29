"""Run the full GoEmotions data workflow: download raw data, map labels, save processed CSV."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.dataset_preparation import load_label_mapping, prepare_dataset
from emotion_detector.public_datasets import download_and_prepare_goemotions


def main() -> None:
    """Download GoEmotions raw data and prepare the final 6-class dataset."""
    raw_output_path = PROJECT_ROOT / "data" / "raw" / "goemotions_raw.csv"
    processed_output_path = PROJECT_ROOT / "data" / "processed" / "goemotions_6class.csv"
    mapping_path = PROJECT_ROOT / "configs" / "goemotions_label_mapping.json"
    reports_dir = PROJECT_ROOT / "artifacts" / "reports" / "dataset_preparation"

    print("Step 1/2: Downloading and exporting raw GoEmotions...")
    raw_frame = download_and_prepare_goemotions(raw_output_path)
    print(f"Raw GoEmotions CSV saved to: {raw_output_path}")
    print(f"Raw rows saved: {len(raw_frame)}")

    print("\nStep 2/2: Applying 6-class mapping and saving processed dataset...")
    label_mapping = load_label_mapping(mapping_path)
    processed_frame = prepare_dataset(
        input_path=raw_output_path,
        output_path=processed_output_path,
        text_column="text",
        label_column="label",
        label_mapping=label_mapping,
        remove_duplicates=True,
        reports_dir=reports_dir,
    )
    print(f"Processed dataset saved to: {processed_output_path}")
    print(f"Processed rows saved: {len(processed_frame)}")
    print(f"Preparation reports saved to: {reports_dir}")
    report_paths = processed_frame.attrs.get("report_paths", {})
    if report_paths:
        print(f"  - Mapping audit: {report_paths['mapping_audit']}")
        print(f"  - Class distribution: {report_paths['class_distribution']}")
        print(f"  - Class weights: {report_paths['class_weights']}")
        print(f"  - Sample examples: {report_paths['sample_examples_per_class']}")
        print(f"  - Warnings: {report_paths['warnings']}")
    print("\nNext step:")
    print("  python scripts/train_baseline.py")


if __name__ == "__main__":
    main()
