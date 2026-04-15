import json
import os

def main():
    report_path = "artifacts/reports/transformer_training_augmented/evaluation_report.json"
    
    if not os.path.exists(report_path):
        print(f"Error: File {report_path} not found.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Transformer (Augmented) evaluation summary")
    
    # Extract metrics from JSON
    accuracy = data.get("accuracy", 0)
    macro_avg = data.get("classification_report", {}).get("macro avg", {})
    weighted_avg = data.get("classification_report", {}).get("weighted avg", {})
    
    macro_precision = macro_avg.get("precision", 0)
    macro_recall = macro_avg.get("recall", 0)
    macro_f1 = macro_avg.get("f1-score", data.get("macro_f1", 0))
    weighted_f1 = weighted_avg.get("f1-score", data.get("weighted_f1", 0))

    # Print exact same format as baseline
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print(f"Detailed report path: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()