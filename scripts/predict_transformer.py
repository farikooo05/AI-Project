from transformers import pipeline
import warnings

# Hide unnecessary warnings for a clean console
warnings.filterwarnings('ignore')

def main():
    model_path = "artifacts/models/transformer_augmented"
    
    print("Loading Transformer model... (this will take a few seconds)")
    try:
        # top_k=None forces the model to return probabilities for ALL emotions
        classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=None)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    print("================================================")
    print("Emotion Detection Demo")
    print("Model: DistilBERT (Transformer Augmented)")
    print("Type a comment and press Enter.")
    print("Type 'exit', 'quit', or 'q' to close the demo.")
    print("================================================")

    while True:
        text = input("\nComment: ").strip()
        if text.lower() in ['exit', 'quit', 'q']:
            break
        if not text:
            continue

        # Get predictions
        results = classifier(text)[0] 
        
        # Sort from highest to lowest score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        top_emotion = sorted_results[0]['label'].upper()
        
        print(f"\nPredicted emotion: {top_emotion}")
        print("Top predictions:")
        for i, res in enumerate(sorted_results[:3]):
            prefix = "-> Top 1" if i == 0 else f"   Top {i+1}"
            print(f"{prefix}: {res['label']} ({res['score']*100:.2f}%)")
        
        print("All class probabilities:")
        probs_str = " | ".join([f"{res['label']}: {res['score']*100:.2f}%" for res in sorted_results])
        print(probs_str)
        
        print("Top contributing words/features:")
        print("  - Transformers process the entire context of the sentence, not just isolated words.")
        print("  - Therefore, individual word weights are not applicable here.")

if __name__ == "__main__":
    main()