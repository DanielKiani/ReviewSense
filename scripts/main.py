import os
import torch
import pandas as pd

try:
    from data_prepare import ReviewDataset, ReviewDataModule
    from models import SentimentClassifier, ReviewSummarizer, AspectAnalyzer, FineTunedSentimentClassifier, AspectExtractor
except ImportError:
    print("CRITICAL ERROR: Make sure 'review_summarizer.py', 'aspect_extractor.py', and 'sentiment_classifier_model.py' are in the same directory.")
    exit()

# --- Configuration ---
# --- IMPORTANT: UPDATE THIS PATH ---
# You need to provide the path to the best checkpoint file that was saved
# during the training of your sentiment model.
SENTIMENT_CHECKPOINT_PATH = "checkpoints/sentiment-binary-best-checkpoint.ckpt"

# --- Pre-defined Aspect Dictionaries for Different Product Categories ---
ASPECT_DICTIONARIES = {
    "Phone": ['camera', 'battery', 'battery life', 'screen', 'performance', 'price', 'design'],
    "Coffee Maker": ['ease of use', 'design', 'noise level', 'coffee quality', 'brew time', 'cleaning'],
    "Book": ['plot', 'characters', 'writing style', 'pacing', 'ending'],
    "Default": ['quality', 'price', 'service', 'design', 'features'] # A fallback list
}

def main():
    """
    Main function to run the command-line review analysis tool.
    """
    # --- 1. Load All Models ---
    print("--- Initializing all models ---")
    sentiment_classifier, summarizer, aspect_analyzer, aspect_extractor = None, None, None, None
    try:
        summarizer = ReviewSummarizer(force_cpu=True)
        aspect_analyzer = AspectAnalyzer(force_cpu=True)
        aspect_extractor = AspectExtractor(force_cpu=True)

        if not os.path.exists(SENTIMENT_CHECKPOINT_PATH):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! WARNING: Sentiment checkpoint path not found or not set.         !!!")
            print(f"!!! Please update the 'SENTIMENT_CHECKPOINT_PATH' variable in main.py")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            sentiment_classifier = FineTunedSentimentClassifier(
                checkpoint_path=SENTIMENT_CHECKPOINT_PATH, force_cpu=True
            )
        print("\n--- All models loaded successfully ---\n")
    except Exception as e:
        print(f"An error occurred during model initialization: {e}")
        return

    # --- 2. Interactive Loop ---
    while True:
        print("\n==================================================")
        print("          Product Review Analysis Tool          ")
        print("==================================================")

        # Get user input
        review_text = input("Enter the product review text (or type 'quit' to exit):\n> ")
        if review_text.lower() == 'quit':
            break

        print("\nAvailable Product Categories:")
        for i, category in enumerate(ASPECT_DICTIONARIES.keys(), 1):
            print(f"{i}. {category}")

        category_choice = input(f"Select a product category (1-{len(ASPECT_DICTIONARIES)}):\n> ")
        try:
            category_idx = int(category_choice) - 1
            product_category = list(ASPECT_DICTIONARIES.keys())[category_idx]
        except (ValueError, IndexError):
            print("Invalid choice. Using 'Default' category.")
            product_category = "Default"

        # --- 3. Run Analysis ---
        print("\n--- Analyzing Review... ---")

        # a. Overall Sentiment
        sentiment_result = sentiment_classifier.classify(review_text)

        # b. Summary
        summary_result = summarizer.summarize(review_text)

        # c. Aspect Extraction and Analysis
        aspect_dictionary = ASPECT_DICTIONARIES.get(product_category)
        extracted_aspects = aspect_extractor.extract(review_text, aspect_dictionary)
        aspect_results = None
        if extracted_aspects:
            aspect_results = aspect_analyzer.analyze(review_text, extracted_aspects)

        # --- 4. Display Results ---
        print("\n-------------------- ANALYSIS RESULTS --------------------")
        print(f"\n[ Overall Sentiment ]")
        print(f"  - Sentiment: {sentiment_result['label']} (Score: {sentiment_result['score']:.2f})")

        print(f"\n[ Generated Summary ]")
        print(f"  - {summary_result}")

        print(f"\n[ Detected Aspect Sentiments ]")
        if aspect_results:
            for aspect, result in aspect_results.items():
                print(f"  - {aspect.title()}: {result['sentiment']} (Score: {result['score']:.2f})")
        else:
            print("  - No relevant aspects from the dictionary were detected in the review.")
        print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
