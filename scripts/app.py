import gradio as gr
import os
import torch
from transformers import AutoTokenizer
import pandas as pd
import re

# --- IMPORTANT ---
# This script assumes you have a 'models.py' file in the same directory
# containing the definitions for all model and inference classes.
try:
    from models import (
        ReviewSummarizer,
        AspectAnalyzer,
        AspectExtractor,
        FineTunedSentimentClassifier
    )
except ImportError:
    print("CRITICAL ERROR: Make sure 'models.py' exists and contains the required classes.")
    # Define dummy classes if imports fail, so Gradio can at least launch with an error message.
    class ReviewSummarizer: pass
    class AspectAnalyzer: pass
    class AspectExtractor: pass
    class FineTunedSentimentClassifier: pass

# --- Configuration ---
# This should be the relative path to your checkpoint file within the repository.
SENTIMENT_CHECKPOINT_PATH = "checkpoints/sentiment-binary-best-checkpoint.ckpt"


# --- Pre-defined Aspect Dictionaries for Different Product Categories ---
ASPECT_DICTIONARIES = {
    "Phone": ['camera', 'battery', 'battery life', 'screen', 'performance', 'price', 'design'],
    "Coffee Maker": ['ease of use', 'design', 'noise level', 'coffee quality', 'brew time', 'cleaning'],
    "Book": ['plot', 'characters', 'writing style', 'pacing', 'ending'],
    "Default": ['quality', 'price', 'service', 'design', 'features'] # A fallback list
}


# --- Load All Models (Global Objects) ---
print("--- Initializing all models for the Gradio App ---")
sentiment_classifier, summarizer, aspect_analyzer, aspect_extractor = None, None, None, None
try:
    summarizer = ReviewSummarizer(force_cpu=True)
    aspect_analyzer = AspectAnalyzer(force_cpu=True)
    aspect_extractor = AspectExtractor(force_cpu=True)

    if os.path.exists(SENTIMENT_CHECKPOINT_PATH):
        sentiment_classifier = FineTunedSentimentClassifier(
            checkpoint_path=SENTIMENT_CHECKPOINT_PATH, force_cpu=True
        )
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Sentiment checkpoint path not found.                    !!!")
        print(f"!!! Path checked: '{SENTIMENT_CHECKPOINT_PATH}'")
        print("!!! The fine-tuned sentiment model will NOT be loaded.               !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\n--- All models loaded successfully ---\n")
except Exception as e:
    print(f"An error occurred during model initialization: {e}")


# --- Define the Core Analysis Function ---
def analyze_review(review_text, product_category):
    if not review_text:
        return {"ERROR": "Please enter a review."}, "", None

    # --- a. Overall Sentiment Analysis ---
    if sentiment_classifier:
        sentiment_result = sentiment_classifier.classify(review_text)
        sentiment_output = {
            sentiment_result['label']: f"{sentiment_result['score']:.2f}"
        }
    else:
        # **ROBUST ERROR HANDLING:** This prevents the app from crashing.
        # It returns a dictionary that the Gradio Label component can display.
        sentiment_output = {"Error: Model Not Loaded": 1.0}

    # --- b. Review Summarization ---
    if summarizer:
        summary_output = summarizer.summarize(review_text)
    else:
        summary_output = "ERROR: Summarizer model not loaded."

    # --- c. Dynamic Aspect Extraction & Analysis ---
    aspect_df = None
    if aspect_extractor and aspect_analyzer:
        aspect_dictionary = ASPECT_DICTIONARIES.get(product_category, ASPECT_DICTIONARIES["Default"])
        extracted_aspects = aspect_extractor.extract(review_text, aspect_dictionary=aspect_dictionary)

        if extracted_aspects:
            aspect_results = aspect_analyzer.analyze(review_text, extracted_aspects)
            aspect_df = pd.DataFrame([
                {'Aspect': aspect, 'Sentiment': result['sentiment'], 'Score': f"{result['score']:.2f}"}
                for aspect, result in aspect_results.items()
            ])

    return sentiment_output, summary_output, aspect_df


# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛍️ ReviewSense: Product Review Analysis Engine")
    gr.Markdown(
        "Enter a product review and select the product category. The tool will automatically "
        "detect relevant features and provide an overall sentiment score, a summary, and a "
        "breakdown of sentiment towards each feature."
    )

    with gr.Row():
        with gr.Column(scale=2):
            review_input = gr.Textbox(
                lines=10,
                label="Enter Product Review Here",
                placeholder="e.g., The camera is amazing, but the battery life is terrible..."
            )
            category_input = gr.Dropdown(
                choices=list(ASPECT_DICTIONARIES.keys()),
                label="Select Product Category",
                value="Phone"
            )
            analyze_button = gr.Button("Analyze Review", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Overall Sentiment")
            sentiment_output = gr.Label()

            gr.Markdown("### Generated Summary")
            summary_output = gr.Textbox(lines=5, label="Summary", interactive=False)

            gr.Markdown("### Detected Aspect Sentiments")
            aspect_output = gr.DataFrame(headers=["Aspect", "Sentiment", "Score"], label="Aspects", interactive=False)

    # Connect the button to the function
    analyze_button.click(
        fn=analyze_review,
        inputs=[review_input, category_input],
        outputs=[sentiment_output, summary_output, aspect_output]
    )

    gr.Examples(
        examples=[
            [
                "The camera on this phone is incredible, the pictures are professional quality. However, the battery life is a total disaster, it barely lasts half a day with light use. The screen is bright and responsive, which I love.",
                "Phone"
            ],
            [
                "I am absolutely in love with this coffee maker! It's incredibly easy to use, brews a perfect cup every single time, and the design looks fantastic on my countertop. It's also surprisingly quiet.",
                "Coffee Maker"
            ],
            [
                "An amazing story with characters that felt so real. The plot had me hooked from the first page, though I felt the ending was a bit rushed.",
                "Book"
            ]
        ],
        inputs=[review_input, category_input]
    )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch()