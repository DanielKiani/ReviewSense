<<<<<<< HEAD
# main.py

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import os
import argparse # For command-line arguments

# Import the logic functions from src
import src.pipeline as pipeline

# --- Global Objects & Setup ---
# (Similar setup as app.py, load models, prompts etc.)
print("--- Starting Local Execution Setup ---")
# 1. Check/Define Model Path
model_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
if not os.path.exists(model_name):
    print(f"ERROR: Model file '{model_name}' not found. Please download it first.")
    exit()

# 2. Prepare Default Sample Data (Optional, for context testing)
default_reviews_text = """...""" # Paste default laptop reviews
default_reviews_list = [r.strip() for r in default_reviews_text.strip().split('---') if r.strip()]

# 3. Load Embedding Model, Text Splitter
print("Loading embedding model and text splitter...")
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs=model_kwargs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)

# 4. Create Default Vector Store
print("Creating default FAISS vector store...")
default_vector_store = pipeline.create_vector_store_from_content(
    "\n---\n".join(default_reviews_list), text_splitter, embeddings
)
if default_vector_store is None: raise ValueError("Failed to create default vector store!")

# 5. Load the LLM
print("Loading LLM (Mistral-7B GGUF)...")
llm = LlamaCpp(
    model_path=model_name, n_gpu_layers=0, n_batch=512, n_ctx=4096,
    f16_kv=True, temperature=0.0, max_tokens=512, verbose=False,
    stop=["[/INST]", "User:", "Assistant:"]
)

# 6. Define All Prompts
print("Defining all prompts...")
# -- Phase 1 --
summary_template = """[INST] ... Reviews:\n{reviews} [/INST]\nConcise Summary:"""
summary_prompt = PromptTemplate(template=summary_template, input_variables=["reviews"])
aspect_template = """[INST] ... Reviews:\n{reviews} [/INST]\nKey Pros and Cons:"""
aspect_prompt = PromptTemplate(template=aspect_template, input_variables=["reviews"])
sentiment_template = """[INST] ... Reviews:\n{reviews} [/INST]\nOverall Sentiment (Score 1-10):"""
sentiment_prompt = PromptTemplate(template=sentiment_template, input_variables=["reviews"])
# -- Phase 2 --
condense_question_template = """[INST] Given the following conversation... Follow Up Input: {question} [/INST]\nStandalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)
qa_system_prompt = """[INST]
You are a factual assistant that answers only using the provided product reviews.
If the reviews include partial or uncertain information, summarize what they say.
If there is no information at all about the user’s question, respond with:
"I'm sorry, there isn't enough information in the reviews to answer that."

Do not use or infer information about price, comparisons to other brands, or availability unless they are directly mentioned in the reviews.
Always include a short "Evidence:" sentence if you found relevant mentions.

Context:
{context}

User question:
{question}
[/INST]
"""
qa_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(qa_system_prompt), HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}\n\nHelpful Answer:")])
intent_template = """
[INST]
**CRITICAL INSTRUCTION:** Classify the user's query into ONLY ONE of two categories: "Product" or "Off-Topic".
Your response MUST be EXACTLY "Product" or EXACTLY "Off-Topic".

**EXAMPLES:**
Query: How is the battery life?
Classification: Product
Query: What are the complaints about the screen?
Classification: Product
Query: Does it come in blue?
Classification: Product
Query: What is the capital of France?
Classification: Off-Topic
Query: Hello there
Classification: Off-Topic
Query: Who are you?
Classification: Off-Topic

**NOW CLASSIFY THIS QUERY:**
Query: {query}
[/INST]
Classification:"""
intent_prompt = PromptTemplate(template=intent_template, input_variables=["query"])

# 7. Memory Object (Needed for chatbot logic)
chat_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

print("--- Local Setup Complete ---")


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReviewSense Analysis or Chat locally.")
    parser.add_argument("--mode", choices=['analyze', 'chat'], required=True, help="Mode to run: 'analyze' reviews from a file, or 'chat' interactively.")
    parser.add_argument("--input", type=str, help="Path to input .txt file for 'analyze' mode, or initial query for 'chat' mode.")
    parser.add_argument("--context", type=str, help="Optional: Path to a .txt file to use as context for 'chat' mode (defaults to built-in laptop reviews).")

    args = parser.parse_args()

    # --- ANALYZE MODE ---
    if args.mode == 'analyze':
        if not args.input or not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found for analyze mode.")
            exit()
        print(f"\n--- Running Analysis on: {args.input} ---")
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                review_content = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            exit()

        summary, aspects, sentiment = pipeline.analyze_reviews_logic(
            review_content, llm, summary_prompt, aspect_prompt, sentiment_prompt
        )
        print("\n--- Analysis Results ---")
        print("\n[Summary]")
        print(summary)
        print("\n[Aspects]")
        print(aspects)
        print("\n[Sentiment]")
        print(sentiment)

    # --- CHAT MODE ---
    elif args.mode == 'chat':
        print("\n--- Starting Interactive Chat ---")
        # Determine context
        chat_vector_store = default_vector_store
        context_name = "Default Laptop Reviews"
        if args.context:
            if not os.path.exists(args.context):
                print(f"Warning: Context file '{args.context}' not found. Using default context.")
            else:
                print(f"Loading context from: {args.context}")
                try:
                    with open(args.context, 'r', encoding='utf-8') as f:
                        context_content = f.read()
                    chat_vector_store = pipeline.create_vector_store_from_content(
                        context_content, text_splitter, embeddings
                    )
                    if chat_vector_store:
                        context_name = f"File: {os.path.basename(args.context)}"
                    else:
                        print("Failed to load context file. Using default context.")
                        chat_vector_store = default_vector_store
                except Exception as e:
                    print(f"Error reading context file '{args.context}': {e}. Using default context.")
                    chat_vector_store = default_vector_store

        print(f"Using context: {context_name}")
        chat_memory.clear() # Start fresh chat session

        # Handle initial query if provided
        if args.input:
            print("\nUser:", args.input)
            response = pipeline.get_chatbot_response(
                message=args.input,
                chat_memory=chat_memory,
                vector_store=chat_vector_store,
                llm=llm,
                intent_prompt=intent_prompt,
                condense_prompt=CONDENSE_QUESTION_PROMPT,
                qa_prompt=qa_prompt
            )
            print("\nAssistant:", response)

        # Interactive loop
        print("\nEnter your questions (type 'quit' or 'exit' to stop):")
        while True:
            try:
                user_message = input("\nUser: ")
                if user_message.lower() in ['quit', 'exit']:
                    break
                if not user_message:
                    continue

                response = pipeline.get_chatbot_response(
                    message=user_message,
                    chat_memory=chat_memory,
                    vector_store=chat_vector_store,
                    llm=llm,
                    intent_prompt=intent_prompt,
                    condense_prompt=CONDENSE_QUESTION_PROMPT,
                    qa_prompt=qa_prompt
                )
                print("\nAssistant:", response)

            except EOFError: # Handle Ctrl+D
                break
            except KeyboardInterrupt: # Handle Ctrl+C
                break
        print("\n--- Chat session ended. ---")

    print("\n--- Local Execution Finished ---")
=======
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
>>>>>>> e6de3c4338f79386345fa6e4bba5b0666ad808da
