![Banner](assets/banner.png)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/LangChain-Integration-blueviolet)](https://python.langchain.com/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 🛍️ ReviewSense v2.0: Product Review Analysis & Chatbot Engine

> *ReviewSense v2.0 expands upon the initial analysis engine, adding a powerful, conversational RAG (Retrieval-Augmented Generation) chatbot. It leverages a single, efficient LLM (Mistral 7B GGUF) to provide both deep batch analysis and interactive Q&A grounded in user-provided reviews.*

This project demonstrates an end-to-end workflow, integrating data processing, local LLM execution with `LlamaCpp`, vector storage with FAISS, conversational memory, intent classification, and an interactive Gradio web application.

![demo_tab1](assets/tab1.png)
![demo_tab2](assets/tab2.png)

You can find the Web demo here ➡ [Web Demo](https://huggingface.co/spaces/DanielKiani/ReviewSense)
**[Note]: running this model on the cpu takes a while to complete you can relax and get a cup of coffee while the model generates responses !☕**

---

## 📋 Table of Contents

- [📖 Overview](#-overview)
- [🚀 What's New in v2.0](#-whats-new-in-v20)
- [✨ Key Features (v2.0)](#-key-features-v20)
- [🧠 How It Works: The v2.0 Pipeline](#-how-it-works-the-v20-pipeline)
- [🔧 Challenges & Limitations](#-challenges--limitations)
- [💡 Prompt Engineering Journey](#-prompt-engineering-journey)
- [🔮 Future Improvements](#-future-improvements)
- [⚙️ Setup and Installation](#%EF%B8%8F-setup-and-installation)
- [▶️ Usage](#️-usage)
- [📁 Project Structure (v2.0)](#-project-structure-v20)
- [🛠️ Technologies and Models (v2.0)](#%EF%B8%8F-technologies-and-models-v20)
- [📜 Version History](#-version-history)

---

## 📖 Overview

Building upon the foundation of ReviewSense v1.0, which focused on extracting insights like sentiment, aspects, and summaries using multiple specialized models, **Version 2.0 introduces a significant upgrade: a conversational chatbot**.

This chatbot allows users to ask specific questions about product reviews and receive answers synthesized directly from the provided text. To achieve this efficiently and enhance overall capabilities, v2.0 consolidates the architecture around a single, powerful yet locally runnable Large Language Model (Mistral 7B GGUF). This unified model now handles both the batch analysis tasks (with improved quality) and the interactive Q&A, demonstrating a modern approach to building multi-functional NLP applications.

---

## 🚀 What's New in v2.0

Version 2.0 represents a major leap in functionality and architecture:

1. **🤖 RAG Chatbot Implementation:** Added an interactive chatbot (Phase 2) that uses Retrieval-Augmented Generation (RAG) to answer user questions based on review context.
2. **🧠 Single LLM Architecture:** Replaced the multiple specialized models (DistilBERT, DistilBART, DeBERTa, POS Tagger) from v1.0 with a single, powerful Mistral 7B GGUF model, executed locally via `LlamaCpp`. This model now handles:
    - Batch Analysis (Summary, Aspects, Sentiment - Phase 1) with higher quality.
    - RAG-based Question Answering (Phase 2).
    - Intent Classification (Guardrail for Phase 2).
3. **📄 Dynamic Context Management:** The chatbot can now operate on a default set of reviews or dynamically update its knowledge base using user-uploaded `.txt` or `.csv` files.
4. **💬 Conversational Memory:** Integrated LangChain's `ConversationBufferMemory` allowing the chatbot to understand follow-up questions.
5. **🛡️ Intent Classification Guardrail:** Implemented a robust intent classifier (using the same LLM) to prevent the chatbot from answering off-topic questions, ensuring responses stay grounded in product reviews.
6. **🖥️ Unified Gradio UI:** Developed a two-tab Gradio interface (`app.py`) providing access to both the Batch Analyzer and the RAG Chatbot in a single application.
7. **💻 Local Execution Script:** Added `main.py` for command-line execution of batch analysis or interactive chat without the Gradio UI.
8. **🧱 Modular Code Structure:** Refactored code into `src/pipeline.py` for core logic, improving organization and maintainability.

---

## ✨ Key Features (v2.0)

Includes all features from v1.0 (now powered by Mistral 7B) **plus**:

- **Interactive RAG Chatbot:**
  - Ask specific questions about product reviews (e.g., "How is the battery life?", "Is the app reliable?").
  - Answers synthesized directly from provided review context using RAG.
  - **Conversational Memory:** Understands follow-up questions ("What about the screen?").
  - **Grounded Responses:** Designed to answer only based on the reviews provided.
  - **Intent Guardrail:** Filters out and responds appropriately to off-topic questions.
- **Dynamic Context Loading:**
  - Chatbot operates on default reviews or context loaded from user-uploaded files (`.txt`/`.csv`).
  - Clear indication of the currently active context.
- **Unified LLM Backend:** All NLP tasks (analysis, Q&A, classification) handled by a single Mistral 7B GGUF model running locally.
- **Dual Interface:** Accessible via Gradio web UI (`app.py`) or command line (`main.py`).

---

## 🧠 How It Works: The v2.0 Pipeline

**Phase 1: Batch Analysis (via `analyze_reviews_only` or `analyze_reviews_logic`)**

1. User provides review text (paste or file).
2. The text is passed to the Mistral LLM using three distinct prompts (Summarization, Aspect Extraction, Sentiment Analysis).
3. The LLM generates the three analysis outputs.

**Phase 2: RAG Chatbot (via `ask_question_with_guardrail` or `get_chatbot_response`)**

1. User asks a question.
2. **Intent Classification:** The query is first sent to the Mistral LLM with the `intent_prompt` (few-shot) to classify it as "Product" or "Off-Topic". Robust parsing checks the LLM output.
3. **Routing:**
    - If "Off-Topic", a canned response is returned.
    - If "Product", proceed to RAG.
4. **Context Retrieval:** The user's question is used to query the current FAISS vector store (containing embeddings of the active review context) to retrieve the top `k` relevant review snippets.
5. **Conversational Chain Execution (`ConversationalRetrievalChain`):**
    - **Condense Question:** If there's chat history, the LLM uses `CONDENSE_QUESTION_PROMPT` to rephrase the current question into a standalone query.
    - **RAG Generation:** The condensed question and retrieved context snippets are passed to the LLM with the strict `qa_prompt`. The LLM synthesizes an answer based *only* on the provided context.
    - **Memory Update:** The question and final answer are added to the `ConversationBufferMemory`.
6. **Response:** The synthesized answer is returned to the user.

---

## 🔧 Challenges & Limitations

Developing v2.0 involved significant experimentation and revealed several challenges:

1. **Consistent Instruction Following:** While powerful, the Mistral 7B GGUF model sometimes struggled to consistently follow complex negative constraints or nuanced instructions in prompts, especially within the RAG chain. This led to:
    - **Context Leakage:** Occasionally including irrelevant details from retrieved chunks (e.g., mentioning webcam when asked about battery).
    - **Hallucination:** Making up information (e.g., mentioning "phone" for laptop battery, inventing prices or product names).
    - **Over-Cautiousness:** Incorrectly stating "cannot find information" even when relevant details were present in the context, particularly for negative aspects (e.g., hardware issues).
    - **Misinterpretation:** Failing to correctly understand the specific user question (e.g., "taste" vs. "type", comparison questions).
2. **Prompt Engineering Complexity:** Finding the right prompt structure required extensive iteration. Simple prompts lacked control, while overly complex prompts sometimes confused the model. Few-shot prompting proved essential for reliable intent classification. Balancing strictness (for grounding) with flexibility (to allow synthesis) in the RAG prompt was difficult.
3. **Intent Classification Brittleness:** Getting the LLM to output *only* the classification label required moving from zero-shot, to strict instructions, to few-shot examples, and finally adding robust parsing logic (`parse_intent`) to handle noisy LLM outputs reliably.
4. **Performance:** Running the 7B parameter GGUF model on a CPU is significantly slower than using smaller models or GPU acceleration. Batch analysis and RAG responses take noticeable time.
5. **Evaluation Bottleneck:** Using external APIs (like OpenAI) for RAGAs evaluation can incur costs and hit rate limits. Using the local model for evaluation is free but slower and potentially less objective.

---

## 💡 Prompt Engineering Journey

Achieving the final, relatively stable performance required significant iteration on the prompts, particularly for the RAG chain (`qa_prompt`) and intent classification (`intent_prompt`).

**Intent Classification (`intent_prompt`):**

- Initial attempts with simple zero-shot prompts failed, with the model providing verbose, incorrect classifications.
- Adding strict formatting rules (`MUST BE EXACTLY...`) helped but wasn't sufficient.
- **Few-Shot Prompting** (providing explicit examples within the prompt) proved crucial for forcing the model to output the correct labels, although often with extra text.
- **Robust Parsing (`parse_intent`)** was added to reliably extract the core "Product" or "Off-Topic" keyword from the model's potentially noisy output.

**Final `intent_template`:**

```python
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
```

**RAG Generation (`qa_system_prompt`):**

- Initial simple prompts led to significant hallucination and context leakage.

- Adding strict rules improved grounding but sometimes made the model overly cautious, failing to find information present in the context.

- Explicitly addressing failure modes (like comparisons) helped for those specific cases.

- Experimenting with different chain types (`stuff`, `map_reduce`, `refine`) showed limitations related to context window size and model instruction following. `stuff` with `ConversationalRetrievalChain` proved most practical.

**Final qa_system_prompt (within qa_prompt):**

```python
# RAG System Prompt (qa_system_prompt)
qa_system_prompt = """[INST]You are a factual assistant providing answers based **only** on the customer reviews provided.
Your task is to answer the user's question concisely using information explicitly found in the 'CONTEXT' snippets below.

**CRITICAL RULES TO FOLLOW:**
1.  **STRICTLY Contextual:** Base your answer ENTIRELY and ONLY on the information within the 'CONTEXT' section. Do NOT use any prior knowledge or external information.
2.  **Direct & Relevant:** Answer ONLY the specific question asked. Do NOT include details from the context that are irrelevant to the question, even if they appear nearby.
3.  **Synthesize Concisely:** Combine relevant facts from potentially multiple snippets into a brief answer (usually 1-3 sentences). Do NOT quote long passages unless absolutely necessary.
4.  **No Comparisons Outside Context:** If the question asks to compare the product to something *not mentioned* in the reviews, state *only*: "Cannot compare based on the provided reviews." Do not add details about the product itself in this case.
5.  **Handle Missing Info Carefully:** If, after carefully reading the context, you genuinely cannot find any information relevant to the question, state *only*: "Based on the provided reviews, I cannot find information about that." Check thoroughly before using this response.
6.  **Factual Tone:** Do NOT apologize, express opinions, make recommendations, or use conversational filler. Just state the facts found in the reviews.

CONTEXT:
---
{context}
---

QUESTION: {question} [/INST]
Answer:"""
```

This iterative process demonstrates the practical challenges and refinement needed when working with local LLMs in complex pipelines.

---

## 🔮 Future Improvements

- **RAG Evaluation**: Fully implement and integrate RAGAs (or TruLens) evaluation using the local LLM or a free tier API to get quantitative metrics on Faithfulness, Answer Relevancy, etc.

- **LLM Upgrade**: Experiment with larger or more advanced instruction-tuned models (e.g., Mixtral GGUF, Llama 3 70/8B Instruct GGUF, or API-based models like GPT-4/Claude 3) to achieve higher consistency in instruction following and synthesis.

- **Advanced Retrieval**: Explore more sophisticated retrieval techniques (e.g., HyDE, MultiQueryRetriever, Re-ranking) to improve the quality of context chunks passed to the LLM, potentially reducing generation errors.

- **Batch Processing for Analysis**: Re-implement batch processing for Phase 1 using techniques like `map_reduce` to handle large numbers of reviews that exceed the LLM's context window.

- **Error Handling & UI**: Add more granular error handling and user feedback in the Gradio UI (e.g., clearer messages if context loading fails).

- **Automated Testing**: Implement unit and integration tests using `pytest` for the core logic in `src/pipeline.py`.

---

## ⚙️ Setup and Installation

**1. Clone the Repository**

```bash
git clone [https://github.com/Deathshot78/ReviewSense.git](https://github.com/Deathshot78/ReviewSense.git) # Replace with your repo URL if different
cd ReviewSense
```

**2. Install Required Packages**

```bash
pip install -r requirements.txt
```

**3. Download LLM Model**

The scripts will attempt to download the Mistral-7B GGUF model (`mistral-7b-instruct-v0.1.Q4_K_M.gguf`, ~4.4GB) automatically via `wget` on the first run if it's not found in the root directory. You can also download it manually from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) and place it in the project root.

---

## ▶️ Usage

**Web App (Gradio)**

Run the Gradio app:

```bash
python app.py
```

Access the interface in your browser

- **Tab 1 ("Batch Analyzer"):** Paste reviews or upload a file to perform Summary, Aspect Extraction, and Sentiment Analysis. This does not affect the chatbot context.

- **Tab 2 ("Ask a Question"):** Chat with the RAG bot. Use the file upload and "Update Chatbot Context" button within this tab to change the reviews the chatbot uses. Use "Reset Chatbot Context to Default" to revert to the built-in laptop reviews. Use "Reset Chat Memory" to clear the conversation history.

---

## 📁 Project Structure (v2.0)

```bash
ReviewSense/
├── 📄 README.md               # Project documentation 
├── 📁 src/                    # Source code for core 
│   ├── 📄 app.py              # Gradio web application
│   ├── 📄 pipeline.py         # Core functions for analysis, RAG, etc. 
│   └── 📄 main.py             # Command-line execution
├── 📄 requirements.txt        # Python dependencies
├── 📄 .gitignore              # Files ignored by Git
└── 🖼️ assets/                 # images 
```

---

## 🛠️ Technologies and Models (v2.0)

**Core Technologies**

- Python 3.10+

- LangChain: Orchestration, Chains (ConversationalRetrievalChain), Memory, Prompts

- llama-cpp-python: Local execution of GGUF models on CPU

- FAISS (faiss-cpu): Efficient vector similarity search

- Sentence-Transformers (all-MiniLM-L6-v2): Text embeddings

- Gradio: Interactive web UI

- PyTorch (dependency via transformers/sentence-transformers)

- Pandas, NumPy (standard data handling)

**Core LLM**

- Mistral 7B Instruct v0.1 (GGUF Q4_K_M): Used for all NLP tasks (Analysis, RAG Generation, Intent Classification). Downloaded from TheBloke on Hugging Face.
  
---

## 📜 Version History

- v2.0 (Current): RAG Chatbot, Single Mistral 7B model, Dynamic Context, Memory, Guardrails, Gradio UI, Code Refactoring.

- v1.0: [https://github.com/DanielKiani/ReviewSense/releases/tag/v1.0] - Initial Batch Analysis Engine using multiple specialized models (DistilBERT, DistilBART, etc.). Focused on Sentiment, Aspects, and Summarization. (See v1.0 README for full details).
