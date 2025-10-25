# src/pipeline.py

import os
import io
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate # Ensure necessary Langchain imports are here if needed directly

# --- Core Logic Functions ---

def analyze_reviews_logic(review_text: str, llm, summary_prompt, aspect_prompt, sentiment_prompt):
    """
    Performs Phase 1 analysis (summary, aspects, sentiment) on the provided text.
    """
    print(f"Running batch analysis logic on {len(review_text)} chars...")
    try:
        summary_result = llm.invoke(summary_prompt.format(reviews=review_text)).strip()
        print("   -> Summary generated.")
        aspect_result = llm.invoke(aspect_prompt.format(reviews=review_text)).strip()
        print("   -> Aspects extracted.")
        sentiment_result = llm.invoke(sentiment_prompt.format(reviews=review_text)).strip()
        print("   -> Sentiment analyzed.")
        return summary_result, aspect_result, sentiment_result
    except Exception as e:
        print(f"ERROR during batch analysis logic: {e}")
        error_msg = f"Error during analysis: {e}"
        return error_msg, error_msg, error_msg

def create_vector_store_from_content(content: str, text_splitter, embeddings):
    """
    Splits content and creates a new FAISS vector store.
    Returns the vector store or None if an error occurs.
    """
    print("Creating new vector store from content...")
    if not content:
        print("Error: No content provided to create vector store.")
        return None

    # Split content
    if "\n---\n" in content:
        reviews_list = [r.strip() for r in content.strip().split('\n---\n') if r.strip()]
    else:
        reviews_list = [r.strip() for r in content.strip().split('\n\n') if r.strip()]
        if len(reviews_list) <= 1: reviews_list = [content.strip()] # Single block case

    if not reviews_list:
        print("Error: Could not extract reviews from content.")
        return None

    review_chunks = text_splitter.create_documents(reviews_list)
    if not review_chunks:
       print("Error: Failed to create document chunks.")
       return None

    try:
        vector_store = FAISS.from_documents(review_chunks, embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None

def parse_intent(llm_output: str) -> str:
    """
    Parses the LLM output to find 'Product' or 'Off-Topic'.
    Defaults to 'Off-Topic' if neither is found or output is unexpected.
    Uses case-insensitive 'in' check for robustness.
    """
    output_lower = llm_output.strip().lower()
    if "product" in output_lower:
        return "Product"
    elif "off-topic" in output_lower:
        return "Off-Topic"
    else:
        print(f"   -> Unexpected classification: '{llm_output.strip()}'. Defaulting to Off-Topic.")
        return "Off-Topic"

def get_chatbot_response(message: str, chat_memory, vector_store, llm, intent_prompt, condense_prompt, qa_prompt):
    """
    Handles Phase 2: Classifies intent and runs RAG if appropriate.
    Returns the chatbot's response string.
    """
    print(f"\nProcessing chatbot query: {message}")

    # --- 1. Classify Intent ---
    formatted_intent_prompt = intent_prompt.format(query=message)
    intent_result_raw = llm.invoke(formatted_intent_prompt)
    print(f"   DEBUG: Raw Intent Output: '{intent_result_raw.strip()}'")
    intent = parse_intent(intent_result_raw)
    print(f"   -> Detected Intent: {intent}")

    # --- 2. Route ---
    if intent == "Product":
        print("   -> Routing to RAG chain...")
        if vector_store is None:
             print("   ERROR: No vector store available for RAG.")
             return "Sorry, I don't have any review context loaded to answer product questions."

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Create chain dynamically for each call
        conv_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=chat_memory,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True, # Required for context list in result
            verbose=False
        )
        try:
            # Pass only question - memory handles history internally
            result = conv_qa_chain.invoke({"question": message})
            answer = result['answer'].strip()
            print(f"   -> RAG Answer: {answer}")
            return answer
        except Exception as e:
             print(f"ERROR during RAG chain execution: {e}")
             # Optionally log traceback: import traceback; traceback.print_exc()
             return "Sorry, I encountered an error trying to find an answer in the reviews."

    else: # Off-Topic
        print("   -> Routing to canned response...")
        answer = "I'm sorry, I can only answer questions about the product reviews for this item."
        # Optional: Save off-topic turn to memory if desired
        # chat_memory.save_context({"question": message}, {"answer": answer})
        return answer