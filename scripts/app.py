# app.py

import gradio as gr
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import os
import io

# Import the logic functions from src
import pipeline

# --- Global Objects & Setup ---
# (Most setup code remains here as it's needed globally for the app)

print("--- Starting App Setup ---")
# 1. Download Model File
model_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
if not os.path.exists(model_name):
    print("Downloading model...")
    os.system(f"wget {model_url}")
else:
    print("Model already downloaded.")

# 2. Prepare Default Sample Data & Example Batch
print("Loading default reviews...")
default_reviews_text = """
This laptop is a beast! The M3 chip is incredibly fast, and the battery lasts a solid 10 hours of heavy use... (rest of laptop reviews) ...dongle life is real.
---
I'm a student, and the battery life is a lifesaver... Highly recommend for college.
---
The keyboard is a dream to type on... Bluetooth connection dropping...
---
Video editing on this machine is flawless... price is very expensive...
---
I bought this for travel... battery easily gets me through a 6-hour flight...
---
Don't buy this if you need a lot of ports... only two USB-C ports...
"""
default_reviews_list = [r.strip() for r in default_reviews_text.strip().split('---') if r.strip()]

example_batch = """
I'm absolutely blown away by the "NovaBlend Pro" blender!... (rest of blender example)... save your money.
"""

# 3. Load Embedding Model, Text Splitter
print("Loading embedding model and text splitter...")
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs=model_kwargs
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)

# 4. Create Default Vector Store
print("Creating default FAISS vector store...")
default_vector_store = pipeline.create_vector_store_from_content(
    "\n---\n".join(default_reviews_list), text_splitter, embeddings
)
if default_vector_store is None:
    raise ValueError("Failed to create default vector store!")
print("Default vector store created successfully.")

# Global variable to hold the CURRENT vector store for the chatbot
# NOTE: Using a global like this works for simple Gradio apps but isn't
# robust for multiple users. Gradio state or session management is better
# for multi-user scenarios, but this keeps it simpler for now.
current_chatbot_vector_store = default_vector_store
current_context_source = "Default Laptop Reviews"

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
summary_template = """[INST] You are a helpful assistant... Reviews:\n{reviews} [/INST]\nConcise Summary:"""
summary_prompt = PromptTemplate(template=summary_template, input_variables=["reviews"])
aspect_template = """[INST] You are a helpful product analyst... Reviews:\n{reviews} [/INST]\nKey Pros and Cons:"""
aspect_prompt = PromptTemplate(template=aspect_template, input_variables=["reviews"])
sentiment_template = """[INST] You are a helpful sentiment analyst... Reviews:\n{reviews} [/INST]\nOverall Sentiment (Score 1-10):"""
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

# 7. Global Memory Object
chat_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

print("--- App Setup Complete ---")


# --- Gradio Helper Functions (Wrappers around pipeline logic) ---

def analyze_reviews_gradio_wrapper(review_text, review_file):
    """Gradio wrapper for Phase 1 analysis."""
    content = ""
    if review_file is not None:
        try:
            if hasattr(review_file, 'name'): file_path = review_file.name; f=open(file_path, 'rb'); byte_content = f.read(); f.close()
            else: byte_content = review_file
            try: content = byte_content.decode('utf-8')
            except UnicodeDecodeError: content = byte_content.decode('latin-1')
        except Exception as e: return f"Error reading file: {e}", "", ""
        if not content: return "Error: File empty", "", ""
    elif review_text:
        content = review_text
    else:
        return "Please paste reviews or upload a file.", "", ""

    # Call the core logic function
    return pipeline.analyze_reviews_logic(
        content, llm, summary_prompt, aspect_prompt, sentiment_prompt
    )

def update_chatbot_context_gradio_wrapper(chatbot_file_upload):
    """Gradio wrapper to update chatbot context."""
    global current_chatbot_vector_store, current_context_source # Modify globals

    if chatbot_file_upload is None:
        return f"No file uploaded. Chatbot context remains: **{current_context_source}**."

    print("Processing chatbot context file via Gradio...")
    content = ""
    file_name = "Uploaded File"
    try:
        if hasattr(chatbot_file_upload, 'name'):
            file_path = chatbot_file_upload.name
            file_name = os.path.basename(file_path)
            with open(file_path, 'rb') as f: byte_content = f.read()
        else: byte_content = chatbot_file_upload
        try: content = byte_content.decode('utf-8')
        except UnicodeDecodeError: content = byte_content.decode('latin-1')
    except Exception as e: return f"Error reading file: {e}. Context not updated."
    if not content: return "File empty. Context not updated."

    # Call the core logic function to create the store
    new_vector_store = pipeline.create_vector_store_from_content(content, text_splitter, embeddings)

    if new_vector_store:
        current_chatbot_vector_store = new_vector_store # Update global store
        current_context_source = f"File: {file_name}"
        status_message = f"Chatbot context updated using **{file_name}**."
        print(status_message)
        return status_message
    else:
        # If store creation failed, keep the old one
        status_message = f"Error creating context from {file_name}. Chatbot context remains: **{current_context_source}**."
        print(status_message)
        return status_message


def chat_responder_gradio_wrapper(message, chat_history):
    """Gradio wrapper for the chatbot response logic."""
    # Pass necessary global objects to the core logic function
    response = pipeline.get_chatbot_response(
        message=message,
        chat_memory=chat_memory,
        vector_store=current_chatbot_vector_store, # Use the current global store
        llm=llm,
        intent_prompt=intent_prompt,
        condense_prompt=CONDENSE_QUESTION_PROMPT,
        qa_prompt=qa_prompt
    )
    return response

def clear_chat_memory_gradio_wrapper():
    """Gradio wrapper to clear memory."""
    print("Clearing chat memory via Gradio button...")
    chat_memory.clear()
    print("Chat memory cleared.")
    return [] # Return empty list to clear ChatInterface display

def reset_context_to_default_gradio_wrapper():
     """Gradio wrapper to reset context to default."""
     global current_chatbot_vector_store, current_context_source
     print("Resetting context via Gradio button...")
     current_chatbot_vector_store = default_vector_store
     current_context_source = "Default Laptop Reviews"
     status_msg = f"Chatbot context reset to **{current_context_source}**."
     print(status_msg)
     return status_msg


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Product Review Intelligence Center")
    gr.Markdown("Analyze product reviews using Mistral-7B (Tab 1) or chat about reviews with customizable context (Tab 2).")

    with gr.Tabs():
        # --- TAB 1: BATCH ANALYZER ---
        with gr.TabItem("Batch Analyzer"):
            gr.Markdown("Paste reviews OR upload a file (.txt, .csv) to analyze them.")
            gr.Markdown("**Note:** This analysis does *not* affect the chatbot's context in Tab 2.")
            with gr.Row():
                with gr.Column(scale=2):
                     review_input_text_tab1 = gr.Textbox(lines=15, placeholder="Paste reviews here...", label="Reviews Text Input")
                     review_input_file_tab1 = gr.File(label="Upload Reviews File (.txt, .csv)", file_types=[".txt", ".csv"])
                with gr.Column(scale=1):
                    summary_output_tab1 = gr.Textbox(label="Overall Summary", lines=5, interactive=False)
                    aspect_output_tab1 = gr.Textbox(label="Key Aspects (Pros/Cons)", lines=5, interactive=False)
                    sentiment_output_tab1 = gr.Textbox(label="Sentiment Analysis", lines=5, interactive=False)
            analyze_button_tab1 = gr.Button("Analyze Reviews")
            gr.Examples(examples=[[example_batch, None]], inputs=[review_input_text_tab1, review_input_file_tab1], outputs=[summary_output_tab1, aspect_output_tab1, sentiment_output_tab1], fn=analyze_reviews_gradio_wrapper, cache_examples=False) # Use wrapper
            analyze_button_tab1.click(fn=analyze_reviews_gradio_wrapper, inputs=[review_input_text_tab1, review_input_file_tab1], outputs=[summary_output_tab1, aspect_output_tab1, sentiment_output_tab1]) # Use wrapper

        # --- TAB 2: CHAT ABOUT REVIEWS ---
        with gr.TabItem("Ask a Question (Chatbot)"):
            gr.Markdown("Ask specific questions about product reviews. Upload a file below to change the chatbot's knowledge base.")
            chatbot_status_display = gr.Markdown(f"Chatbot is currently using: **{current_context_source}**")
            with gr.Row():
                chatbot_context_file = gr.File(label="Upload Chatbot Context File (.txt, .csv)", file_types=[".txt", ".csv"], scale=3)
                update_context_button = gr.Button("Update Chatbot Context", scale=1)
            chatbot_interface = gr.ChatInterface(
                fn=chat_responder_gradio_wrapper, # Use wrapper
                examples=["How is the battery life?", "What about the screen?", "What are the complaints about connectivity?", "What is the capital of France?"],
                title="Review Chatbot"
            )
            with gr.Row():
                reset_memory_button = gr.Button("🔄 Reset Chat Memory")
                reset_context_button = gr.Button("🔄 Reset Chatbot Context to Default")
            # Link actions to wrapper functions
            update_context_button.click(fn=update_chatbot_context_gradio_wrapper, inputs=[chatbot_context_file], outputs=[chatbot_status_display])
            reset_memory_button.click(fn=clear_chat_memory_gradio_wrapper, inputs=None, outputs=[chatbot_interface])
            reset_context_button.click(fn=reset_context_to_default_gradio_wrapper, inputs=None, outputs=[chatbot_status_display])

# --- Launch Command ---
if __name__ == "__main__":
    chat_memory.clear() # Clear memory each time app starts
    demo.launch(debug=True)
