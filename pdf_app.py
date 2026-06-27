import gradio as gr
import ollama
from pypdf import PdfReader
import numpy as np

# Global in-memory storage for our local RAG system
PDF_CHUNKS = []
CHUNK_EMBEDDINGS = []

def extract_and_chunk_pdf(file):
    """Extracts text from a massive PDF page-by-page and splits it into chunks."""
    global PDF_CHUNKS, CHUNK_EMBEDDINGS
    PDF_CHUNKS = []
    CHUNK_EMBEDDINGS = []
    
    if file is None:
        return "❌ Please upload a valid PDF file first."
        
    try:
        reader = PdfReader(file.name)
        full_text = ""
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        # Splitting text into chunks
        words = full_text.split()
        chunk_size = 600
        overlap = 150
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        PDF_CHUNKS = chunks
        total_chunks = len(PDF_CHUNKS)
        
        if total_chunks == 0:
            return "⚠️ PDF processing completed, but no readable text was found."

        # Generate embeddings for quick mathematical vector searches
        for chunk in PDF_CHUNKS:
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            CHUNK_EMBEDDINGS.append(response["embedding"])
            
        return f"✅ Successfully processed document!\n• Total Pages: {len(reader.pages)}\n• Total Text Chunks: {total_chunks}"
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def get_macro_context(max_chunks=4):
    """Samples key segments from across the document for global summaries/notes."""
    if len(PDF_CHUNKS) <= max_chunks:
        return "\n\n".join(PDF_CHUNKS)
    step = len(PDF_CHUNKS) // max_chunks
    sampled = [PDF_CHUNKS[i * step] for i in range(max_chunks)]
    return "\n--- Document Section ---\n".join(sampled)

def generate_notes():
    if not PDF_CHUNKS:
        return "### ❌ Error\nPlease upload and process a PDF document first!"
    
    context = get_macro_context()
    prompt = f"Based on these representative sections of the document, generate detailed, structured study notes:\n\n{context}"
    
    try:
        response = ollama.generate(model="llama3.2:1b", prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"### ❌ Error calling Ollama\n{str(e)}"

def generate_mindmap():
    if not PDF_CHUNKS:
        return "### ❌ Error\nPlease upload and process a PDF document first!"
    
    context = get_macro_context()
    prompt = (
        f"Analyze this text context and generate a visual hierarchy mindmap. "
        f"You must output ONLY a valid Mermaid.js diagram block starting with 'graph TD'. "
        f"Do not include any conversational text outside the code block.\n\nText Context:\n{context}"
    )
    
    try:
        response = ollama.generate(model="llama3.2:1b", prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"### ❌ Error calling Ollama\n{str(e)}"

def cosine_similarity(a, b):
    """Calculates mathematical vector similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def ask_anything(message, history):
    if not PDF_CHUNKS or not CHUNK_EMBEDDINGS:
        return "Please upload and index a PDF file before asking questions!"
        
    try:
        # 1. Turn the user's question into a vector embedding
        query_response = ollama.embeddings(model="nomic-embed-text", prompt=message)
        query_vector = query_response["embedding"]
        
        # 2. Rank chunks by similarity using numpy and isolate the 3 most relevant sections
        similarities = [cosine_similarity(query_vector, chunk_emb) for chunk_emb in CHUNK_EMBEDDINGS]
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        # 3. Pull out the text of those top matching chunks
        retrieved_context = "\n\n".join([PDF_CHUNKS[idx] for idx in top_indices])
        
        # 4. Supply context to the local LLM
        system_instructions = f"Answer the user's question accurately using only the relevant document snippets provided below.\n\nContext:\n{retrieved_context}"
        response = ollama.generate(model="llama3.2:1b", prompt=f"{system_instructions}\n\nUser Question: {message}")
        
        return response["response"]
        
    except Exception as e:
        return f"An error occurred during local generation: {str(e)}"

# --- Gradio UI Engine Layout ---
# FIX: Removed the theme argument from gr.Blocks initialization
with gr.Blocks(title="Local PDF Intelligence Suite") as demo:
    gr.Markdown("# 📚 Local Massive PDF Intelligence Suite")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload Massive PDF", file_types=[".pdf"])
            process_btn = gr.Button("⚙️ Index & Analyze Document", variant="primary")
            status_output = gr.Textbox(label="System Processing Log", interactive=False, lines=5)
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📝 Structured Notes"):
                    notes_btn = gr.Button("✨ Generate Study Notes", variant="secondary")
                    notes_display = gr.Markdown()
                    
                with gr.TabItem("🗺️ Visual Mindmap"):
                    mindmap_btn = gr.Button("🌿 Render Hierarchical Mindmap", variant="secondary")
                    mindmap_display = gr.Markdown()
                    
                with gr.TabItem("💬 Ask Anything (Local RAG)"):
                    # FIX: Removed type="messages" from ChatInterface
                    chatbot = gr.ChatInterface(fn=ask_anything)

    # Wire interface interactive triggers
    process_btn.click(fn=extract_and_chunk_pdf, inputs=[pdf_input], outputs=[status_output])
    notes_btn.click(fn=generate_notes, inputs=[], outputs=[notes_display])
    mindmap_btn.click(fn=generate_mindmap, inputs=[], outputs=[mindmap_display])

if __name__ == "__main__":
    # FIX: Moved the theme setting down to the launch command
    demo.launch(theme=gr.themes.Soft())