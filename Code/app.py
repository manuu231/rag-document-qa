import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile

# ── We use HuggingFaceHub as LLM (free, no OpenAI key needed)
# ── Embeddings via sentence-transformers (free, runs on CPU)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

vectorstore = None  # global store after PDF upload


def process_pdf(pdf_file, hf_token):
    """Load PDF, chunk it, embed it, store in FAISS."""
    global vectorstore

    if pdf_file is None:
        return "Please upload a PDF first."
    if not hf_token or len(hf_token.strip()) < 10:
        return "Please enter your HuggingFace API token."

    # Set token for HuggingFaceHub
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token.strip()

    try:
        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file)
            tmp_path = tmp.name

        # 1. Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 2. Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # 3. Embed + store in FAISS (CPU, free)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return f"PDF processed successfully. {len(chunks)} chunks indexed. You can now ask questions below."

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def answer_question(question, hf_token):
    """Retrieve relevant chunks and generate answer."""
    global vectorstore

    if vectorstore is None:
        return "Please upload and process a PDF first."
    if not question.strip():
        return "Please enter a question."
    if not hf_token or len(hf_token.strip()) < 10:
        return "Please enter your HuggingFace API token."

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token.strip()

    try:
        # Free LLM via HuggingFace Inference API
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0.3, "max_new_tokens": 512},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False,
        )

        result = qa_chain.run(question)
        return result.strip()

    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ── Gradio UI
with gr.Blocks(title="RAG Document Q&A | Manpreet Kaur") as demo:

    gr.Markdown("""
    # RAG-Powered Document Q&A System
    **Built with:** LangChain · FAISS · Mistral-7B · HuggingFace · Gradio  
    **By:** Manpreet Kaur | [LinkedIn](https://linkedin.com/in/manpreetkaurmahal) | [GitHub](https://github.com/manuu231)

    Upload any PDF, then ask questions about its content. Powered by Retrieval-Augmented Generation (RAG).
    """)

    with gr.Row():
        hf_token = gr.Textbox(
            label="HuggingFace API Token (free at huggingface.co/settings/tokens)",
            placeholder="hf_...",
            type="password",
            scale=2
        )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="binary"
            )
            process_btn = gr.Button("Process PDF", variant="primary")
            process_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about your document",
                placeholder="e.g. What are the main findings of this paper?",
                lines=3
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=8
            )

    gr.Markdown("""
    ---
    **How it works:**
    1. PDF is loaded and split into 500-token chunks
    2. Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2`
    3. Chunks are stored in a FAISS vector index
    4. Your question is embedded and top-4 relevant chunks are retrieved
    5. Mistral-7B generates an answer grounded in those chunks
    """)

    # Wire up buttons
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, hf_token],
        outputs=process_status
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, hf_token],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch()
