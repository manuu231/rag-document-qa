# RAG-Powered Document Q&A System

Upload any PDF and ask questions about it using RAG.

**Built with:** LangChain · FAISS · Mistral-7B · HuggingFace · Gradio

**Live Demo:** https://huggingface.co/spaces/Manpreet02/rag-document-qa

## How it works
1. PDF is chunked into 500-token segments
2. Chunks are embedded using sentence-transformers
3. Stored in FAISS vector index
4. Questions retrieve top-4 relevant chunks
5. Mistral-7B generates grounded answers
