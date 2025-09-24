# Hybrid RAG Chatbot

A robust, production-grade Retrieval-Augmented Generation (RAG) chatbot framework. This project combines hybrid document retrieval (dense, sparse, and RRF fusion) with LLM-powered conversational QA, intent-routing, logging, and evaluation—all orchestrated through an interactive Streamlit interface.

---

##  Features

- **Unified Knowledge Base**: Ingest, chunk, and embed documents (PDF, TXT, DOCX) with full page/chunk metadata, supporting multi-document and filtered retrieval.
- **Hybrid Retrieval**: Vector (dense), full-text (sparse), and Reciprocal Rank Fusion (RRF) for state-of-the-art accuracy.
- **LLM Routing & Response**: Determines when to retrieve vs. chitchat/handoff using intent classification; generates answers grounded ONLY in retrieved context.
- **Session Logging & Metrics**: Logs all queries, responses, latency, retrieval info, and conversational summaries for analytics.
- **Configurable Chat UI**: Select retrieval mode, top-k, LLM model, and upload/choose docs.
- **Evaluation & Explainability**: Metrics tabs for latency, retrieval stats, retrieved chunk display, and downloadable logs.
- **Security**: Strict context grounding—never hallucinates answers outside the provided KB.

---

##  Project Structure & Key Components

### 1. `app.py` — **Streamlit Main App**
- **Purpose**: Orchestrates the UI, user session, document selection/upload, and ties together retrieval, LLM QA, and logging.
- **Why?**: Offers an interactive, modular interface for both end-users and developers to experiment, upload new docs, and visualize metrics.

### 2. `preprocess/document_loader.py` — **Document Loader & Preprocessing**
- **Purpose**: Loads files (PDF, TXT, DOCX), extracts and cleans text, and (for PDFs) maps each chunk to its source page.
- **Why?**: Clean, normalized input is essential for high-quality retrieval and avoids garbage-in/garbage-out issues.

### 3. `preprocess/chunker.py` — **Chunking**
- **Purpose**: Splits text into overlapping, fixed-size "chunks" using `langchain`'s recursive splitter.
- **Why?**  
    - **Chunking** is critical because LLMs and vector databases have context window limits. Chunking preserves semantic cohesion, ensures answerability, and supports precise retrieval.
    - Overlap ensures that answers spanning chunk boundaries are still retrievable.

### 4. `embeddings/embedder.py` — **Chunk Embedding**
- **Purpose**: Encodes each chunk into a dense embedding using `sentence-transformers`.
- **Why?**: Enables semantic search—retrieving relevant passages by meaning, not just keywords.

### 5. `db/ingestor.py` — **Ingestor**
- **Purpose**: Handles document indexing: loads, chunks, embeds, and stores all chunk metadata in LanceDB tables.
- **Why?**: Centralizes ingestion with deduplication, versioning, and rich chunk metadata (page, ordinal, hash, etc.).

### 6. `retrieval/retriever.py` — **Retriever**
- **Purpose**: Hybrid retrieval (dense, sparse, or both) with strict document filtering, RRF fusion, and full timing/metrics.
- **Why?**: Maximizes recall and precision by combining strengths of vector and keyword search, crucial for real-world knowledge bases.

### 7. `llm/conversational.py` — **Conversational LLM QA**
- **Purpose**: Generates answers strictly from retrieved chunks and conversation context using OpenAI or Groq LLMs.
- **Why?**: Prevents hallucination—answers are only given if they can be directly supported by retrieved context. Responds with "I do not have knowledge about this" otherwise.

### 8. `llm/summarizer.py` — **Rolling Conversation Summarizer**
- **Purpose**: Updates a short, rolling summary of the session for context-aware answering.
- **Why?**: Maintains context for multi-turn conversations without exceeding LLM input limits.

### 9. `db/session_logger.py` — **Session Logging & Analytics**
- **Purpose**: Logs all queries, answers, metrics, and session summaries to LanceDB for analytics, debugging, and traceability.
- **Why?**: Essential for monitoring, auditing, and continuous improvement of the system.

### 10. `router.py` — **Intent Routing & Decision Logic**
- **Purpose**: Rule-based and LLM-based intent detection; routes between retrieval QA, chitchat, handoff, safety, and OOS responses.
- **Why?**: Improves reliability by only triggering retrieval and LLM QA for in-scope, safe, and answerable queries.

---

##  Getting Started

### 1. **Install Requirements**
```bash
pip install -r requirements.txt
```
- Requires: `streamlit`, `lancedb`, `sentence-transformers`, `langchain`, `PyPDF2`, `python-docx`, `ftfy`, `psutil`, `dotenv`, and LLM API access (OpenAI, Groq).

### 2. **Set Up API Keys**
- Create a `.env` file with your keys:
    ```
    OPENAI_API_KEY=your-openai-key
    GROQ_API_KEY=your-groq-key
    ```

### 3. **Run the App**
```bash
streamlit run app.py
```

### 4. **Upload Documents & Start Chatting**
- Use the UI to upload PDF/TXT/DOCX files or select from the knowledge base.
- Choose retrieval/LLM modes, ask questions, and explore logs/metrics.

---

##  Why RAG? Why Hybrid? Why All This Engineering?

- **RAG** (Retrieval-Augmented Generation) ensures factual, up-to-date, and auditable answers by grounding LLMs in your actual knowledge base.
- **Chunking** ensures retrieval precision and keeps answers tightly linked to their source context.
- **Hybrid Retrieval** (Dense + Sparse + RRF) overcomes limitations of either approach for real-world, messy documents.
- **Strict Routing & Logging** provide safety, explainability, and debuggability—vital for enterprise and banking domains.

---


