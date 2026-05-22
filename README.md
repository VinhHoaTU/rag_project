# 📥 Enterprise RAG Chatbot with AWS OpenSearch & Gradio

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system designed to act as an internal knowledge assistant. This chatbot securely ingests, indexes, and retrieves company data—including company overviews, client contracts, employee directories, and product documentation—to provide accurate, context-aware answers to user queries.

---

## 🚀 Key Features

*   **Multi-Domain Knowledge Base:** Seamlessly handles diverse document types (Corporate culture, legal client contracts, HR directories, and technical product specs).
*   **Hybrid Storage & Search: Initially architected with a knowledge base hosted on **AWS S3** and indexed via **AWS OpenSearch** for hybrid vector and keyword retrieval. The system was subsequently migrated to **Pinecone** to optimize infrastructure costs, reduce operational overhead, and maintain high-performance retrieval.
*   **Granular Metadata Tracking:** Automated extraction of document attributes (e.g., `doc_type`) to filter and contextualize search results.
*   **Rigorous Retrieval Evaluation:** Search performance is quantitatively measured using **MRR**, **nDCG**, and **Keyword Coverage**.
*   **LLM-as-a-Judge Evaluation:** Generative answers are automatically audited for quality across **Accuracy**, **Completeness**, and **Relevance**.
*   **Interactive UI:** Built with **Gradio** for rapid internal prototyping and a seamless user chat experience.

---

## 🛠️ Tech Stack

*   **Orchestration:** LangChain / Python
*   **Cloud Storage:** AWS S3
*   **Vector Database:** AWS OpenSearch/ Pinecone
*   **User Interface:** Gradio

---

## 📐 Architecture & Workflow

1.  **Ingestion:** Raw company documents (`.md`) are uploaded to designated prefixes in an **AWS S3** bucket (`documents/`).
2.  **Processing & Embedding:** A data pipeline fetches files, extracts metadata (`doc_type` based on directory structures), splits them into semantic chunks using a text splitter, and generates vector embeddings.
3.  **Indexing:** Chunked data and corresponding metadata are upserted into **AWS OpenSearch**/ **Pinecone**.
4.  **Retrieval & Generation:** When a user asks a question via the **Gradio UI**, OpenSearch retrieves the most relevant chunks. The LLM synthesizes these chunks into a final answer.

---

## 📊 Evaluation Framework

To transition from a prototype to a production-ready system, this project implements a dual-stage evaluation matrix:

### 1. Retrieval Metrics (Search Quality)
*   **MRR (Mean Reciprocal Rank):** Measures how high up the first relevant document appears in the search results.
*   **nDCG (Normalized Discounted Cumulative Gain):** Evaluates the grading quality and order of the retrieved documents.
*   **Keyword Coverage:** Ensures critical company terminology and domain-specific jargon are successfully captured during retrieval.

### 2. Generation Metrics (LLM-as-a-Judge)
We leverage a powerful LLM to evaluate final answers based on a 3-axis framework:
*   **Accuracy (Faithfulness):** Is the answer factually grounded *only* in the retrieved context, avoiding hallucinations?
*   **Completeness:** Does the chatbot address all parts of the user's prompt without omitting essential corporate details?
*   **Relevance:** Is the response direct and free of fluff or off-topic information?

---
