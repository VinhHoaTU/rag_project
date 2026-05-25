# 📥 Enterprise RAG Chatbot

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system designed to act as an internal knowledge assistant. This chatbot securely ingests, indexes, and retrieves sensitive corporate data (client contracts, HR directories, product documentation) to provide accurate, context-aware answers via an interactive interface.

### 🏛️ Architectural Evolution & Strategy
To demonstrate advanced cloud-native patterns and dual-environment deployment, this project was architected to support two distinct operational modes:
1. **Local & Zero-Config Evaluation:** Local testing without third-party API dependencies, the system can be fully run locally using an embedded **ChromaDB** instance.
2. **Production & Enterprise Scale:** Initially deployed using an **AWS S3** data lake coupled with **AWS OpenSearch Serverless** for high-throughput hybrid retrieval. The cloud vector storage was later migrated to **Pinecone** as part of a **FinOps (Cost Optimization)** initiative, significantly reducing operational overhead and cloud spend while maintaining sub-second retrieval performance.
---

## 🎯 The Mission

This project serves as an end-to-end demonstration of Production AI Engineering, structured around 6 core technical pillars:

* **Data Lake Ingestion:** Building an automated pipeline to ingest multi-domain corporate documents (`.md`) from a structured file system.
* **Semantic Chunking:** Implementing smart text-splitting to maintain context boundaries and automate granular metadata tracking (e.g., `doc_type`, `source`).
* **Vector Embedding:** Transforming textual data into high-dimensional vectors (3072-dim) using OpenAI embedding models.
* **Vector Database Management:** Indexing, configuration, and scaling vector upserts within **ChomaDB** (**AWS OpenSearch** and **Pinecone** in cloud version).
* **Dual-Stage Evaluation (LLM-as-a-Judge):** * *Retrieval Quality:* Benchmarked via **MRR** (Mean Reciprocal Rank) and **nDCG** (Normalized Discounted Cumulative Gain) metrics.
    * *Generation Quality:* Automated auditing across **Accuracy** (Faithfulness), **Completeness**, and **Relevance**.
* **Application Deployment:** Designing and serving a production-ready chatbot web interface using **Gradio**.

---

## 🛠️ Tech Stack

* **Orchestration & Framework:** LangChain / Python
* **Package Management:** `uv` (Fast Python package installer)
* **Vector Database:** Pinecone / AWS OpenSearch
* **LLM & Embeddings:** OpenAI (`gpt-4o` / `text-embedding-3-large`)
* **User Interface:** Gradio

---

## ⚙️ Setup Instructions

This project uses **`uv`** for ultra-fast dependency management. Make sure you have it installed (`curl -fsSL https://astral.sh/uv/install.sh`).

```bash
1. Clone the Repository
git clone [https://github.com/VinhHoaTU/rag_project.git](https://github.com/VinhHoaTU/rag_project.git)
cd rag_project

2. Environment Setup
python -m venv .venv
OPENAI_API_KEY="your_openai_api_key"

3. Install Dependencies & Build Local DatabaseBash
uv sync
uv run src/implementation/ingest.py

4. Launch the Chatbot Application
