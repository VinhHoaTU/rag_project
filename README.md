# 📥 Enterprise RAG Chatbot

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system designed to act as an internal knowledge assistant (client contracts, HR directories, product documentation)

---

### 🏛️ Architectural Evolution & Strategy
To demonstrate advanced cloud-native patterns and dual-environment deployment, this project was architected to support two distinct operational modes:
1. **Local & Zero-Config Evaluation:** Local testing without third-party API dependencies, the system can be fully run locally using an embedded **ChromaDB** instance.
2. **Production & Enterprise Scale:** Initially deployed using an **AWS S3** data lake coupled with **AWS OpenSearch Serverless** for high-throughput hybrid retrieval. The cloud vector storage was later migrated to **Pinecone** as part of a **FinOps (Cost Optimization)** initiative, significantly reducing operational overhead and cloud spend while maintaining sub-second retrieval performance.
---

## 🎯 The Mission

This project serves as an end-to-end demonstration of Production AI Engineering, structured around 6 core technical pillars:

* **Data Lake Ingestion:** AWS S3
* **Semantic Chunking:** Recursive
* **Vector Embedding:** text-embedding-3-large
* **Vector Database Management:** ChomaDB (AWS OpenSearch and Pinecone for cloud version).
* **Dual-Stage Evaluation (LLM-as-a-Judge):** Benchmarked via MRR (Mean Reciprocal Rank) and nDCG (Normalized Discounted Cumulative Gain) metrics.
    * *Generation Quality:* Automated auditing across Accuracy (Faithfulness), Completeness and Relevance.
* **Application Deployment:** Gradio.


---
## ⚙️ Setup Instructions

This project uses **`uv`** for ultra-fast dependency management. Make sure you have it installed:

```bash
curl -fsSL https://astral.sh/uv/install.sh
```

### 1. Clone the Repository

```bash
git clone https://github.com/VinhHoaTU/rag_project.git
cd rag_project
```

### 2. Environment Setup

Create a `.env` file at the root of the project and add your API keys:

```
python -m venv .venv
source .venv/bin/activate
OPENAI_API_KEY="your_openai_api_key"
# Required only if you switch to the cloud deployment mode:
PINECONE_API_KEY="your_pinecone_api_key"
```

### 3. Install Dependencies & Build Local Database

Sync your virtual environment and run the ingestion pipeline to parse local documents, generate embeddings, and build your database:

```bash
# Sync dependencies and automatically create the virtual environment (.venv)
uv sync

# Run the ingestion script
uv run src/implementation/ingest.py
```

### 4. Launch the Chatbot Application

Run the Gradio application interface locally:

```bash
# Launch the application interface
uv run src/app.py
```

### 5. Launch the Chatbot Evaluation (optional)

Run the Gradio application to evaluate the performance of the model's retrivals and answers:

```bash
# Launch the application interface
uv run src/evaluator.py
```