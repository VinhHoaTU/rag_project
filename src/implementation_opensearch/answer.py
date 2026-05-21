import boto3, os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
from langchain_pinecone import PineconeVectorStore

MODEL = "gpt-4.1-nano"

load_dotenv(override=True)

opensearch_url = os.getenv("OPENSEARCH_URL")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

RETRIEVAL_K = 4      # docs récupérés par variante de requête
RERANKER_TOP_K = 4    # docs conservés après reranking

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# Auth
session = boto3.Session()
credentials = session.get_credentials()
region = "eu-west-3"
awsauth = AWSV4SignerAuth(credentials, region, "aoss")

# Connexion à l'index existant (pas de from_documents)
vectorstore = OpenSearchVectorSearch(
    opensearch_url=opensearch_url,
    index_name="rag-vector-database",
    embedding_function=embeddings,  # nécessaire pour embedder les questions
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    is_aoss=True,
)


llm = ChatOpenAI(temperature=0, model_name=MODEL)

# ── Step 1 : base retriever
# fetch_k large pour donner assez de candidats au MMR
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVAL_K, "fetch_k": 30}
)

# ── Step 2 : MultiQueryRetriever 
# chaque variante fait une recherche: résultats fusionnés + dédupliqués
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# ── Step 3 : Cross-encoder reranker 
# Rerank tous les docs récupérés par MultiQuery
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=RERANKER_TOP_K)

# ── Step 4 : Pipeline complet 
# ContextualCompressionRetriever branche le reranker sur le MultiQuery
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever,
)

def fetch_context(question: str) -> list[Document]:
    """
    Full pipeline :
    question → MultiQuery (3 variantes) → retrieval MMR (k=10 par variante)
             → fusion + déduplication → reranker → top 4 docs
    """
    return compression_retriever.invoke(question)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs


if __name__ == "__main__":
    answer, docs = answer_question("qui est Alex Thomson")
    print("Réponse :", answer)
    print(f"\n{len(docs)} documents utilisés comme contexte")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content[:150]}...")
