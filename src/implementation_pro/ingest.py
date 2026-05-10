import os
import glob
import boto3
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, OpenSearchVectorSearch, RequestsHttpConnection, AWSV4SignerAuth

MODEL = "gpt-4.1-nano"

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
opensearch_url = "https://zsa1frfc7uvp0cjmphv7.eu-west-3.aoss.amazonaws.com"

# Auth
session = boto3.Session()
credentials = session.get_credentials()
region = "eu-west-3"
awsauth = AWSV4SignerAuth(credentials, region, "aoss")

# # Test de connexion
# client = boto3.client("opensearchserverless", region_name="eu-west-3")

# # Data access policies
# print("=== DATA ACCESS POLICIES ===")
# policies = client.list_access_policies(type="data")
# for p in policies["accessPolicySummaries"]:
#     detail = client.get_access_policy(type="data", name=p["name"])
#     print(json.dumps(detail["accessPolicyDetail"]["policy"], indent=2))

# # ✅ list_security_policies (not list_access_policies)
# print("=== NETWORK POLICIES ===")
# net_policies = client.list_security_policies(type="network")
# for p in net_policies["securityPolicySummaries"]:
#     detail = client.get_security_policy(type="network", name=p["name"])
#     print(json.dumps(detail["securityPolicyDetail"]["policy"], indent=2))



def fetch_documents():
    
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*")) # folders = folders of company, contracts, employee, products
    documents = []
    for folder in folders: # each folder
        doc_type = os.path.basename(folder) # doc_type = ["company", "contracts", "employee", "products"]
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs: # each document
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):

    os_client = OpenSearch(
        hosts=[{"host": opensearch_url.replace("https://", ""), "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    
    # ✅ Supprimer l'index s'il existe
    try:
        if os_client.indices.exists(index="rag-vector-database"):
            os_client.indices.delete(index="rag-vector-database")
            print("🗑️ Index supprimé")
        else:
            print("ℹ️ Index n'existe pas encore")
    except Exception as e:
        print(f"⚠️ Erreur suppression : {e}")

    vectorstore = OpenSearchVectorSearch.from_documents(
        documents=chunks,   # obligatoire
        embedding=embeddings, # obligatoire
        opensearch_url=opensearch_url, # obligatoire
        index_name="rag-vector-database", # obligatoire
        http_auth=awsauth, # obligatoire
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        is_aoss=True,
        timeout=60,       # obligatoire (si timeout trop court, erreur 408)
    )
    return vectorstore

if __name__ == "__main__":
    documents = fetch_documents()
    print(f"📄 Documents récupérés : {len(documents)}")
    
    chunks = create_chunks(documents)
    print(f"🔪 Chunks créés : {len(chunks)}")
    
    create_embeddings(chunks)
    print("✅ Ingestion complete")