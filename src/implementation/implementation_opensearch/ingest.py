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
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_core.documents import Document


MODEL = "gpt-4.1-nano"

BUCKET_NAME = "rag-insurellm-bucket"
S3_PREFIX = "base_de_connaissance/"  

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

load_dotenv(override=True)

opensearch_url = os.getenv("OPENSEARCH_URL")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Auth
session = boto3.Session()
credentials = session.get_credentials()
region = "eu-west-3"
awsauth = AWSV4SignerAuth(credentials, region, "aoss")

s3 = boto3.client("s3", region_name="eu-west-3")


def fetch_documents_s3():
    documents = []
    
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)
    
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            
            if not key.endswith(".md"):
                continue
            
            # Lire directement le contenu sans unstructured
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            content = response["Body"].read().decode("utf-8")
            
            # Extraire doc_type depuis le chemin
            relative = key.replace(S3_PREFIX, "")
            doc_type = relative.split("/")[0]
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"s3://{BUCKET_NAME}/{key}",
                    "doc_type": doc_type
                }
            )
            documents.append(doc)
            print(f"Chargé : {key} (doc_type={doc_type})")
    
    print(f"\n {len(documents)} documents chargés depuis S3")
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
    
    # Supprimer l'index s'il existe
    try:
        if os_client.indices.exists(index="rag-vector-database"):
            os_client.indices.delete(index="rag-vector-database")
            print(" Index supprimé")
        else:
            print("Index n'existe pas encore")
    except Exception as e:
        print(f" Erreur suppression : {e}")

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
    documents = fetch_documents_s3()
    print(f"Documents récupérés : {len(documents)}")
    
    chunks = create_chunks(documents)
    print(f"Chunks créés : {len(chunks)}")
    
    create_embeddings(chunks)
    print("Ingestion complete")
