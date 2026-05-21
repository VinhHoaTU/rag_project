import os
import glob
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

MODEL = "gpt-4.1-nano"
index_name = "insurellm-rag"

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""
    documents = []

    base_path = Path(KNOWLEDGE_BASE_PATH) 

    for folder in base_path.iterdir():
        if folder.is_dir(): 
            doc_type = folder.name
            for file in folder.rglob("*.md"):
                with open(file, "r", encoding="utf-8") as f:
                    # On crée un vrai objet Document de LangChain
                    doc = Document(
                        page_content=f.read(), # Le texte va obligatoirement ici
                        metadata={             # Tout le reste va dans le dictionnaire metadata
                            "type": doc_type, 
                            "source": file.as_posix()
                        }
                    )
                    documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks, # Tes morceaux de texte existants
        embedding=embeddings,
        index_name=index_name
    )     
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    print(f"Documents récupérés : {len(documents)}")
    
    chunks = create_chunks(documents)
    print(f"Chunks créés : {len(chunks)}")
    
    create_embeddings(chunks)
    print("Ingestion complete")
