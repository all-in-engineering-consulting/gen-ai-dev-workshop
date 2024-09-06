from langchain_community.embeddings import FastEmbedEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
import pickle
import re

def extract_fenced_text(text, fence="```"):
    # print(f"[extract_fenced_text] input: text: {text}")

    pattern = f"{re.escape(fence)}(?:\w+)?\s*(.*?){re.escape(fence)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text_return = match.group(1).strip()
        # print(f"[extract_fenced_text] output: {text_return}")
        return text_return
    
    # print(f"[extract_fenced_text] output: None")
    return text


def faiss_index(chunks=None, embedding_model=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"), index_name="faiss_index"):
    """
    Set up FAISS database for storing document embeddings or return existing one.

    Takes chunks of documents and embedding model, and creates
    FAISS vector store for efficient similarity search. If chunks
    are not provided, a filename for the FAISS store should be provided.

    Args:
        chunks (list, optional): Chunks of documents that should be embedded and stored.
        embedding_model (FastEmbedEmbeddings): Embedding model that should be used for vectorization.
        faiss_store_filename (str, optional): Filename for the FAISS store if chunks are not provided.

    Returns:
        FAISS: FAISS vector store populated with document embeddings.
    """
    if chunks:
        # Create FAISS vector store from chunks
        faiss_store = FAISS.from_documents(chunks, embedding_model)
        # Save the FAISS store to a local file
        faiss_store.save_local(index_name)
        return faiss_store
    else:
        # Load FAISS index and additional data from the provided filename
        new_vector_store = FAISS.load_local(index_name, embedding_model, allow_dangerous_deserialization=True)
        return new_vector_store