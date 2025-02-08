import logging

from dotenv import dotenv_values, find_dotenv
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

ollama_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def get_documents(vdb, to_be_converted_text):
    """
    This function returns a list of stored vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored vectorized documents
    """
    text_embedder = OllamaTextEmbedder(model=ollama_model, url=ollama_url)
    embedded_text = text_embedder.run(text=to_be_converted_text)
    embedding_retriever = QdrantEmbeddingRetriever(document_store=vdb)
    retrieved_documents = embedding_retriever.run(query_embedding=embedded_text['embedding'])

    return retrieved_documents

def create_vectorized_documents(vdb, vectorized_documents):
    """
    This function stores vectorized documents in Qdrant document store.
    :return:
    """

    try:
        return {"count": vdb.write_documents(documents=vectorized_documents, policy=DuplicatePolicy.SKIP)}
    except Exception as e:
        logger.error(f"Failed to write documents to Qdrant document store: {e}")