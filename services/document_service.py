import logging

from dotenv import dotenv_values, find_dotenv
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from services.pdf_service import convert_pdf_to_document

ollama_embed_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_EMBED_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def get_documents(vdb, to_be_converted_text, generation_kwargs_config=None):
    """
    This function returns a list of stored vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored vectorized documents
    """
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    text_embedder = OllamaTextEmbedder(model=ollama_embed_model, url=ollama_url, generation_kwargs=generation_kwargs_config)
    embedded_text = text_embedder.run(text=to_be_converted_text)
    embedding_retriever = QdrantEmbeddingRetriever(document_store=vdb)
    retrieved_documents = embedding_retriever.run(query_embedding=embedded_text['embedding'])

    return retrieved_documents

def create_vectorized_documents(vdb, file, generation_kwargs_config=None):
    """
    This function stores vectorized documents in Qdrant document store.
    :return:
    """
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    documents = convert_pdf_to_document(file["file_path"], file["authors"])
    document_cleaner = DocumentCleaner(remove_repeated_substrings=True)
    cleaned_documents = document_cleaner.run(documents=documents)
    document_splitter = DocumentSplitter(split_by="word", split_length=400)
    split_documents = document_splitter.run(documents=cleaned_documents['documents'])
    document_embedder = OllamaDocumentEmbedder(model=ollama_embed_model, url=ollama_url, generation_kwargs=generation_kwargs_config)
    vectorized_documents = document_embedder.run(documents=split_documents['documents'])

    try:
        return {"count": vdb.write_documents(documents=vectorized_documents['documents'], policy=DuplicatePolicy.SKIP)}
    except Exception as e:
        logger.error(f"Failed to write documents to Qdrant document store: {e}")