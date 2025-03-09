import logging

from dotenv import dotenv_values, find_dotenv
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from src.pipeline import create_docs_first_process_pipeline
from src.services.pdf_service import convert_pdf_to_document

ollama_embed_model = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_EMBED_MODEL')
ollama_url = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_URL')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

async def get_documents(vdb, to_be_converted_text, generation_kwargs_config=None):
    """
    This function returns a list of stored vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored vectorized documents
    """
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    try:
        text_embedder = OllamaTextEmbedder(model=ollama_embed_model, url=ollama_url, generation_kwargs=generation_kwargs_config)
        embedded_text = text_embedder.run(text=to_be_converted_text)
        embedding_retriever = QdrantEmbeddingRetriever(document_store=vdb)
        retrieved_documents = embedding_retriever.run(query_embedding=embedded_text['embedding'])
        vdb.client.close()
        return retrieved_documents
    except Exception as e:
        raise e

async def get_all_documents(vdb):
    """
    This function returns a list of stored vectorized documents from the document store.

    :return: A list of stored vectorized documents
    """
    try:
        result = vdb.filter_documents()
        vdb.client.close()
        return result
    except Exception as e:
        logger.error(f"Failed to retrieve documents from document store: {e}")
        raise e

async def create_vectorized_documents(vdb, files, generation_kwargs_config=None):
    """
    This function stores vectorized documents in Qdrant document store.
    :return:
    """
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    documents = []
    for file_object in files:
        converted_documents = convert_pdf_to_document(file_object["file_path"], file_object["authors"])
        pipeline = create_docs_first_process_pipeline()
        vectorized_documents = pipeline.run(
            data={"document_cleaner": {"documents": converted_documents}},
        )
        documents += vectorized_documents['document_embedder']['documents']
    try:
        count = vdb.write_documents(documents=documents, policy=DuplicatePolicy.SKIP)
        vdb.client.close()
        return {"count": count}
    except Exception as e:
        logger.error(f"Failed to write documents to Qdrant document store: {e}")
        raise e

async def delete_documents(vdb):
    """
    This function deletes all stored documents from the document store.
    :param vdb:
    :return:
    """
    try:
        vdb.recreate_collection(
            collection_name=vdb.index,
            distance=vdb.get_distance(similarity=vdb.similarity),
            embedding_dim=vdb.embedding_dim
        )
        vdb.client.close()
    except Exception as e:
        logger.error(f"Failed to delete documents from document store: {e}")
        raise e