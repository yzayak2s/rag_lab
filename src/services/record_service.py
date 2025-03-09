import logging
from dotenv import dotenv_values, find_dotenv
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.components.CustomXLSXToDocuments import CustomXLSXToDocuments
from src.pipeline import create_records_pipeline

ollama_embed_model = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_EMBED_MODEL')
ollama_url = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_URL')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

async def create_records(vdb: QdrantDocumentStore, file, generation_kwargs_config=None):
    """
    This function stores records as vectorized documents in Qdrant document store.
    :return:
    """

    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    custom_xlsx_to_docs_component = CustomXLSXToDocuments()
    documents = custom_xlsx_to_docs_component.run(file=file)

    try:
        pipeline = create_records_pipeline(vdb)
        result = pipeline.run(
            data={"document_embedder": {"documents": documents}},
        )
        vdb.client.close()
        return result['document_writer']
    except Exception as e:
        logger.error(f"Failed to write records as documents to Qdrant document store: {e}")

async def get_records(vdb: QdrantDocumentStore, to_be_converted_text, generation_kwargs_config=None):
    """
    This function returns a list of stored records as vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored records as vectorized documents
    """

    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    try:
        text_embedder = OllamaTextEmbedder(model=ollama_embed_model, url=ollama_url, generation_kwargs=generation_kwargs_config)
        embedded_text = text_embedder.run(text=to_be_converted_text)
        embedding_retriever = QdrantEmbeddingRetriever (document_store=vdb)
        retrieved_documents = embedding_retriever.run(query_embedding=embedded_text['embedding'])
        vdb.client.close()
    except Exception as e:
        logger.error(f"Failed to retrieve records from document store: {e}")
        raise e

    return retrieved_documents

async def get_all_records(vdb: QdrantDocumentStore):
    """
    This function returns a list of stored records as vectorized documents from the document store.

    :return: A list of stored records as vectorized documents
    """
    try:
        result = vdb.filter_documents()
        vdb.client.close()
        return result
    except Exception as e:
        logger.error(f"Failed to retrieve records from document store: {e}")
        raise e

async def delete_records(vdb: QdrantDocumentStore):
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