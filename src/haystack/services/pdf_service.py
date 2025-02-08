from haystack.components.converters import PyPDFToDocument
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.document_stores.types import DuplicatePolicy

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def convert_pdf_to_document(file_path: str, authors: str):
    """
    This function converts a PDF to a document-type object in list.

    :param file_path:
    :param authors:
    :returns: `documents`: A list of converted documents
    """

    # Initialize PyPDFToDocument converter
    converter = PyPDFToDocument()

    # Convert PDFs to documents
    results = converter.run(sources=[file_path], meta=[{"authors": authors}])

    return results["documents"]

def store_vectorized_documents(vectorized_documents):
    """
    This function stores vectorized documents in Qdrant document store.
    :return:
    """
    # Initialize QdrantDocumentStore with the specified path
    document_store = QdrantDocumentStore(
        path="qdrant/storage_local",
        index="Document",
        embedding_dim=3072,
        # recreate_index=True,
        # hnsw_config={"m": 16, "ef_construct": 64} # Optional
    )

    try:
        return {"count": document_store.write_documents(documents=vectorized_documents, policy=DuplicatePolicy.SKIP)}
    except Exception as e:
        logger.error(f"Failed to write documents to Qdrant document store: {e}")