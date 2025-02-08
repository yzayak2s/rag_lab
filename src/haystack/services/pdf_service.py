from haystack.components.converters import PyPDFToDocument

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

def create_vectorized_documents(vdb, vectorized_documents):
    """
    This function stores vectorized documents in Qdrant document store.
    :return:
    """

    try:
        return {"count": vdb.write_documents(documents=vectorized_documents, policy=DuplicatePolicy.SKIP)}
    except Exception as e:
        logger.error(f"Failed to write documents to Qdrant document store: {e}")