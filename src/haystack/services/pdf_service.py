from haystack.components.converters import PyPDFToDocument

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