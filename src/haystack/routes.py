from dotenv import dotenv_values, find_dotenv
from flask import Blueprint, request, jsonify
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

from services.pdf_service import convert_pdf_to_document, create_vectorized_documents
from qdrant_store import get_qdrant_document_store

# Define the blueprint
api = Blueprint("api", __name__)
qdrant_document_store = get_qdrant_document_store()

ollama_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')

@api.route('/storePDF', methods=['POST'])
def store_pdf():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    documents = convert_pdf_to_document(file["file_path"], file["authors"])
    document_cleaner = DocumentCleaner(remove_repeated_substrings=True)
    cleaned_documents = document_cleaner.run(documents=documents)
    document_splitter = DocumentSplitter(split_by="word", split_length=400)
    split_documents = document_splitter.run(documents=cleaned_documents['documents'])
    document_embedder = OllamaDocumentEmbedder(model=ollama_model, url=ollama_url)
    vectorized_documents = document_embedder.run(documents=split_documents['documents'])

    return create_vectorized_documents(qdrant_document_store, vectorized_documents['documents'])