from flask import Blueprint, request, jsonify

from services.document_service import get_documents, create_vectorized_documents
from qdrant_store import get_qdrant_document_store

# Define the blueprint
api = Blueprint("api", __name__)
qdrant_document_store = get_qdrant_document_store()

@api.route('/storePDF', methods=['POST'])
def store_pdf():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    return create_vectorized_documents(qdrant_document_store, file)

@api.route('/getVecDocs', methods=['POST'])
def get_vectorized_documents():
    to_be_converted_text = request.json.get("to_be_converted_text")
    if not to_be_converted_text:
        return jsonify({"error": "No text information provided"}), 400
    retrieved_documents = get_documents(
        vdb=qdrant_document_store,
        to_be_converted_text=to_be_converted_text
    )

    return {"vec_docs": retrieved_documents}