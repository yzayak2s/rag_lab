from functools import reduce
from operator import add

from flask import Blueprint, request, jsonify

from services.document_service import get_documents, create_vectorized_documents
from services.chat_service import chat_documents
from qdrant_store import get_qdrant_document_store
from generator import get_ollama_generator
from services.record_service import create_records, get_records

# Define the blueprint
api = Blueprint("api", __name__)

# Initialize ollama generator and vector database
ollama_generator = get_ollama_generator()
qdrant_document_store = get_qdrant_document_store()

@api.route('/getRecords', methods=['POST'])
def get_vectorized_records():
    to_be_converted_text = request.json.get("to_be_converted_text")
    if not to_be_converted_text:
        return jsonify({"error": "No text information provided"}), 400
    retrieved_documents = get_records(
        vdb=qdrant_document_store,
        to_be_converted_text=to_be_converted_text
    )

    return {"vec_docs": retrieved_documents}

@api.route('/record', methods=['POST'])
def store_records():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    return create_records(qdrant_document_store, file)

@api.route('/getPDFs', methods=['POST'])
def get_vectorized_documents():
    to_be_converted_text = request.json.get("to_be_converted_text")
    if not to_be_converted_text:
        return jsonify({"error": "No text information provided"}), 400
    retrieved_documents = get_documents(
        vdb=qdrant_document_store,
        to_be_converted_text=to_be_converted_text
    )

    return {"vec_docs": retrieved_documents}

@api.route('/pdf', methods=['POST'])
def store_pdf():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    return create_vectorized_documents(qdrant_document_store, file)

@api.route('/pdfs', methods=['POST'])
def store_pdfs():
    files = request.json.get("files")
    if not files:
        return jsonify({"error": "No files information provided"}), 400
    count_array = [create_vectorized_documents(qdrant_document_store, file_object['file'])['count'] for file_object in files]
    return {"total_count": reduce(add, count_array)}

@api.route('/chat', methods=['POST'])
def chat_with_documents():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    prompted_documents = chat_documents(
        vdb=qdrant_document_store,
        question=question,
        generator=ollama_generator
    )
    return jsonify({"prompted_documents": prompted_documents})
