from functools import reduce
from operator import add

from flask import Blueprint, request, jsonify

from src.services.document_service import get_documents, create_vectorized_documents
from src.services.chat_service import chat_documents
from src.document_store import get_document_store
from src.generator import get_ollama_generator
from src.services.record_service import create_records, get_records, delete_records

# Define the blueprint
api = Blueprint("api", __name__)

# Initialize ollama generator and vector database
ollama_generator = get_ollama_generator()
document_store = get_document_store()

@api.route('/getRecords', methods=['POST'])
def get_vectorized_records():
    to_be_converted_text = request.json.get("to_be_converted_text")
    if not to_be_converted_text:
        return jsonify({"error": "No text information provided"}), 400
    try:
        retrieved_documents = get_records(
            vdb=document_store,
            to_be_converted_text=to_be_converted_text
        )
        return {"vec_docs": retrieved_documents}
    except Exception as e:
        return {"error": f"Something went wrong with retrieving documents: {e}"}, 500

@api.route('/record', methods=['POST'])
def store_records():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    return create_records(document_store, file)

@api.route('/record', methods=['DELETE'])
def remove_records():
    documents = request.json.get("documents")
    if not documents and documents != []:
        return jsonify({"error": "No documents prop provided"}), 400

    try:
        return jsonify(delete_records(document_store))
    except Exception as e:
        return {"error": f"Something went wrong with deleting documents: {e}"}, 500

@api.route('/record/drop', methods=['DELETE'])
def drop_record():
    try:
        is_deleted = document_store.client.delete_collection(collection_name=document_store.index)
        return jsonify({"message": f"Successfully deleted '{document_store.index}' collection", "result": is_deleted})
    except Exception as e:
        return {"error": f"Something went wrong when trying to delete collection with name {document_store.index}: {e}"}

@api.route('/getDocuments', methods=['POST'])
def get_vectorized_documents():
    to_be_converted_text = request.json.get("to_be_converted_text")
    if not to_be_converted_text:
        return jsonify({"error": "No text information provided"}), 400
    try:
        retrieved_documents = get_documents(
            vdb=document_store,
            to_be_converted_text=to_be_converted_text
        )

        return {"vec_docs": retrieved_documents}
    except Exception as e:
        return {"error": f"Something went wrong with retrieving documents: {e}"}, 500

@api.route('/document', methods=['POST'])
def store_pdf():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    try:
        return create_vectorized_documents(document_store, file)
    except Exception as e:
        return {"error": f"Something went wrong with writing a pdf as documents in document store: {e}"}, 500

@api.route('/documents', methods=['POST'])
def store_pdfs():
    files = request.json.get("files")
    if not files:
        return jsonify({"error": "No files information provided"}), 400
    try:
        count_array = [create_vectorized_documents(document_store, file_object['file'])['count'] for file_object in files]
        return {"total_count": reduce(add, count_array)}
    except Exception as e:
        return {"error": f"Something went wrong with writing pdfs as documents in document store: {e}"}, 500

@api.route('/chat', methods=['POST'])
def chat_with_documents():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    prompted_documents = chat_documents(
        vdb=document_store,
        question=question,
        generator=ollama_generator
    )
    return jsonify({"prompted_documents": prompted_documents})
