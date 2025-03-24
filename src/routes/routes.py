from quart import Blueprint, request, jsonify

from src.services.document_service import get_documents, create_vectorized_documents, delete_documents, get_all_documents
from src.services.chat_service import chat_documents
from src.document_store import get_document_store
from src.generator import get_ollama_generator
from src.services.record_service import create_records, get_records, delete_records, get_all_records

# Define the blueprint
api = Blueprint("api", __name__)

# Initialize ollama generator and vector database
ollama_generator = get_ollama_generator()

@api.route('/getRecords', methods=['POST'])
async def get_vectorized_records():
    document_store = get_document_store(collection_name='record')
    data = await request.get_json()
    if data:
        try:
            retrieved_documents = await get_records(
                vdb=document_store,
                to_be_converted_text=data['to_be_converted_text']
            )
            return {"vec_docs": retrieved_documents}
        except Exception as e:
            return {"error": f"Something went wrong with retrieving documents: {e}"}, 500
    return jsonify({"error": "No text information provided"}), 400

@api.route('/record', methods=['GET'])
async def retrieve_records():
    document_store = get_document_store(collection_name='record')
    try:
        return jsonify(await get_all_records(document_store))
    except Exception  as e:
        return {"error": f"Something went wrong with retrieving all records: {e}"}, 400

@api.route('/record', methods=['POST'])
async def store_records():
    document_store = get_document_store(collection_name='record')
    data = await request.get_json()
    if data:
        return await create_records(document_store, data["file"])
    return jsonify({"error": "No file information provided"}), 400

@api.route('/record', methods=['DELETE'])
async def remove_records():
    document_store = get_document_store(collection_name='record')
    data = await request.get_json()
    if not data and not hasattr(data, "documents") and data["documents"] != []:
        return jsonify({"error": "No documents prop provided"}), 400

    try:
        return jsonify(await delete_records(document_store))
    except Exception as e:
        return {"error": f"Something went wrong with deleting documents: {e}"}, 500

@api.route('/record/drop', methods=['DELETE'])
async def drop_record():
    document_store = get_document_store(collection_name='record')
    try:
        is_deleted = document_store.client.delete_collection(collection_name=document_store.index)
        document_store.client.close()
        return jsonify({"message": f"Successfully deleted '{document_store.index}' collection", "result": is_deleted})
    except Exception as e:
        return {"error": f"Something went wrong when trying to delete collection with name {document_store.index}: {e}"}

@api.route('/document', methods=['GET'])
async def retrieve_all_documents():
    document_store = get_document_store(collection_name='document')
    try:
        return await get_all_documents(document_store)
    except Exception as e:
        return {"error": f"Something went wrong with retrieving documents: {e}"}, 500

@api.route('/getDocuments', methods=['POST'])
async def get_vectorized_documents():
    document_store = get_document_store(collection_name='document')
    data = await request.get_json()
    if data:
        try:
            retrieved_documents = await get_documents(
                vdb=document_store,
                to_be_converted_text=data["to_be_converted_text"]
            )

            return {"vec_docs": retrieved_documents}
        except Exception as e:
            return {"error": f"Something went wrong with retrieving documents: {e}"}, 500
    return jsonify({"error": "No text information provided"}), 400

@api.route('/document', methods=['POST'])
async def store_pdf():
    document_store = get_document_store(collection_name='document')
    data = await request.get_json()
    query_params = request.args.to_dict()
    if data:
        try:
            return await create_vectorized_documents(document_store, data["files"], query_params)
        except Exception as e:
            return {"error": f"Something went wrong with writing a pdf as documents in document store: {e}"}, 500
    return jsonify({"error": "No file information provided"}), 400

@api.route('/documents', methods=['DELETE'])
async def remove_documents():
    document_store = get_document_store(collection_name='document')
    data = await request.get_json()
    if not data and data != []:
        return jsonify({"error": "No documents prop provided"}), 400

    try:
        return jsonify(await delete_documents(document_store))
    except Exception as e:
        return {"error": f"Something went wrong with deleting documents: {e}"}, 500

@api.route('/documents/drop', methods=['DELETE'])
async def drop_documents():
    document_store = get_document_store(collection_name='document')
    try:
        is_deleted = document_store.client.delete_collection(collection_name=document_store.index)
        return jsonify({"message": f"Successfully deleted '{document_store.index}' collection", "result": is_deleted})
    except Exception as e:
        return {"error": f"Something went wrong when trying to delete collection with name {document_store.index}: {e}"}

@api.route('/chat', methods=['POST'])
async def chat_with_documents():
    query_params = request.args.to_dict()
    document_store = get_document_store(collection_name=query_params['collection'])
    data = await request.get_json()
    if data:
        prompted_documents = await chat_documents(
            vdb=document_store,
            question=data['question'],
            generator=ollama_generator
        )
        return jsonify({"prompted_documents": prompted_documents})
    return jsonify({"error": "No question provided"}), 400

