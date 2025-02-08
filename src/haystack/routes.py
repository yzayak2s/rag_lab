from flask import Blueprint, request, jsonify

from services.pdf_service import convert_pdf_to_document

# Define the blueprint
api = Blueprint("api", __name__)

@api.route('/storePDF', methods=['POST'])
def store_pdf():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    convert_pdf_to_document(file["file_path"], file["authors"])

    return {}