from flask import Blueprint, request, jsonify

# Define the blueprint
api = Blueprint("api", __name__)

@api.route('/storePDF', methods=['POST'])
def store_pdf():
    file = request.json.get("file")
    if not file:
        return jsonify({"error": "No file information provided"}), 400

    return {}