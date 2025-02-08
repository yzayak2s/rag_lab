# Importing required libraries
from flask import Flask, request, jsonify
from routes import api
from dotenv import load_dotenv

# Load environment variables from .env and .flaskenv files
load_dotenv()

app = Flask(__name__)
app.register_blueprint(api, url_prefix='/api')

@app.route('/')
def welcome():
    return "<h1>Welcome to your rag application!</h1>"

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/generate', methods=['POST'])
def generate_text():
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    result = llm.run(input_text)
    return jsonify({"generated_text": result})