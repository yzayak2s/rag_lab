# Importing required libraries
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from haystack_integrations.components.generators.ollama import OllamaGenerator

# Load environment variables from .env and .flaskenv files
load_dotenv()

app = Flask(__name__)

# Initialize LLM – TODO: Will be outsourced in a separate python-file
llm = OllamaGenerator(
    model="llama3.2",
    url="http://127.0.0.1:11434",
    generation_kwargs={
        # "num_predict": 100, # maximum number of tokens to predict when generation text
        "temperature": 0.0, # creativity – TODO: Will be dynamically adjustable
    }
)

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