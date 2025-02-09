from dotenv import dotenv_values, find_dotenv
from haystack_integrations.components.generators.ollama import OllamaGenerator

ollama_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')

def get_ollama_generator():
    return OllamaGenerator(model=ollama_model, url=ollama_url)