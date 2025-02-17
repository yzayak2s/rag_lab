from dotenv import dotenv_values, find_dotenv
from haystack_integrations.components.generators.ollama import OllamaGenerator

ollama_chat_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_CHAT_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')

def get_ollama_generator(generation_kwargs_config=None):
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}
    return OllamaGenerator(
        model=ollama_chat_model,
        url=ollama_url,
        generation_kwargs=generation_kwargs_config
    )