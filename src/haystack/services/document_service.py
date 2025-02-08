from dotenv import dotenv_values, find_dotenv
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

ollama_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')


def get_documents(vdb, to_be_converted_text):
    """
    This function returns a list of stored vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored vectorized documents
    """
    text_embedder = OllamaTextEmbedder(model=ollama_model, url=ollama_url)
    embedded_text = text_embedder.run(text=to_be_converted_text)
    embedding_retriever = QdrantEmbeddingRetriever(document_store=vdb)
    retrieved_documents = embedding_retriever.run(query_embedding=embedded_text['embedding'])

    return retrieved_documents