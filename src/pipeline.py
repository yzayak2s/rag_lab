from dotenv import dotenv_values, find_dotenv
from haystack import Pipeline
from haystack.components.writers import DocumentWriter

from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

ollama_embed_model = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_EMBED_MODEL')
ollama_url = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_URL')

def create_records_pipeline(vdb, generation_kwargs_config=None):
    """
    Creates a haystack pipeline for records and returns it.
    :return:
    """
    document_embedder = OllamaDocumentEmbedder(
        model=ollama_embed_model,
        url=ollama_url,
        generation_kwargs=generation_kwargs_config
    )
    document_writer = DocumentWriter(document_store=vdb, policy=DuplicatePolicy.SKIP)

    pipeline = Pipeline()
    pipeline.add_component(instance=document_embedder, name="document_embedder")
    pipeline.add_component(instance=document_writer, name="document_writer")

    pipeline.connect(sender="document_embedder.documents", receiver="document_writer.documents")
    return pipeline

def create_docs_first_process_pipeline(generation_kwargs_config=None):
    """
    Creates a haystack first process pipeline for documents and returns it.
    :return:
    """
    document_cleaner = DocumentCleaner(remove_repeated_substrings=True)
    document_splitter = DocumentSplitter(split_by="word", split_length=400, respect_sentence_boundary=True)
    document_splitter.warm_up()
    document_embedder = OllamaDocumentEmbedder(
        model=ollama_embed_model,
        url=ollama_url,
        generation_kwargs=generation_kwargs_config
    )

    pipeline = Pipeline()
    pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    pipeline.add_component(instance=document_splitter, name="document_splitter")
    pipeline.add_component(instance=document_embedder, name="document_embedder")

    pipeline.connect(sender="document_cleaner.documents", receiver="document_splitter.documents")
    pipeline.connect(sender="document_splitter.documents", receiver="document_embedder.documents")
    return pipeline