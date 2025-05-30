from dotenv import dotenv_values, find_dotenv
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter

from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever

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

def create_docs_first_process_pipeline(split_args, generation_kwargs_config=None):
    """
    Creates a haystack first process pipeline for documents and returns it.

    :param split_args:
    :param generation_kwargs_config:
    :return:
    """
    document_converter = PyPDFToDocument(store_full_path=True)
    document_cleaner = DocumentCleaner(remove_repeated_substrings=True)
    document_splitter = DocumentSplitter(
        split_by=split_args['split_by'],
        split_length=int(split_args['split_length']),
        split_overlap=int(split_args['split_overlap']),
        split_threshold=int(split_args['split_threshold']),
        respect_sentence_boundary=True
    )
    document_splitter.warm_up()
    document_embedder = OllamaDocumentEmbedder(
        model=ollama_embed_model,
        url=ollama_url,
        generation_kwargs=generation_kwargs_config
    )

    pipeline = Pipeline()
    pipeline.add_component(instance=document_converter, name="document_converter")
    pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    pipeline.add_component(instance=document_splitter, name="document_splitter")
    pipeline.add_component(instance=document_embedder, name="document_embedder")

    pipeline.connect(sender="document_converter.documents", receiver="document_cleaner.documents")
    pipeline.connect(sender="document_cleaner.documents", receiver="document_splitter.documents")
    pipeline.connect(sender="document_splitter.documents", receiver="document_embedder.documents")
    return pipeline

# TODO: Maybe make it as default and for general use (outsource?!)
def create_docs_second_process_pipeline(vdb, generation_kwargs_config=None):
    """
    Creates a haystack second process pipeline for documents and returns it.
    :param vdb:
    :param generation_kwargs_config:
    :return:
    """
    text_embedder = OllamaTextEmbedder(
        model=ollama_embed_model, url=ollama_url,
        generation_kwargs=generation_kwargs_config
    )
    # embedding_retriever = QdrantEmbeddingRetriever(document_store=vdb)
    embedding_retriever = ElasticsearchEmbeddingRetriever(document_store=vdb)

    pipeline = Pipeline()
    pipeline.add_component(instance=text_embedder, name="text_embedder")
    pipeline.add_component(instance=embedding_retriever, name="embedding_retriever")

    pipeline.connect(sender="text_embedder.embedding", receiver="embedding_retriever")
    return pipeline