import logging
import os

import pandas as pd
from dotenv import dotenv_values, find_dotenv
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.models.Record import Record

ollama_embed_model = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_EMBED_MODEL')
ollama_url = dotenv_values(find_dotenv(".flaskenv")).get('OLLAMA_URL')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def create_records(vdb: QdrantDocumentStore, file, generation_kwargs_config=None):
    """
    This function stores records as vectorized documents in Qdrant document store.
    :return:
    """

    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    # Get columns specified by column names from Excel file
    dd = pd.read_excel(
        file['file_path'],
        usecols=["Authors", "Article Title", "Source Title", "Abstract", "Publication Year"],
    )
    csv_dd_file = 'qdrant/storage_local/xlsxs/literature_web_of_science.csv'

    # Convert input of Excel file to csv and save it local
    dd.to_csv(
        path_or_buf=csv_dd_file,
        index=False, sep="|", header=False
    )

    # Get saved csv file
    csv_df = pd.read_csv(csv_dd_file, header=None, sep="|")

    # Delete created csv file
    os.remove(csv_dd_file)

    # Iterate through data frame and for each row instantiate a record of type Record.
    records = [Record(*row) for row in csv_df.itertuples(index=False)]

    document_splitter = DocumentSplitter(split_by="sentence", split_length=1)
    document_splitter.warm_up()

    documents = []

    # Iterate through records to instantiate document objects of type Document
    for row in records:
        documents.append(
            Document(
                content=row.abstract,
                meta={
                    "authors": row.authors,
                    "article_title": row.article_title,
                    "source_title": row.source_title,
                    "publication_year": row.publication_year
                }
            )
        )

    split_documents = document_splitter.run(documents=documents)

    document_embedder = OllamaDocumentEmbedder(
        model=ollama_embed_model,
        url=ollama_url,
        generation_kwargs=generation_kwargs_config
    )
    vectorized_documents = document_embedder.run(documents=split_documents['documents'])

    try:
        return {"count": vdb.write_documents(documents=vectorized_documents['documents'], policy=DuplicatePolicy.SKIP)}
    except Exception as e:
        logger.error(f"Failed to write records as documents to Qdrant document store: {e}")

def get_records(vdb: QdrantDocumentStore, to_be_converted_text, generation_kwargs_config=None):
    """
    This function returns a list of stored records as vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored records as vectorized documents
    """

    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    try:
        text_embedder = OllamaTextEmbedder(model=ollama_embed_model, url=ollama_url, generation_kwargs=generation_kwargs_config)
        embedded_text = text_embedder.run(text=to_be_converted_text)
        embedding_retriever = QdrantEmbeddingRetriever (document_store=vdb)
        retrieved_documents = embedding_retriever.run(query_embedding=embedded_text['embedding'])
    except Exception as e:
        logger.error(f"Failed to retrieve records from document store: {e}")
        raise e

    return retrieved_documents

def delete_records(vdb: QdrantDocumentStore):
    """
    This function deletes all stored documents from the document store.
    :param vdb:
    :return:
    """
    try:
        vdb.recreate_collection(
            collection_name=vdb.index,
            distance=vdb.get_distance(similarity=vdb.similarity),
            embedding_dim=vdb.embedding_dim
        )
    except Exception as e:
        logger.error(f"Failed to delete documents from document store: {e}")
        raise e