import logging
import os

import pandas as pd
from dotenv import dotenv_values, find_dotenv
from haystack import Document
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.pipeline import create_records_pipeline
from src.models.Record import Record

ollama_embed_model = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_EMBED_MODEL')
ollama_url = dotenv_values(find_dotenv(".quartenv")).get('OLLAMA_URL')

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

async def create_records(vdb: QdrantDocumentStore, file, generation_kwargs_config=None):
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

    try:
        pipeline = create_records_pipeline(vdb)
        result = pipeline.run(
            data={"document_embedder": {"documents": documents}},
        )
        vdb.client.close()
        return result['document_writer']
    except Exception as e:
        logger.error(f"Failed to write records as documents to Qdrant document store: {e}")

async def get_records(vdb: QdrantDocumentStore, to_be_converted_text, generation_kwargs_config=None):
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
        vdb.client.close()
    except Exception as e:
        logger.error(f"Failed to retrieve records from document store: {e}")
        raise e

    return retrieved_documents

async def get_all_records(vdb: QdrantDocumentStore):
    """
    This function returns a list of stored records as vectorized documents from the document store.

    :return: A list of stored records as vectorized documents
    """
    try:
        result = vdb.filter_documents()
        vdb.client.close()
        return result
    except Exception as e:
        logger.error(f"Failed to retrieve records from document store: {e}")
        raise e

async def delete_records(vdb: QdrantDocumentStore):
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
        vdb.client.close()
    except Exception as e:
        logger.error(f"Failed to delete documents from document store: {e}")
        raise e