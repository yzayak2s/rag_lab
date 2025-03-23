import logging

from haystack.document_stores.types import DuplicatePolicy

from src.pipeline import create_docs_first_process_pipeline, create_docs_second_process_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

async def get_documents(vdb, to_be_converted_text, generation_kwargs_config=None):
    """
    This function returns a list of stored vectorized documents from the Qdrant document store
    based on embedded text (converted text).
    :return: A list of stored vectorized documents
    """
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    try:
        pipeline = create_docs_second_process_pipeline(vdb, generation_kwargs_config)
        retrieved_documents = pipeline.run(
            data={"text_embedder": {"text": to_be_converted_text}},
        )
        vdb.client.close()
        return retrieved_documents
    except Exception as e:
        raise e

async def get_all_documents(vdb):
    """
    This function returns a list of stored vectorized documents from the document store.

    :return: A list of stored vectorized documents
    """
    try:
        result = vdb.filter_documents()
        vdb.client.close()
        return result
    except Exception as e:
        logger.error(f"Failed to retrieve documents from document store: {e}")
        raise e

async def create_vectorized_documents(vdb, files, split_args, generation_kwargs_config=None):
    """
    This function stores vectorized documents in Qdrant document store.
    :return:
    """
    if generation_kwargs_config is None:
        generation_kwargs_config = {"temperature": 0.0}

    documents = []
    for file_object in files:
        pipeline = create_docs_first_process_pipeline(split_args, generation_kwargs_config)
        vectorized_documents = pipeline.run(
            data={
                "document_converter": {"sources": [file_object["file_path"]], "meta": [{"authors": file_object["authors"]}]},
            },
        )
        documents += vectorized_documents['document_embedder']['documents']
    try:
        count = vdb.write_documents(documents=documents, policy=DuplicatePolicy.SKIP)
        vdb.client.close()
        return {"count": count}
    except Exception as e:
        logger.error(f"Failed to write documents to Qdrant document store: {e}")
        raise e

async def delete_documents(vdb):
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