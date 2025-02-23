from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

def get_document_store():
    """
    Returns the initialized document store.

    :return:
    """
    return QdrantDocumentStore(
        path="qdrant/storage_local",
        index="Document",
        embedding_dim=768, # it differs from model to model
        recreate_index=False,
        hnsw_config={"m": 64, "ef_construct": 512}, # Optional
    )