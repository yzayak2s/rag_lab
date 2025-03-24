from quart import g
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

def get_document_store(collection_name):
    """
    Returns the initialized document store.

    :return:
    """
    if "vdb" not in g:
        # g.vdb = QdrantDocumentStore(
        #     path="qdrant/storage_local",
        #     index=collection_name,
        #     embedding_dim=768, # it differs from model to model
        #     recreate_index=False,
        #     hnsw_config={"m": 64, "ef_construct": 512}, # Optional
        # )
        g.vdb = ElasticsearchDocumentStore(
            hosts="http://localhost:9200",
            index=collection_name,
        )
    return g.vdb