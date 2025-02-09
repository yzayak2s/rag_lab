from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

# Initialize QdrantDocumentStore with the specified path
qdrant_document_store = QdrantDocumentStore(
    path="qdrant/storage_local",
    index="Document",
    embedding_dim=3072, # it differs from model to model
    # recreate_index=True,
    # hnsw_config={"m": 16, "ef_construct": 64} # Optional
)

def get_qdrant_document_store():
    return qdrant_document_store