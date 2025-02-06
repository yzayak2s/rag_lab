# references:
#   https://docs.haystack.deepset.ai/docs/ollamagenerator
#   https://mer.vin/2024/01/summarise-legal-documents-mixtral-jina-ai/

# Importing required libraries
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder
from haystack_integrations.components.embedders.jina import JinaTextEmbedder

from haystack.utils import Secret
from dotenv import load_dotenv

import os

# Load environment variables from .env file
load_dotenv()

# Get the API keys from the environment variables
jina_api_key = os.environ.get("JINA_API_KEY")
# Convert string to 'Secret' type
jina_api_key_secret = Secret.from_token(jina_api_key)
hf_token = os.environ.get("HUGGINGFACE_API_KEY")
hf_token_secret = Secret.from_token(hf_token)

# Retrieving the PDF document
document_store = ChromaDocumentStore()
fetcher = LinkContentFetcher()
converter = PyPDFToDocument()

cleaner = DocumentCleaner(remove_repeated_substrings=True)
splitter = DocumentSplitter(split_by="word", split_length=500)
writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
retriever = ChromaEmbeddingRetriever(document_store)
document_embedder = JinaDocumentEmbedder(
    task="retrieval.query", api_key=jina_api_key_secret, model="jina-embeddings-v3"
)

# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=fetcher, name="fetcher")
indexing_pipeline.add_component(instance=converter, name="converter")
indexing_pipeline.add_component(instance=cleaner, name="cleaner")
indexing_pipeline.add_component(instance=splitter, name="splitter")
indexing_pipeline.add_component(instance=document_embedder, name="embedder")
indexing_pipeline.add_component(instance=writer, name="writer")

indexing_pipeline.connect("fetcher.streams", "converter.sources")
indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

urls = [
    "https://cases.justia.com/federal/district-courts/california/candce/3:2020cv06754/366520/813/0.pdf"
]
indexing_pipeline.draw("indexing_pipeline.png")

indexing_pipeline.run(data={"fetcher": {"urls": urls}})

# Define prompt template
prompt = """ 
Answer the question, based on the content in the documents. If you can't answer based on the documents, say so.

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Question: {{question}}
"""

text_embedder = JinaTextEmbedder(
    api_key=jina_api_key_secret, model="jina-embeddings-v3"
)

# Initialize prompt builder
prompt_builder = PromptBuilder(template=prompt)

# Initialize Ollama generator
generator = OllamaGenerator(
    model="llama3.2",
    url="http://127.0.0.1:11434",
    generation_kwargs={
        "num_predict": 100,
        "temperature": 0.9,
    }
)

# Create and configure RAG pipeline
rag = Pipeline()
rag.add_component("text_embedder", text_embedder)
rag.add_component(instance=prompt_builder, name="prompt_builder")
rag.add_component("retriever", retriever)
rag.add_component("generator", generator)

rag.connect("text_embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt_builder.documents")
rag.connect("prompt_builder.prompt", "generator.prompt")
rag.draw("rag_pipeline.png") # zeichnet die Pipeline auf

# Run the pipeline with a sample question
question = "Summarize what happened in Google v. Sonos"
# question ="What should Sonos have done differently?"

result = rag.run(
    data={
        "text_embedder": {"text": question},
        "retriever": {"top_k": 3},
        "prompt_builder": {"question": question},
        "generator": {"generation_kwargs": {"max_new_tokens": 350}},
    }
)
print(result['generator']['replies'][0])
