# references:
#   https://docs.haystack.deepset.ai/docs/ollamagenerator
#   https://mer.vin/2024/01/haystack-ai-to-create-rag-pipeline/

# Importing required libraries
from datasets import load_dataset
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator  # Updated import statement
from haystack import Pipeline  # Updated import statement

# Load dataset and create documents
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

# Initialize document store and write documents
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

# Initialize retriever
retriever = InMemoryBM25Retriever(document_store)

# Define prompt template
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

# Initialize prompt builder
prompt_builder = PromptBuilder(template=template)

# Initialize Ollama generator
generator = OllamaGenerator(
    model="llama3.2",
    url="http://127.0.0.1:11434",
    generation_kwargs={
        "num_predict": 100,
        "temperature": 0.9,
    }
)

# Create and configure pipeline
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

# Run the pipeline with a sample question
question = "What does Rhodes Statue look like?"
response = basic_rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question}
    }
)
print(response["llm"]["replies"][0])