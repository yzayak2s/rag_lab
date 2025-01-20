# references:
#   https://docs.haystack.deepset.ai/docs/ollamagenerator
#   https://mer.vin/2024/01/healthcare-chatbot-with-mixtral-8x7b/
#   original source: https://haystack.deepset.ai/blog/mixtral-8x7b-healthcare-chatbot

# Importing required libraries
from pymed import PubMed
from typing import List
from haystack import component
from haystack import Document
from haystack import Pipeline

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

from haystack.utils import Secret
from dotenv import load_dotenv

import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables and convert string to 'Secret' type
hf_token = os.environ.get("HUGGINGFACE_API_KEY")
hf_token_secret = Secret.from_token(hf_token)

pubmed = PubMed(tool="Haystack2.6.1Prototype", email="younes.zayakh@smail.inf.h-brs.de")

def documentize(article):
    return Document(
        content=article.abstract,
        meta={
            "title": article.title,
            "keywords": article.keywords
        })

@component
class PubMedFetcher:

    @component.output_types(articles=List[Document])
    def run(self, queries: list[str]):
        cleaned_queries = queries[0].strip().split('\n')

        articles = []
        try:
            for query in cleaned_queries:
                response = pubmed.query(query, max_results=1)
                docments = [documentize(article) for article in response]
                articles.extend(docments)
        except Exception as e:
            print(e)
            print(f"Couldn't fetch articles for queries: {queries}")
        results = {"articles": articles}
        return results

# Initialize first LLM for keywords extraction
keyword_llm = OllamaGenerator(
    model="mistral",
    url="http://127.0.0.1:11434",
    generation_kwargs={
        "num_predict": 100,
        "temperature": 0.9,
    }
)

# Initialize second LLM for article summarization
llm = OllamaGenerator(
    model="mistral",
    url="http://127.0.0.1:11434",
    generation_kwargs={
        "num_predict": 100,
        "temperature": 0.9,
    }
)

# Define keyword prompt template
keyword_prompt_template = """
Your task is to convert the following question into 3 keywords that can be used to find relevant medical research papers on PubMed.
Here is an examples:
question: "What are the latest treatments for major depressive disorder?"
keywords:
Antidepressive Agents
Depressive Disorder, Major
Treatment-Resistant depression
---
question: {{ question }}
keywords:
"""

# Define prompt template
prompt_template = """
Answer the question truthfully based on the given documents.
If the documents don't contain an answer, use your existing knowledge base.

q: {{ question }}
Articles:
{% for article in articles %}
    {{article.content}}
    keywords: {{article.meta['keywords']}}
    title: {{article.meta['title']}}
{% endfor %}

"""

# Initialize prompt builder
keyword_prompt_builder = PromptBuilder(template=keyword_prompt_template)
prompt_builder = PromptBuilder(template=prompt_template)
fetcher = PubMedFetcher()

# Create and configure RAG pipeline
pipe = Pipeline()

pipe.add_component("keyword_prompt_builder", keyword_prompt_builder)
pipe.add_component("keyword_llm", keyword_llm)
pipe.add_component("pubmed_fetcher", fetcher)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("keyword_prompt_builder.prompt", "keyword_llm.prompt")
pipe.connect("keyword_llm.replies", "pubmed_fetcher.queries")

pipe.connect("pubmed_fetcher.articles", "prompt_builder.articles")
pipe.connect("prompt_builder.prompt", "llm.prompt")

# Run the pipeline with a sample question
question = "What are the most current treatments for long COVID?"
result = pipe.run(
    data={
        "keyword_prompt_builder": {"question": question},
        "prompt_builder": {"question": question},
        "llm": {"generation_kwargs": {"max_new_tokens": 500}}
    }
)
print(question)
print(result['llm']['replies'][0])