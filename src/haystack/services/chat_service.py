from haystack.components.builders import PromptBuilder

from services.prompt_service import get_prompt_template
from services.document_service import get_documents

def chat_documents(vdb, question, generator):
    """
    This function returns a response to a given question based on the content in the documents.

    :param vdb: A Qdrant document store
    :param question: A question
    :param generator:

    :return: A response to the given question
    """
    embedded_documents = get_documents(vdb, question)
    prompt_builder = PromptBuilder(template=get_prompt_template())
    prompted_documents = prompt_builder.run(
        documents=embedded_documents['documents'],
        question=question
    )

    return generator.run(prompt=prompted_documents['prompt'])