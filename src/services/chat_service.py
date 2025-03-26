from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders import PromptBuilder

from src.services.prompt_service import get_prompt_template, get_prompt_template_answer_with_reference
from src.services.document_service import get_documents

async def chat_documents(vdb, question, generator):
    """
    This function returns a response to a given question based on the content in the documents.

    :param vdb: A Qdrant document store
    :param question: A question
    :param generator:

    :return: A response to the given question
    """
    embedded_documents = await get_documents(vdb, question)
    prompt_builder = PromptBuilder(template=get_prompt_template())
    prompted_documents = prompt_builder.run(
        documents=embedded_documents['embedding_retriever']['documents'],
        question=question
    )

    return generator.run(prompt=prompted_documents['prompt'])

async def chat_documents_with_reference(vdb, question, generator):
    """
    This function returns a response to a given question based on the content in the documents.

    :param vdb: A Qdrant document store
    :param question: A question
    :param generator:

    :return: A response to the given question
    """
    embedded_documents = await get_documents(vdb, question)
    prompt_builder = PromptBuilder(template=get_prompt_template_answer_with_reference())
    prompted_documents = prompt_builder.run(
        documents=embedded_documents['embedding_retriever']['documents'],
        question=question
    )
    result_dict = generator.run(prompt=prompted_documents['prompt'])
    answer_builder = AnswerBuilder()
    answers = answer_builder.run(
        replies=result_dict['replies'],
        query=question,
        meta=result_dict['meta'],
        documents=embedded_documents['embedding_retriever']['documents']
    )

    return answers