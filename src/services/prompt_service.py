def get_prompt_template():
    """
        Gibt ein Standard-Prompt-Template zurück.
    """
    return """
Beantworte die Frage wahrheitsgemäß auf der Grundlage der angegebenen PDF-Inhalte.
Die PDF-Inhalte mit der selben source_id gehören zusammen. 
Wenn die bereitgestellten PDF-Inhalte keine Antwort enthalten, dann sage es bitte auch.

PDF-Inhalte:
{% for document in documents %}
    Source_ID: {{document.meta.source_id}}
    Autoren: {{document.meta['authors']}}
    Inhalt: {{document.content}}
{% endfor %}

Frage: {{ question }}
Antwort:
"""

def get_prompt_template_answer_with_reference():
    """
    Gives a prompt template that returns the answer with references.
    :return: str
    """
    return """
Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents.  
You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text.  
Cite the documents using the following format at the end of your answer for example (Authors, Paths).  

You will find the Author and Path at the end of each document.
If multiple documents contain the answer, compare the authors and paths.  
If the documents do not contain the answer to the question, state: ‘Answering is not possible given the available information.’

Given the following information, answer the question.

{% for document in documents %}
    Document[{{loop.index}}]: {{ document.content }} ({{document.meta['authors']}}, {{document.meta['file_path']}})  
{% endfor %}

Question: {{question}}
Answer:
"""