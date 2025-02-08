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
# TODO: Passe den rest des templates an und gestalte die Funktion dynamischer.
# TODO: Sollte nicht im service directory sein.