# RAGLab
Konzeption und prototypische Umsetzung eines RAG-Systems zur Durchführung wissenschaftlicher Literaturanalysen

## Beschreibung
Dies ist ein API-Backend bzw. Microservice, dass auf die Haystack Technologie basiert. Sie repräsentiert ein RAG-System das bei der Durchführung wissenschaftlicher Literaturanalysen unterstützt.

## Inhaltsverzeichnis
- Installation
- Verwendung

## Installation
Schritte zur Konfiguration und Installation des Projekts:
1. Python 3.10.7 manuell oder mit `pyenv`
2. Ins Projektverzeichnis navigieren mit `cd PFAD_ZU_RAG_LAB`
3. Optional: Virtuelle Environment aufsetzen [(Referenz)](https://docs.python.org/3.10/library/venv.html)
4. Paketmanager mit `pip install poetry`
5. Notwendigen Pakete bzw. Dependencies mit `poetry install --no-root`
6. Ollama [(Referenz)](https://ollama.com)
7. LLMs mit `ollama`
   1. `ollama pull nomic-embed-text` für Vektorisierung
   2. `ollama pull llama3.2` für Textgenerierung
8. Tool für Abfragen (z.B. Postman)

## Verwendung
Starten der Anwendung:
```bash
quart --app src run
```

Abfragen erfolgt über die vorbereitete Postman Kollektion.
Optional auch mit cURL, wget usw. (wird jedoch noch nicht empfohlen) 