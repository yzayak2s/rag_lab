# reference: https://docs.haystack.deepset.ai/docs/ollamagenerator#on-its-own

from haystack_integrations.components.generators.ollama import OllamaGenerator
import json

generator = OllamaGenerator(model="llama3.2",
                            url = "http://localhost:11434",
                            generation_kwargs={
                                "num_predict": 100,
                                "temperature": 0.9,
                            })

print(json.dumps(generator.run("Who is the best American actor?"), indent=4))