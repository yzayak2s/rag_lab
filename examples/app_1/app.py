# reference: https://docs.haystack.deepset.ai/docs/ollamagenerator#on-its-own

from haystack_integrations.components.generators.ollama import OllamaGenerator

generator = OllamaGenerator(model="mistral",
                            url = "http://localhost:11434",
                            generation_kwargs={
                                "num_predict": 100,
                                "temperature": 0.9,
                            })

print(generator.run("Who is the best American actor?"))