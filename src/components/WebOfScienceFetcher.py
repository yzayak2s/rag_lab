from typing import List

from haystack import component
from haystack import Document

@component
class WebOfScienceFetcher:

    @component.output_types(documents=List[Document])
    def run(self, queries: list[str]):
        # TODO: Here the implementation for retrieving data from web of science.
        return