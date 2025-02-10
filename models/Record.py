from dataclasses import dataclass

@dataclass
class Record:
    """
    This model specifies the format of a single record and are stored in qdrant document store.
    """

    def __init__(
            self,
            authors,
            article_title,
            source_title,
            abstract,
            publication_year
    ):
        self.authors = authors
        self.article_title = article_title
        self.source_title = source_title
        self.abstract = abstract
        self.publication_year =publication_year