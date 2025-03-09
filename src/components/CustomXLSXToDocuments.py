from typing import List
import os

import pandas as pd

from src.models.Record import Record

from haystack import Document, component

@component
class CustomXLSXToDocuments:
    """
    Converts a xlsx file to documents.

    ### Usage example:

    ```python
    from src.components import CustomXLSXToDocument

    document_converter = CustomXLSXToDocument()
    documents = document_converter.run(file="FILE_PATH_TO_XLSX")

    assert documents[0].content == "Some content ..."
    ```
    """
    @component.output_types(documents=List[Document])
    def run(self, file):
        """
        Converts a xlsx file to documents.

        :param file: file path to xlsx
        :return: A list of converted documents.
        """
        # Get columns specified by column names from Excel file
        dd = pd.read_excel(
            file['file_path'],
            usecols=["Authors", "Article Title", "Source Title", "Abstract", "Publication Year"],
        )
        csv_dd_file = 'qdrant/storage_local/xlsxs/literature_web_of_science.csv'

        # Convert input of Excel file to csv and save it local
        dd.to_csv(
            path_or_buf=csv_dd_file,
            index=False, sep="|", header=False
        )

        # Get saved csv file
        csv_df = pd.read_csv(csv_dd_file, header=None, sep="|")

        # Delete created csv file
        os.remove(csv_dd_file)

        # Iterate through data frame and for each row instantiate a record of type Record.
        records = [Record(*row) for row in csv_df.itertuples(index=False)]

        documents = []

        # Iterate through records to instantiate document objects of type Document
        for row in records:
            documents.append(
                Document(
                    content=row.abstract,
                    meta={
                        "authors": row.authors,
                        "article_title": row.article_title,
                        "source_title": row.source_title,
                        "publication_year": row.publication_year
                    }
                )
            )
        return documents