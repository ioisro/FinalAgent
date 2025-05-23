import pandas as pd
from langchain.tools import Tool
from tools.filetools.filefetcher import FileFetcher

class ExcelToDataFrame:
    def __call__(self, task_id: str):
        """
        Downloads the Excel file using get_file and returns a pandas DataFrame.
        Optionally specify a sheet name.
        """
        file_obj = FileFetcher.get_file(task_id)
        df = pd.read_excel(file_obj)
        return df

excel_to_df = ExcelToDataFrame()