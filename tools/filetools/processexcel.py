import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import Tool
from tools.filetools.exceldf import excel_to_df
from langchain_openai import ChatOpenAI
import json

load_dotenv()

# Instantiate your LLM
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
def parse_kv_string(s: str) -> dict:
        """
        Tries to parse a string as JSON, then as key:value pairs.
        """
        try:
            # Try JSON first
            return json.loads(s)
        except Exception:
            pass
        # Fallback: try key:value pairs
        result = {}
        for part in s.split(","):
            if ":" in part:
                key, value = part.split(":", 1)
                result[key.strip()] = value.strip()
        return result

class ProcessExcelQuestionTool:
    def __call__(self, params) -> str:
        if isinstance(params, str):
            params = parse_kv_string(params)
        task_id = params.get("task_id")
        question = params.get("question")
        df = excel_to_df(task_id)
        df_agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
        return df_agent.run(question)

process_excel_question_tool = Tool(
    name="process_excel_question_tool",
    func=ProcessExcelQuestionTool(),
    description="Given 1 dictionary with 2 keys task_id and question, loads the Excel as a DataFrame and answers the question using pandas."
)