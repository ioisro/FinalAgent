import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional, List
from langgraph.prebuilt import create_react_agent
from tools.searchtools.wikisearch import wikipedia_search_langchain_tool
from tools.searchtools.youtubetranscript import youtube_transcript_tool
from tools.filetools.audiotranscript import AudioTranscriber
from tools.filetools.processexcel import process_excel_question_tool
from langchain_community.tools.tavily_search import TavilySearchResults

import base64
from tools.filetools.filefetcher import FileFetcher

load_dotenv()
search_tool = TavilySearchResults()
transcriber = AudioTranscriber()

# Instantiate your LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    search_tool,
    wikipedia_search_langchain_tool,
    youtube_transcript_tool,
    process_excel_question_tool
]

agent = create_react_agent(
    tools=tools,
    model=llm,
    prompt="You are a helpful general assistant. I will ask question. You can either directly answer or take help of tools to get additional information before answering. Use the wikipedia_search_tool only if wikipedia is mentioned in the query.",
)

format = "You are a answer refiner. I will give you the user question and the answer. You need to refine the answer based on strict rules." \
    "Your answer should be a number or as few words as required. " \
    "When giving a number, do not give commas or symbols." \
    "When giving a list, use separators like comma by default or if something specific mentioned in question, use that." \
    "Always follow these rules."

def run_agent(task_id: str, question: str, file_name: str = "") -> str:
    messages = []
    file_obj = None
    if task_id and (file_name.endswith((".png", ".mp3", ".py"))):
        file_obj = FileFetcher.get_file(task_id)
    
    if file_name.endswith(".png") and file_obj:
        # Read and encode image as base64 data URL
        image_bytes = file_obj.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{image_b64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
    elif file_name.endswith(".mp3") and file_obj:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        transcript = transcriber(tmp_path)
        messages = [HumanMessage(content=f"{question}\n\nHere is the transcript of the audio file:\n{transcript}")]
    elif file_name.endswith(".py") and file_obj:
        # Python file: send code as context
        code = file_obj.read().decode("utf-8")
        messages = [HumanMessage(content=f"{question}\n\nHere is the Python code:\n```python\n{code}\n```")]
    elif file_name.endswith(".xlsx"):
        # Instead of sending the file content, instruct the agent to use the converter tool with the task_id
        messages = [HumanMessage(
            content=(
                f"{question}\n\n"
                f"To access the Excel data, use the following task_id: {task_id}."
            )
        )]
    else:
        # Default: just the question
        messages = [HumanMessage(content=question)]
    response = agent.invoke({"messages": messages})

    print(response)
    ai_answer = response["messages"][-1].content

    question_prompt = f"""Given the original user question, the AI answer and the desired format, return the answer in DESIRED FORMAT.
    User question: {question}
    AI answer: {ai_answer}
    Desired format: {format}
    """
    return llm.invoke(question_prompt).content
