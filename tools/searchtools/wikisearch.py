from langchain.utilities import WikipediaAPIWrapper
from langchain.requests import RequestsWrapper
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.schema import Document
from langchain.tools import Tool

class WikipediaSearchTool:
    def __init__(self):
        self.wikipedia = WikipediaAPIWrapper(doc_content_chars_max=1000, top_k_results=1)
        self.requests_wrapper = RequestsWrapper(headers={"User-Agent": "Mozilla/5.0"})
        self.soup_transformer = BeautifulSoupTransformer()
    
    def __call__(self, query: str) -> str:
        # Step 1: Get Wikipedia page title from summary
        result = self.wikipedia.load(query)
        page_title = result[0].metadata['title']
        url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        # Step 2: Fetch HTML content
        html_content = self.requests_wrapper.get(url)
        # Step 3: Transform HTML to plain text
        document = Document(page_content=html_content)
        transformed_data = self.soup_transformer.transform_documents([document])
        # Step 4: Return the plain text
        return transformed_data[0].page_content

# Instantiate the tool
wiki_search_tool = WikipediaSearchTool()

# Register as a LangChain Tool
wikipedia_search_langchain_tool = Tool(
    name="wikipedia_search_tool",
    func=wiki_search_tool,
    description="Fetches Wikipedia content for a given query."
)

# abc = wikipedia_search_langchain_tool.run("Mercedes Sosa")  # Example usage
# def save_transformed_text_to_file(transformed_text: str, file_name: str = "transformed_output.txt") -> None:
#     """
#     Save the transformed text to a file.

#     Args:
#         transformed_text (str): The transformed text to save.
#         file_name (str): The name of the file to save the text to (default: "transformed_output.txt").
#     """
#     with open(file_name, "w", encoding="utf-8") as file:
#         file.write(transformed_text)
#     print(f"Transformed text saved to {file_name}")
# save_transformed_text_to_file(abc, "mercedes_sosa_transformed.txt")