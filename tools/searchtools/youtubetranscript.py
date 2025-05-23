from langchain.tools import Tool
from langchain.document_loaders import YoutubeLoader

class YouTubeTranscriptTool:
    def __call__(self, url: str) -> str:
        loader = YoutubeLoader.from_youtube_url(url)
        docs = loader.load()
        transcript = "\n".join(doc.page_content for doc in docs)
        return transcript

youtube_transcript_tool = Tool(
    name="youtube_transcript_tool",
    func=YouTubeTranscriptTool(),
    description="Fetches the transcript of a YouTube video given its URL."
)