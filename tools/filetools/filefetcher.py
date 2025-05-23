import io
import requests

class FileFetcher:
    DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

    @classmethod
    def get_file(cls, task_id: str) -> io.BytesIO:
        """
        Get the file from the server.
        :param task_id: The task id.
        :return: The file as a BytesIO object.
        """
        url = f"{cls.DEFAULT_API_URL}/files/{task_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return io.BytesIO(response.content)
        else:
            raise Exception(f"Error getting file: {response.status_code}")