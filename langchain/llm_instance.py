from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import langchain
import os

# for running as module
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

class LLMInstance:
    def __init__(self):
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.system_prompt = self._get_system_prompt()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        # print("✅ LLM Instance Initialized Successfully!")

    def _get_system_prompt(self) -> str:
        return """
        hey you are a drunkard who cusses a lot irrespective of what anyone says
        """

# Example (for testing if run standalone)
if __name__ == "__main__":
    llm_instance = LLMInstance()
    response = llm_instance.llm.invoke("hello how you doing")
    print(response)
