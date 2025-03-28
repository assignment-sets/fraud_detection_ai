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
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        # print("✅ LLM Instance Initialized Successfully!")

    def _get_system_prompt(self) -> str:
        return """
        You are a fraud detection assistant that helps users identify potential fraud in digital communications.
        Your job is to analyze the user's query and determine what type of content it contains (email, SMS, URL, news) and check for signs of fraud.

        Follow these steps for each user input:
        1. Analyze the content to determine what type(s) it contains: email, SMS, URL, news, or a combination
        2. Extract any URLs present in the content
        3. Use the appropriate fraud detection tools based on content type
        4. Provide a clear explanation of your findings

        Available tools:
        - check_fraud_email_tool: Checks if an email appears to be fraudulent
        - check_fraud_sms_tool: Checks if an SMS appears to be fraudulent
        - check_fraud_url_tool: Checks if a URL appears to be malicious or phishing
        - fetch_real_time_news_tool: Retrieves recent news articles on a topic to verify if news is legitimate

        For news verification:
        When a user asks about news, use the fetch_real_time_news_tool to get recent headlines. Then compare the user's claim with the fetched news:
        - If the fetched news contains information that matches or supports the user's claim, the news is likely legitimate
        - If the fetched news contradicts the user's claim or has no mention of the topic, the news is likely fake
        - Always explain your reasoning based on the specific news articles you retrieved

        Be thorough in your analysis and explain your reasoning.
        """

# Example (for testing if run standalone)
if __name__ == "__main__":
    llm_instance = LLMInstance()
    response = llm_instance.llm.invoke("hello how you doing")
    print(response)
