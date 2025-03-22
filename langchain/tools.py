from typing import List
from langchain_core.tools import tool
from utils.fraud_detection_handler import FraudDetectionHandler

handler = FraudDetectionHandler()


@tool
def check_fraud_email_tool(email_text: str) -> bool:
    """
    Detect if an email text is potentially fraudulent.
    :param email_text: The content of the email.
    :return: True if fraud, False if legitimate.
    """
    try:
        return handler.check_fraud_email(email_text)
    except Exception as e:
        print(f"Error checking fraud email: {str(e)}")
        return False


@tool
def check_fraud_sms_tool(sms_text: str) -> bool:
    """
    Detect if an SMS text is potentially fraudulent.
    :param sms_text: The content of the SMS.
    :return: True if fraud, False if legitimate.
    """
    try:
        return handler.check_fraud_sms(sms_text)
    except Exception as e:
        print(f"Error checking fraud SMS: {str(e)}")
        return False


@tool
def check_fraud_url_tool(url_list: List[str]) -> bool:
    """
    Detect if a given list of URLs are potentially phishing or safe.
    :param url_list: The list of URLs to check.
    :return: True if majority are phishing, False if majority are safe.
    """
    try:
        results = [handler.check_fraud_url(url) for url in url_list]
        return results.count(True) > results.count(False)
    except Exception as e:
        print(f"Error checking fraud URLs: {str(e)}")
        return False


@tool
def fetch_real_time_news_tool(query: str) -> dict:
    """
    Fetch real-time news articles for a given query.
    :param query: Search keyword.
    :return: Dictionary of news articles.
    """
    try:
        return handler.fetch_real_time_news(query)
    except Exception as e:
        return {"error": f"Error fetching news: {str(e)}"}


if __name__ == '__main__':
    print("tools built")
