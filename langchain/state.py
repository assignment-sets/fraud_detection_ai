from typing import List, Optional, Dict, TypedDict

class State(TypedDict):
    user_query: str  # The text input from the user (email/text/url to analyze)
    # Could be multiple types detected or provided (e.g., ["email", "url"])
    input_types: List[str]
    # Full conversation history (system prompts, user queries, AI responses)
    messages: List
    # List of tools that the agent can use (like fraud checkers, news fetcher)
    available_tools: List[str]
    # The final decision or intermediate decision (e.g., "likely fraud", "clean", "fetch more data")
    decision: Optional[str]
    # A step-by-step log of actions or reasoning the LLM performed
    list_of_actions: List[str]
    # Result of URL fraud detection (True = fraud, False = clean, None if not checked)
    is_fraud_url: Optional[bool]
    # Result of SMS fraud detection (True/False/None)
    is_fraud_sms: Optional[bool]
    # Result of Email fraud detection (True/False/None)
    is_fraud_email: Optional[bool]
    # Real-time fetched news articles with title, source, etc.
    fetched_news: Optional[Dict[str, Dict[str, str]]]
    # A short LLM-generated explanation of why the final decision was made
    final_reasoning_summary: Optional[str]
    url_list: List[str]  # List of all the extracted urls
    executed_tools: List[str]  # Track which tools have been executed
    is_fake_news: Optional[bool]  # Result of news verification (True=fake, False=legitimate, None=not checked)