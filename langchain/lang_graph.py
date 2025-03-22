from langgraph.graph import StateGraph, START, END
from typing import Optional, List, Literal
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.llm_instance import LLMInstance
from langchain.tools import fetch_real_time_news_tool, check_fraud_url_tool, check_fraud_sms_tool, check_fraud_email_tool

# ------------------ Initialize the LLM instance ----------------------

llm_instance = LLMInstance()

# Access the LLM and system prompt
llm = llm_instance.llm
SYSTEM_PROMPT = llm_instance.system_prompt

print("✅ LLM and System Prompt imported and ready!")

# ------------------ Bind Tools to LLM ----------------------

tools = [
    fetch_real_time_news_tool,
    check_fraud_url_tool,
    check_fraud_sms_tool,
    check_fraud_email_tool
]

tools_by_name = {tool.name: tool for tool in tools}

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

print("✅ Tools successfully bound to LLM!")

# ------------------ def workflow state ----------------------

class State(TypedDict):
    user_input: str  # User's query
    images: List[str]  # List of image URLs/Base64 strings
    messages: List  # Stores the chat history (System, Human, AI messages)
    decision: Optional[str]  # What step to take next
    action: Optional[str]  # The action currently being executed
    available_tools: List[str]  # Names of available tools
    extracted_text: Optional[str]  # Stores extracted text (if any)
    summary: Optional[str]  # Stores summarized text (if any)
    pdf: Optional[bytes]  # Stores generated PDF (if any)