from langgraph.graph import StateGraph, START, END
from typing import Optional, List, Literal, Dict, TypedDict
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.llm_instance import LLMInstance
from langchain.tools import fetch_real_time_news_tool, check_fraud_url_tool, check_fraud_sms_tool, check_fraud_email_tool
import re

# ------------------ Define State ----------------------

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

# ------------------ Initialize the LLM instance ----------------------

llm_instance = LLMInstance()

# Access the LLM and system prompt
llm = llm_instance.llm
SYSTEM_PROMPT = """You are a fraud detection assistant that helps users identify potential fraud in digital communications.
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

print("✅ LLM and System Prompt initialized!")

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

# ------------------ Define Nodes ----------------------

def llm_node(state: State) -> State:
    """
    LLM node and deciding brain for the fraud detection graph.
    
    This node:
    1. Analyzes the user query to identify input types (email, SMS, URL, news)
    2. Extracts URLs from the query
    3. Updates the state with this information
    4. Decides which tools to use
    5. Provides a response to the user
    6. For news queries, analyzes returned news articles to check if user's claim is supported
    """
    messages = state["messages"]
    
    # Initialize state fields if they don't exist
    if "input_types" not in state or not state["input_types"]:
        state["input_types"] = []
    if "url_list" not in state or not state["url_list"]:
        state["url_list"] = []
    if "list_of_actions" not in state:
        state["list_of_actions"] = []
    if "executed_tools" not in state:
        state["executed_tools"] = []
    if "is_fake_news" not in state:
        state["is_fake_news"] = None
    
    # Ensure the system message is only added once
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    
    # First time processing the user query
    if len(messages) == 1:  # Only system message exists
        user_query = state["user_query"]
        messages.append(HumanMessage(content=user_query))
        
        # Extract URLs from user query using regex
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        extracted_urls = re.findall(url_pattern, user_query)
        if extracted_urls:
            state["url_list"] = extracted_urls
            state["list_of_actions"].append(f"Extracted {len(extracted_urls)} URLs from user query")
        
        # Add instruction for the LLM to identify input types
        messages.append(SystemMessage(content=f"""
        Before responding to the user, analyze the query and:
        1. Determine what types of content this might be (email, SMS, URL, news)
        2. I've already extracted these URLs from the query: {extracted_urls if extracted_urls else 'None'}
        3. Decide which tools you need to use for fraud detection
        4. Update your understanding based on tool results
        """))
    
    # Check if news has been fetched and needs verification
    elif "fetched_news" in state and state["fetched_news"] and state["is_fake_news"] is None:
        # News has been fetched but not yet verified - prompt LLM to analyze
        news_data = state["fetched_news"]
        news_context = ""
        
        for i, (title, details) in enumerate(news_data.items(), 1):
            news_context += f"Article {i}: {title}\n"
            if "description" in details:
                news_context += f"Description: {details['description']}\n"
            if "source" in details:
                news_context += f"Source: {details['source']}\n"
            news_context += "\n"
        
        # Add special instruction for news verification
        messages.append(SystemMessage(content=f"""
        News verification required:
        
        The user query was: "{state['user_query']}"
        
        I've retrieved the following news articles:
        
        {news_context}
        
        Please analyze these articles and determine:
        1. Do they support, contradict, or have no relation to the user's claim?
        2. Based on this analysis, is the user's claim likely to be legitimate news or fake news?
        3. Provide your reasoning with specific references to the articles
        
        Remember: If none of the articles mention the topic or they contradict the claim, it's more likely to be fake news.
        """))
        
        # Invoke LLM for news verification
        try:
            news_verification_response = llm.invoke(messages)
            messages.append(news_verification_response)
            
            # Determine if news is fake based on the response
            content = news_verification_response.content.lower()
            if "fake" in content or "false" in content or "not supported" in content or "no evidence" in content:
                state["is_fake_news"] = True
                state["list_of_actions"].append("Determined news is likely fake based on article analysis")
            elif "legitimate" in content or "supported" in content or "confirmed" in content or "real" in content:
                state["is_fake_news"] = False
                state["list_of_actions"].append("Determined news is likely legitimate based on article analysis")
            else:
                # If unclear, default to considering it suspicious
                state["is_fake_news"] = True
                state["list_of_actions"].append("News verification inconclusive - treating as potentially fake")
            
            # Set this as the final reasoning summary if not already set
            if not state.get("final_reasoning_summary"):
                state["final_reasoning_summary"] = news_verification_response.content
            
            return {**state, "messages": messages}
            
        except Exception as e:
            print("❌ ERROR in news verification:", str(e))
            messages.append(AIMessage(content="I encountered an error while verifying the news. Treating it as potentially suspicious."))
            state["is_fake_news"] = True
            state["list_of_actions"].append(f"Error during news verification: {str(e)}")
            return {**state, "messages": messages}
    
    # If this message follows tool execution, add appropriate context
    elif len(messages) > 1 and any(isinstance(msg, ToolMessage) for msg in messages[-3:]):
        # Check if we need to process fetched news
        needs_news_verification = False
        for msg in messages[-3:]:
            if isinstance(msg, ToolMessage) and "fetch_real_time_news_tool" in msg.content:
                needs_news_verification = True
                break
        
        if needs_news_verification and "fetched_news" in state and state["fetched_news"]:
            # Don't add instructions - we'll handle news verification in the next loop iteration
            pass
        else:
            # Add a system message to guide the LLM to use the tool results
            messages.append(SystemMessage(content="""
            Based on the tool results you've received:
            1. Update your assessment of potential fraud
            2. Decide if you need more information from other tools
            3. If you've gathered enough information, provide a final assessment to the user
            """))
    
    # Invoke LLM with messages (except in the news verification case which is handled above)
    if not (("fetched_news" in state and state["fetched_news"] and state["is_fake_news"] is None)):
        try:
            ai_response = llm_with_tools.invoke(messages)
            messages.append(ai_response)
            
            # If this is the first AI response, try to extract input types from it
            if "input_types" in state and not state["input_types"]:
                content = ai_response.content.lower()
                possible_types = ["email", "sms", "url", "news"]
                detected_types = [t for t in possible_types if t in content]
                if detected_types:
                    state["input_types"] = detected_types
                    state["list_of_actions"].append(f"Detected input types: {', '.join(detected_types)}")
            
            # Check if the AI seems to be providing a final assessment
            if not any(getattr(ai_response, "tool_calls", [])) and ("fraud" in ai_response.content.lower() or "legitimate" in ai_response.content.lower()):
                if not state.get("final_reasoning_summary"):
                    state["final_reasoning_summary"] = ai_response.content
        
        except Exception as e:
            print("❌ ERROR in LLM invocation:", str(e))
            messages.append(AIMessage(content=f"I encountered an error while analyzing your query. Please try again."))
    
    return {
        **state,
        "messages": messages,
    }

def tool_node(state: State) -> State:
    """
    Calls the correct tool(s) based on the LLM's tool call response.
    
    This node:
    1. Identifies which tools the LLM wants to use
    2. Executes those tools with appropriate arguments
    3. Updates the state with tool results
    4. Tracks which tools have been executed
    """
    messages = state["messages"]
    last_message = messages[-1]  # Last AI message
    
    # Extract tool calls from the last AI response
    tool_calls = getattr(last_message, "tool_calls", [])
    
    if not tool_calls:
        print("No tool calls detected in the last message.")
        return state
    
    # Initialize tracking fields if they don't exist
    if "executed_tools" not in state:
        state["executed_tools"] = []
    if "list_of_actions" not in state:
        state["list_of_actions"] = []
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        
        # Skip if this exact tool with these exact args has already been executed
        tool_signature = f"{tool_name}:{str(tool_args)}"
        if tool_signature in state["executed_tools"]:
            results.append(ToolMessage(
                content=f"Note: This exact tool call has already been executed. Using previous results.",
                tool_call_id=tool_id
            ))
            continue
        
        # Execute the tool
        if tool_name and tool_name in tools_by_name:
            try:
                # Execute tool based on name
                result = tools_by_name[tool_name].invoke(tool_args)
                results.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                
                # Update state based on tool results
                if tool_name == "check_fraud_url_tool":
                    state["is_fraud_url"] = result
                    state["list_of_actions"].append(f"Executed check_fraud_url_tool: Result={result}")
                elif tool_name == "check_fraud_sms_tool":
                    state["is_fraud_sms"] = result
                    state["list_of_actions"].append(f"Executed check_fraud_sms_tool: Result={result}")
                elif tool_name == "check_fraud_email_tool":
                    state["is_fraud_email"] = result
                    state["list_of_actions"].append(f"Executed check_fraud_email_tool: Result={result}")
                elif tool_name == "fetch_real_time_news_tool":
                    state["fetched_news"] = result
                    state["list_of_actions"].append(f"Executed fetch_real_time_news_tool: Articles fetched")
                    # Reset is_fake_news to None to indicate news fetched but not yet verified
                    state["is_fake_news"] = None
                
                # Track executed tools to prevent redundant calls
                state["executed_tools"].append(tool_signature)
                
            except Exception as e:
                error_msg = f"Tool error: {str(e)}"
                results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                state["list_of_actions"].append(f"Error executing {tool_name}: {str(e)}")
        else:
            error_msg = f"Error: Unknown tool '{tool_name}'"
            results.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
            state["list_of_actions"].append(error_msg)
    
    # Append tool execution results to messages
    messages.extend(results)
    
    return {**state, "messages": messages}

def should_continue(state: State) -> Literal["tool_node", END]: # type: ignore
    """
    Determines if another tool call is needed or if execution should end.
    
    This node:
    1. Checks if the last AI message contains tool calls
    2. Decides whether to continue to the tool node or end execution
    3. Ends if all necessary tools have been executed or if a final assessment has been made
    4. Special handling for news verification to ensure it completes properly
    """
    messages = state["messages"]
    
    if not messages:
        return END
    
    # Special case: If news has been fetched but not verified, continue to LLM node
    if "fetched_news" in state and state["fetched_news"] and state.get("is_fake_news") is None:
        return "llm"
    
    # Find the last AI message
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
    
    if not last_ai_message:
        return END
    
    # Check if this AI message has tool calls
    tool_calls = getattr(last_ai_message, "tool_calls", [])
    
    # If the last message in the conversation is an AI message with tool calls,
    # we need to execute those tools
    if tool_calls and messages[-1] == last_ai_message:
        return "tool_node"
    
    # If we have a final reasoning summary, we're done
    if "final_reasoning_summary" in state and state["final_reasoning_summary"]:
        return END
    
    # If we've executed all possible tools based on input types, we're done
    input_types = state.get("input_types", [])
    executed_all_relevant_tools = False
    
    if "url" in input_types and "is_fraud_url" in state:
        executed_all_relevant_tools = True
    if "email" in input_types and "is_fraud_email" in state:
        executed_all_relevant_tools = True
    if "sms" in input_types and "is_fraud_sms" in state:
        executed_all_relevant_tools = True
    if "news" in input_types and "is_fake_news" in state and state["is_fake_news"] is not None:
        executed_all_relevant_tools = True
    
    # If the last message is a tool response and all relevant tools executed, we're done
    if isinstance(messages[-1], ToolMessage) and executed_all_relevant_tools:
        return END
    
    # By default, continue the conversation
    return END

# ------------------ Create Graph ----------------------

# Initialize the workflow graph
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tool_node", tool_node)

# Define edges to control the workflow
graph_builder.add_edge(START, "llm")  # Start with the LLM node

graph_builder.add_conditional_edges(
    "llm",
    should_continue,  # Determines whether to call a tool or stop
    {
        "tool_node": "tool_node",  # If tools are required, execute them
        END: END,  # Otherwise, stop execution
    },
)

# Return to LLM after tool execution
graph_builder.add_edge("tool_node", "llm")

# Compile the graph
workflow_graph = graph_builder.compile()

if __name__ == '__main__':
    try:
        initial_state = {
            "user_query": """
                Dear Support Team,

                I hope this message finds you well. I received an email with a link: https://unstop.com, saying elon musk is dead.

                Furthermore, I found a few articles on https://chatgpt.com which doesnt say anythong about this news ?
                i am not sure if elon is dead
                Best regards,
                User
            """,
            "input_types": [],
            "messages": [],
            "available_tools": [
                "check_fraud_email_tool",
                "check_fraud_sms_tool",
                "check_fraud_url_tool",
                "fetch_real_time_news_tool"
            ],
            "decision": None,
            "list_of_actions": [],
            "is_fraud_url": None,
            "is_fraud_sms": None,
            "is_fraud_email": None,
            "fetched_news": None,
            "final_reasoning_summary": None,
            "url_list": [],
            "executed_tools": [],
            "is_fake_news": None
        }

        final_state = workflow_graph.invoke(initial_state)
        print("===== Final State =====")
        print(final_state)

    except Exception as e:
        print("❌ Error occurred during execution:", str(e))

