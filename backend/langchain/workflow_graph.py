from langgraph.graph import StateGraph, START, END
from typing import Optional, List, Literal, Dict, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.llm_instance import LLMInstance
from langchain.tools import fetch_real_time_news_tool, check_fraud_url_tool, check_fraud_sms_tool, check_fraud_email_tool
from pprint import pprint
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
    news_verification_requested: Optional[bool]  # Flag to track if news verification has been requested
    is_irrelevant_input: Optional[bool]  # Flag to track if the input is irrelevant to fraud detection

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

IMPORTANT: If the user input is NOT related to fraud detection or doesn't contain any email, SMS, URL, or news to analyze, respond with a firm but polite message explaining that you are a fraud detection assistant and can only help with analyzing digital communications for fraud. DO NOT attempt to answer unrelated questions or engage with irrelevant input. Make it clear that the user needs to provide content for fraud analysis.

IMPORTANT: After completing your analysis, ALWAYS provide a final conclusion that summarizes your findings. This will be used as the final reasoning summary.
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
    7. Handles irrelevant inputs that don't match any of the expected types
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
    if "news_verification_requested" not in state:
        state["news_verification_requested"] = False
    if "final_reasoning_summary" not in state:
        state["final_reasoning_summary"] = None
    if "is_irrelevant_input" not in state:
        state["is_irrelevant_input"] = False
    if "is_fraud_url" not in state:
        state["is_fraud_url"] = None
    if "is_fraud_sms" not in state:
        state["is_fraud_sms"] = None
    if "is_fraud_email" not in state:
        state["is_fraud_email"] = None
    
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
        5. If the user query doesn't contain any email, SMS, URL, or news to analyze, mark it as irrelevant and respond accordingly
        6. End your response with a clear FINAL CONCLUSION paragraph summarizing your findings.
        
        IMPORTANT: If the input does not appear to be related to fraud detection (email, SMS, URL, or news), respond with a firm but polite message stating that you are a fraud detection assistant and can only help with analyzing potential fraud in digital communications. DO NOT engage with irrelevant questions or inputs.
        """))
    
    # Check if news has been fetched and needs verification
    elif "fetched_news" in state and state["fetched_news"] and (state["is_fake_news"] is None) and not state["news_verification_requested"]:
        # Mark that we've requested news verification to prevent duplicate requests
        state["news_verification_requested"] = True
        state["list_of_actions"].append("Requesting news verification")
        
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
        
        IMPORTANT: End your analysis with a clear FINAL CONCLUSION paragraph that states whether the news is FAKE or LEGITIMATE.
        This will be used as the final reasoning summary.
        """))
        
        # Invoke LLM for news verification
        try:
            news_verification_response = llm.invoke(messages)
            messages.append(news_verification_response)
            
            # Determine if news is fake based on the response
            content = news_verification_response.content.lower()
            
            # Extract final conclusion for final_reasoning_summary
            final_conclusion = ""
            if "final conclusion" in content:
                # Get text after "final conclusion"
                final_conclusion = content.split("final conclusion")[1].strip()
                # If it contains a colon, take what's after the colon
                if ":" in final_conclusion:
                    final_conclusion = final_conclusion.split(":", 1)[1].strip()
                # Get just the first paragraph
                if "\n\n" in final_conclusion:
                    final_conclusion = final_conclusion.split("\n\n")[0].strip()
                elif "\n" in final_conclusion:
                    final_conclusion = final_conclusion.split("\n")[0].strip()
            
            # If no specific final conclusion section, use the last paragraph
            if not final_conclusion:
                paragraphs = content.split("\n\n")
                final_conclusion = paragraphs[-1].strip()
                
            # Save this as the final reasoning summary 
            state["final_reasoning_summary"] = final_conclusion
            state["list_of_actions"].append("Extracted final reasoning summary from news verification")
            
            # Determine is_fake_news value
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
            
            return {**state, "messages": messages}
            
        except Exception as e:
            print("❌ ERROR in news verification:", str(e))
            messages.append(AIMessage(content="I encountered an error while verifying the news. Treating it as potentially suspicious."))
            state["is_fake_news"] = True
            state["list_of_actions"].append(f"Error during news verification: {str(e)}")
            state["final_reasoning_summary"] = "Error during news verification. Treating as potentially suspicious."
            return {**state, "messages": messages}
    
    # If this message follows tool execution, add appropriate context
    elif len(messages) > 1 and any(isinstance(msg, ToolMessage) for msg in messages[-3:]):
        # Add a system message to guide the LLM to use the tool results
        messages.append(SystemMessage(content="""
        Based on the tool results you've received:
        1. Update your assessment of potential fraud
        2. Decide if you need more information from other tools
        3. If you've gathered enough information, provide a final assessment to the user
        4. End your response with a clear FINAL CONCLUSION paragraph summarizing your findings.
        """))
    
    # Invoke LLM with messages (except in the news verification case which is handled above)
    if not (("fetched_news" in state and state["fetched_news"] and state["is_fake_news"] is None and not state["news_verification_requested"])):
        try:
            ai_response = llm_with_tools.invoke(messages)
            messages.append(ai_response)
            
            # If this is the first AI response, try to extract input types from it
            if "input_types" in state and not state["input_types"]:
                content = ai_response.content.lower()
                possible_types = ["email", "sms", "url", "news"]
                detected_types = [t for t in possible_types if t in content]
                
                # Check if the response indicates this is an irrelevant input
                irrelevant_indicators = [
                    "i am a fraud detection assistant",
                    "i can only help with",
                    "i'm a fraud detection assistant",
                    "i'm designed to analyze",
                    "not related to fraud detection",
                    "doesn't contain any email, sms, url, or news",
                    "not an email, sms, url, or news",
                    "please provide a digital communication",
                    "i can only assist with fraud detection",
                    "not within my scope"
                ]
                
                is_irrelevant = any(indicator in content for indicator in irrelevant_indicators)
                
                if is_irrelevant:
                    state["is_irrelevant_input"] = True
                    state["input_types"] = ["irrelevant"]
                    state["list_of_actions"].append("Detected irrelevant input not related to fraud detection")
                    
                    # Extract the final reasoning summary for irrelevant input
                    if "final conclusion" in content:
                        final_conclusion = content.split("final conclusion")[1].strip()
                        if ":" in final_conclusion:
                            final_conclusion = final_conclusion.split(":", 1)[1].strip()
                        if "\n\n" in final_conclusion:
                            final_conclusion = final_conclusion.split("\n\n")[0].strip()
                        elif "\n" in final_conclusion:
                            final_conclusion = final_conclusion.split("\n")[0].strip()
                        state["final_reasoning_summary"] = final_conclusion
                    else:
                        # Default message for irrelevant inputs
                        state["final_reasoning_summary"] = "I am a fraud detection assistant. I can only help with analyzing potential fraud in digital communications (emails, SMS, URLs, or news)."
                    
                    state["list_of_actions"].append("Set final reasoning summary for irrelevant input")
                    
                elif detected_types:
                    state["input_types"] = detected_types
                    state["list_of_actions"].append(f"Detected input types: {', '.join(detected_types)}")
                    # If "news" is detected but not yet checked, set is_fake_news to None to indicate it needs verification
                    if "news" in detected_types and state["is_fake_news"] is None:
                        state["is_fake_news"] = None
                        state["list_of_actions"].append("Marked news for verification")
            
            # Check if we need to extract a final reasoning summary
            if not getattr(ai_response, "tool_calls", []) and not state["final_reasoning_summary"]:
                content = ai_response.content.lower()
            
                # Look for a final conclusion section
                if "final conclusion" in content:
                    # Get text after "final conclusion"
                    final_conclusion = content.split("final conclusion")[1].strip()
                    # If it contains a colon, take what's after the colon
                    if ":" in final_conclusion:
                        final_conclusion = final_conclusion.split(":", 1)[1].strip()
                    # Get just the first paragraph
                    if "\n\n" in final_conclusion:
                        final_conclusion = final_conclusion.split("\n\n")[0].strip()
                    elif "\n" in final_conclusion:
                        final_conclusion = final_conclusion.split("\n")[0].strip()
                    
                    state["final_reasoning_summary"] = final_conclusion
                    state["list_of_actions"].append("Extracted final reasoning summary from conclusion section")
                    
                    # Check if we need to update any fraud flags based on the conclusion
                    if "url" in state.get("input_types", []) and state["is_fraud_url"] is None:
                        if "fraud" in final_conclusion or "malicious" in final_conclusion or "phishing" in final_conclusion:
                            state["is_fraud_url"] = True
                            state["list_of_actions"].append("Updated is_fraud_url to True based on conclusion")
                        elif "safe" in final_conclusion or "legitimate" in final_conclusion:
                            state["is_fraud_url"] = False
                            state["list_of_actions"].append("Updated is_fraud_url to False based on conclusion")
                    
                    if "sms" in state.get("input_types", []) and state["is_fraud_sms"] is None:
                        if "fraud" in final_conclusion or "scam" in final_conclusion:
                            state["is_fraud_sms"] = True
                            state["list_of_actions"].append("Updated is_fraud_sms to True based on conclusion")
                        elif "safe" in final_conclusion or "legitimate" in final_conclusion:
                            state["is_fraud_sms"] = False
                            state["list_of_actions"].append("Updated is_fraud_sms to False based on conclusion")
                    
                    if "email" in state.get("input_types", []) and state["is_fraud_email"] is None:
                        if "fraud" in final_conclusion or "phishing" in final_conclusion or "scam" in final_conclusion:
                            state["is_fraud_email"] = True
                            state["list_of_actions"].append("Updated is_fraud_email to True based on conclusion")
                        elif "safe" in final_conclusion or "legitimate" in final_conclusion:
                            state["is_fraud_email"] = False
                            state["list_of_actions"].append("Updated is_fraud_email to False based on conclusion")
        
        except Exception as e:
            print("❌ ERROR in LLM invocation:", str(e))
            messages.append(AIMessage(content=f"I encountered an error while analyzing your query. Please try again."))
            state["final_reasoning_summary"] = "Error during analysis. Please try again."
    
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
                    # Reset news_verification_requested flag
                    state["news_verification_requested"] = False
                
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

def should_continue(state: State) -> Literal["tool_node", "llm", END]: # type: ignore
    """
    Determines if another tool call is needed or if execution should end.
    
    This node:
    1. Checks if the last AI message contains tool calls
    2. Decides whether to continue to the tool node or end execution
    3. Ends if all necessary tools have been executed or if a final assessment has been made
    4. Special handling for news verification to ensure it completes properly
    5. Ends immediately if the input is determined to be irrelevant
    """
    messages = state["messages"]
    
    if not messages:
        return END
    
    # If the input is irrelevant and we have a final reasoning summary, end immediately
    if state.get("is_irrelevant_input", False) and state.get("final_reasoning_summary"):
        return END
    
    # Special case: If news has been fetched but not verified, continue to LLM node
    if ("fetched_news" in state and state["fetched_news"] and 
        state.get("is_fake_news") is None and not state.get("news_verification_requested", False)):
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
    
    # If news is in input_types but no news verification has been done, make sure we don't end prematurely
    if ("news" in state.get("input_types", []) and 
        state.get("is_fake_news") is None and 
        "fetched_news" in state and state["fetched_news"]):
        return "llm"  # Force another LLM call to handle news verification
    
    # If we've executed all possible tools based on input types, we're done
    input_types = state.get("input_types", [])
    executed_all_relevant_tools = False
    
    if not input_types:
        # If we can't determine input types yet, we need more processing
        executed_all_relevant_tools = False
    elif "irrelevant" in input_types:
        # Irrelevant inputs don't need tool execution
        executed_all_relevant_tools = True
    else:
        # For each detected input type, check if the relevant tool has been executed
        type_checks = []
        if "url" in input_types:
            type_checks.append(state.get("is_fraud_url") is not None)
        if "email" in input_types:
            type_checks.append(state.get("is_fraud_email") is not None)
        if "sms" in input_types:
            type_checks.append(state.get("is_fraud_sms") is not None)
        if "news" in input_types:
            type_checks.append(state.get("is_fake_news") is not None)
        
        # Only consider all tools executed if we have at least one type and all types have been checked
        executed_all_relevant_tools = len(type_checks) > 0 and all(type_checks)
    
    # If the last message is a tool response and all relevant tools executed, we need a final LLM call for conclusion
    if isinstance(messages[-1], ToolMessage) and executed_all_relevant_tools:
        return "llm"  # One more LLM call to get the final reasoning summary
    
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
        "llm": "llm",  # Return to LLM node for news verification
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
            "user_query": r"""SAM-OFFER FOR YOU. I can get you
    approved for a no-interest credit
    card today ONLY. Learn more:
    aihfori.it.car.com
            """,
            "input_types": [],
            "messages": [],
            "available_tools": [
                "check_fraud_email_tool",
                "check_fraud_sms_tool",
                "check_fraud_url_tool",
                "fetch_real_time_news_tool"
            ],
            "list_of_actions": [],
            "is_fraud_url": None,
            "is_fraud_sms": None,
            "is_fraud_email": None,
            "fetched_news": None,
            "final_reasoning_summary": None,
            "url_list": [],
            "executed_tools": [],
            "is_fake_news": None,
            "news_verification_requested": None,
            "is_irrelevant_input": None
        }

        final_state = workflow_graph.invoke(initial_state)

        def pretty_print_state(state: dict):
            print("\n===== FINAL STATE =====")
            
            print(f"user_query:\n{state.get('user_query', '')}\n")
            print(f"input_types: {state.get('input_types', [])}")
            
            print("\nmessages:")
            for i, msg in enumerate(state.get("messages", []), start=1):
                msg_type = getattr(msg, 'type', 'Unknown')
                msg_preview = msg.content.strip()[:300].replace("\n", " ") + ("..." if len(msg.content.strip()) > 300 else "")
                print(f"  {i}. [{msg_type}] {msg_preview}")
            
            print(f"\navailable_tools: {state.get('available_tools', [])}")
            
            print("\nlist_of_actions:")
            actions = state.get("list_of_actions", [])
            if actions:
                for i, action in enumerate(actions, start=1):
                    print(f"  {i}. {action}")
            else:
                print("  (empty)")
            
            print(f"\nis_fraud_url: {state.get('is_fraud_url')}")
            print(f"is_fraud_sms: {state.get('is_fraud_sms')}")
            print(f"is_fraud_email: {state.get('is_fraud_email')}")
            
            print(f"\nfetched_news:")
            fetched_news = state.get('fetched_news')
            if fetched_news:
                pprint(fetched_news)
            else:
                print("  (empty)")
            
            print(f"\nfinal_reasoning_summary:\n{state.get('final_reasoning_summary')}\n")
            print(f"url_list: {state.get('url_list', [])}")
            
            print("\nexecuted_tools:")
            executed = state.get('executed_tools', [])
            if executed:
                for i, tool in enumerate(executed, start=1):
                    print(f"  {i}. {tool}")
            else:
                print("  (empty)")
            
            print(f"\nis_fake_news: {state.get('is_fake_news')}")
            print(f"\nis_irrelevant_input: {state.get('is_irrelevant_input')}")
            print("\n===== END OF STATE =====")

        pretty_print_state(final_state)

    except Exception as e:
        print("❌ Error occurred during execution:", str(e))