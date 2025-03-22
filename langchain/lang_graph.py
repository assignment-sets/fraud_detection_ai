from langgraph.graph import StateGraph, START, END
from typing import Optional, List, Literal, Dict
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.llm_instance import LLMInstance
from langchain.tools import fetch_real_time_news_tool, check_fraud_url_tool, check_fraud_sms_tool, check_fraud_email_tool
from langchain.state import State

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

# ------------------ Define Nodes ----------------------

def llm_node(state: State) -> State:
    """
    LLM node and deciding brain for the fraud detection graph.

    Uses the current conversation state and system prompt to invoke the LLM,  
    decide next actions, and update the state with the LLM's response.
    """
    messages = state["messages"]

    # Ensure the system message is only added once
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    # If this is the first message, append a clarification about tool usage
    if len(messages) == 1:  # Only system message exists
        user_message = state["user_input"]
        messages.append(HumanMessage(content=user_message))
    else:
        # For subsequent messages, just pass along the conversation history
        if len(messages) > 1 and isinstance(messages[-1], ToolMessage):
            # If the last message was a tool response, add a system note
            messages.append(SystemMessage(
                content="Please respond to the user based on the tool results. Only use additional tools if required."))

    # Invoke LLM with messages
    try:
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)
    except Exception as e:
        print("❌ ERROR in LLM invocation:", str(e))
        return state

    return {
        **state,
        "messages": messages,
        "decision": ai_response.content,
    }

def tool_node(state: State) -> State:
    """
    Calls the correct tool based on the LLM's tool call response.
    """
    messages = state["messages"]
    last_message = messages[-1] # Last AI message

    # Extract tool calls from the last AI response
    tool_calls = last_message.tool_calls if hasattr(
        last_message, "tool_calls") else []

    if not tool_calls:
        print("❌ No tool calls detected.")
        messages.append(ToolMessage(content="Error: No tool call detected."))
        return {**state, "messages": messages}

    # field to keep track of executed tools
    if "executed_tools" not in state:
        state["executed_tools"] = []

    results = []
    for idx, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")

        # Check if this tool has already been executed
        if tool_name in state["executed_tools"]:
            print(f"⚠️ Tool {tool_name} has already been executed. Skipping.")
            results.append(ToolMessage(
                content=f"Note: Tool '{tool_name}' has already been executed. Skipping redundant call.",
                tool_call_id=tool_id))
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
                    state["is_fetched_news"] = result
                    state["list_of_actions"].append(f"Executed fetch_real_time_news_tool: Articles fetched")

                # Mark this tool as executed
                state["executed_tools"].append(tool_name)

            except Exception as e:
                results.append(ToolMessage(content=f"Tool error: {str(e)}", tool_call_id=tool_id))
                
        else:
            results.append(ToolMessage(
                content=f"Error: Unknown tool '{tool_name}'",
                tool_call_id=tool_id))

    # Append tool execution results to messages
    messages.extend(results)
    
    return {**state, "messages": messages}


def should_continue(state: State) -> Literal["tool_node", END]:  # type: ignore
    """
    Determines if another tool call is needed or if execution should end.
    Only continues if the last message is an AI message with tool calls.
    """
    messages = state["messages"]

    if not messages:
        return END

    # Find the last AI message
    last_ai_message_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_message_index = i
            break

    if last_ai_message_index == -1:
        return END

    # Check if this AI message has tool calls
    last_ai_message = messages[last_ai_message_index]
    tool_calls = getattr(last_ai_message, "tool_calls", [])

    # Only continue if this AI message has tool calls AND we haven't processed them yet
    # (i.e., the last message isn't a tool response to this AI message)
    if tool_calls and last_ai_message_index == len(messages) - 1:
        return "tool_node"

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