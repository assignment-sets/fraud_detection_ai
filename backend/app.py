from langchain.workflow_graph import workflow_graph
from flask import Flask, request, jsonify
from pprint import pprint
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    print(f"decision: {state.get('decision')}")
    
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
    print("\n===== END OF STATE =====")
    

@app.route('/analyze', methods=['POST'])
def analyze_query():
    try:
        data = request.json
        user_query = data.get("user_query")

        if not user_query:
            return jsonify({"error": "user_query is required"}), 400

        initial_state = {
            "user_query": user_query,
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
        # pretty_print_state(final_state)

        # Optional: Just return the final reasoning summary and key results in response
        response = {
            "final_reasoning_summary": final_state.get("final_reasoning_summary"),
            "is_fraud_email": bool(final_state.get("is_fraud_email")),  # Ensure it's always a bool
            "is_fraud_sms": bool(final_state.get("is_fraud_sms")),
            "is_fraud_url": bool(final_state.get("is_fraud_url")),
            "is_fake_news": bool(final_state.get("is_fake_news")),
            "is_irrelevant_input": bool(final_state.get("is_irrelevant_input")),
            "actions_taken": final_state.get("list_of_actions") or [],  # Ensure list is not None
        }

        return jsonify(response)

    except Exception as e:
        print("❌ Error occurred during execution:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)