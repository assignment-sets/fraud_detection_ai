from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.workflow_graph import workflow_graph
from mangum import Mangum


# Define input schema using Pydantic
class AnalyzeRequest(BaseModel):
    user_query: str = Field(..., min_length=10, description="User's input query")


# Define response schema (optional for docs and clarity)
class AnalyzeResponse(BaseModel):
    final_reasoning_summary: Optional[str]
    is_fraud_email: bool
    is_fraud_sms: bool
    is_fraud_url: bool
    is_fake_news: bool
    is_irrelevant_input: bool
    actions_taken: List[str]


app = FastAPI(
    title="Analyzer API",
    version="1.0",
    description="API for analyzing user queries and detecting fraudulent content",
)
handler = Mangum(app) 

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_query(payload: AnalyzeRequest):
    user_query = payload.user_query.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="user_query must not be empty.")

    # Prepare initial state for the workflow
    initial_state = {
        "user_query": user_query,
        "input_types": [],
        "messages": [],
        "available_tools": [
            "check_fraud_email_tool",
            "check_fraud_sms_tool",
            "check_fraud_url_tool",
            "fetch_real_time_news_tool",
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
        "is_irrelevant_input": None,
    }

    try:
        final_state = workflow_graph.invoke(initial_state)

        return AnalyzeResponse(
            final_reasoning_summary=final_state.get("final_reasoning_summary"),
            is_fraud_email=bool(final_state.get("is_fraud_email")),
            is_fraud_sms=bool(final_state.get("is_fraud_sms")),
            is_fraud_url=bool(final_state.get("is_fraud_url")),
            is_fake_news=bool(final_state.get("is_fake_news")),
            is_irrelevant_input=bool(final_state.get("is_irrelevant_input")),
            actions_taken=final_state.get("list_of_actions") or [],
        )

    except Exception as e:
        print("❌ Error occurred:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
