import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import QuestionRequest, QuestionResponse
from app.agent.graph import agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title="Country Information AI Agent",
    description="Ask natural-language questions about countries.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse, tags=["agent"])
async def ask(request: QuestionRequest):
    """
    Submit a natural-language question about one or more countries.

    Examples:
    - "What is the population of Germany?"
    - "What currency does Japan use?"
    - "Compare the area and population of India and China."
    """
    # Convert history entries to plain dicts for the agent state
    history = [entry.model_dump() for entry in request.history]

    result = await agent.ainvoke(
        {
            "query": request.question,
            "history": history,
            "countries": [],
            "fields": [],
            "raw_data": [],
            "answer": "",
            "error": None,
        }
    )

    return QuestionResponse(
        answer=result["answer"],
        countries=result.get("countries", []),
        fields=result.get("fields", []),
        error=result.get("error"),
    )
