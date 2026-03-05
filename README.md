Country Information AI Agent

Country Information AI Agent is a system that answers natural language questions about countries using public data from the REST Countries API.

The project is implemented with LangGraph, FastAPI, and LangChain. It receives a user question, determines which country information is required, retrieves the relevant data from the API, and generates a clear response.
Architecture
The agent operates through a sequence of processing steps. A user submits a question in natural language. The system first analyzes the question to determine which countries and which types of information are requested. After identifying this intent, the system retrieves the required data from the REST Countries API. Finally, the system generates a readable answer based strictly on the data that was retrieved.

This workflow can be understood as the following process:
User question → intent extraction → data retrieval → answer generation → response returned to the user.

Agent Nodes
The LangGraph agent is divided into several nodes, each responsible for a specific task.

Intent node
This component uses an LLM to analyze the user's question. It extracts the country names mentioned in the query and identifies the specific fields requested, such as population, capital, currency, or region.

Tool node
This node retrieves data from the REST Countries API for every country that was identified in the previous step.
Synthesis node
After data is retrieved, the LLM composes a human readable answer. The response is grounded only in the data returned by the API.

Error node
If any stage fails, this node generates a clear error message so the user understands what went wrong.

Graph Flow
The system typically follows this sequence:
The intent node extracts countries and fields from the user query.
The tool node fetches the required country data from the REST Countries API.
The synthesis node produces the final answer using the fetched information.
If the system cannot identify any countries in the question, the workflow stops and an error message is returned. If the API fails to provide data for all requested countries, the error node is triggered. When only some countries fail to load, the system still produces a response using the available data and includes a note explaining which countries could not be retrieved.

Tech Stack
The project uses the following technologies:
LangGraph: Provides state based orchestration for the agent workflow.
LangChain: Handles interaction with language models and supports multiple LLM providers.
FastAPI: Serves as the REST API framework used to expose the application endpoints.
httpx: An asynchronous HTTP client used to communicate with the REST Countries API.
Pydantic: Used for validating request and response schemas.


Getting Started:
Prerequisites
-Python 3.9 or higher
-An API key for a supported LLM provider such as OpenAI, Anthropic, or OpenRouter.

Setup
-Clone the repository.
-git clone https://github.com/<your-username>/cloudeagle.git
-Navigate to the project directory.
-cd cloudeagle

Create a virtual environment.
-python3 -m venv .venv

Activate the environment.
-source .venv/bin/activate

Install the required dependencies.
-pip install -r requirements.txt

Copy the environment configuration template.
-cp .env.example .env

Open the .env file and add your API key and preferred LLM provider.
Run the Application
Start the FastAPI server using Uvicorn.
-python3 -m uvicorn app.main:app --reload
Once the server is running, open http://localhost:8000 in a browser to access the interface.
API Endpoints:
GET / :Provides the chat interface.
GET /health : Returns the service health status.
POST /ask : Accepts a user question and returns the generated answer.
GET /docs : Displays the automatically generated Swagger API documentation.

Example Request
curl -X POST http://localhost:8000/ask
-H "Content-Type: application/json"
-d '{"question": "What is the population of Germany?"}'
Example Response
{
"answer": "The population of Germany is 83,240,525.",
"countries": ["Germany"],
"fields": ["population"],
"error": null
}
Supported Fields
The system currently supports queries for the following country attributes:
-population
-capital
-currency
-area
-languages
-region
-flag
-timezones
-borders
-continent

LLM Provider Configuration
The language model provider is defined using the LLM_PROVIDER variable in the .env file.
Supported options include the following:
-OpenAI
Configuration value: openai
Default model: gpt-4o-mini
-Anthropic
Configuration value: anthropic
Default model: claude-sonnet-4-6
-OpenRouter
Configuration value: openrouter
Default model: nvidia/nemotron-3-nano-30b-a3b:free

Project Structure
The repository is organized into the following main directories.
The app directory contains the core application logic.
main.py defines the FastAPI application and routes.
config.py manages configuration related to LLM providers.
models.py contains Pydantic models for API requests and responses.
Inside the agent subdirectory:
graph.py defines the LangGraph workflow.
nodes.py implements the logic for each node in the agent.
state.py defines the schema for agent state.
tools.py contains the client used to access the REST Countries API.
The static directory includes the web interface.
index.html provides a simple chat interface for interacting with the agent.
Additional project files include requirements.txt for dependencies, .env.example for configuration guidance, and README.md for documentation.

Design Decisions:
LangGraph was chosen instead of a single prompt approach so that different responsibilities can be separated into distinct components. This improves maintainability and makes debugging easier.
Structured extraction of requested fields ensures that only relevant information is passed to the answer generation stage. This reduces token usage and helps maintain accuracy.
The system is designed to handle partial failures gracefully. Even if some country data cannot be retrieved, the system still generates a response using the available information and clearly communicates any missing results.
The configuration layer allows the language model provider to be changed without modifying application code.
Country matching uses both exact matching and partial matching to improve flexibility when users provide slightly different country names.


Known Limitations:
The agent currently processes each question independently and does not maintain conversation history.
The REST Countries API does not include certain data points such as GDP, so queries requiring those fields cannot be answered.
The system currently relies on a single external data source. Additional APIs could be integrated in the future to improve coverage.
There is no caching layer, which means repeated questions will trigger new API calls every time.
Rate limits depend on both the REST Countries API and the configured LLM provider.
