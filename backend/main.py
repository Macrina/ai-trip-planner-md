from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register  # pyright: ignore[reportMissingImports]
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import httpx


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None
    # Optional fields for enhanced session tracking and observability
    user_input: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Feature flag for optional RAG demo (opt-in for learning)
ENABLE_RAG = os.getenv("ENABLE_RAG", "0").lower() not in {"0", "false", "no"}


# RAG helper: Load curated local guides as LangChain documents
def _load_local_documents(path: Path) -> List[Document]:
    """Load local guides JSON and convert to LangChain Documents."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for row in raw:
        description = row.get("description")
        city = row.get("city")
        if not description or not city:
            continue
        interests = row.get("interests", []) or []
        metadata = {
            "city": city,
            "interests": interests,
            "source": row.get("source"),
        }
        # Prefix city + interests in content so embeddings capture location context
        interest_text = ", ".join(interests) if interests else "general travel"
        content = f"City: {city}\nInterests: {interest_text}\nGuide: {description}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class LocalGuideRetriever:
    """Retrieves curated local experiences using vector similarity search.
    
    This class demonstrates production RAG patterns for students:
    - Vector embeddings for semantic search
    - Fallback to keyword matching when embeddings unavailable
    - Graceful degradation with feature flags
    """
    
    def __init__(self, data_path: Path):
        """Initialize retriever with local guides data.
        
        Args:
            data_path: Path to local_guides.json file
        """
        self._docs = _load_local_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None
        
        # Only create embeddings when RAG is enabled and we have an API key
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                # Gracefully degrade to keyword search if embeddings fail
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        """Check if any documents were loaded."""
        return not self._docs

    def retrieve(self, destination: str, interests: Optional[str], *, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant local guides for a destination.
        
        Args:
            destination: City or destination name
            interests: Comma-separated interests (e.g., "food, art")
            k: Number of results to return
            
        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys
        """
        if not ENABLE_RAG or self.is_empty:
            return []

        # Use vector search if available, otherwise fall back to keywords
        if not self._vectorstore:
            return self._keyword_fallback(destination, interests, k=k)

        query = destination
        if interests:
            query = f"{destination} with interests {interests}"
        
        try:
            # LangChain retriever ensures embeddings + searches are traced
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(destination, interests, k=k)

        # Format results with metadata and scores
        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_val,
            })

        if not results:
            return self._keyword_fallback(destination, interests, k=k)
        return results

    def _keyword_fallback(self, destination: str, interests: Optional[str], *, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval when embeddings unavailable.
        
        This demonstrates graceful degradation for students learning about
        fallback strategies in production systems.
        """
        dest_lower = destination.lower()
        interest_terms = [part.strip().lower() for part in (interests or "").split(",") if part.strip()]

        def _score(doc: Document) -> int:
            score = 0
            city_match = doc.metadata.get("city", "").lower()
            # Match city name
            if dest_lower and dest_lower.split(",")[0] in city_match:
                score += 2
            # Match interests
            for term in interest_terms:
                if term and term in " ".join(doc.metadata.get("interests") or []).lower():
                    score += 1
                if term and term in doc.page_content.lower():
                    score += 1
            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]
        
        results = []
        for score, doc in top_docs:
            if score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
        return results


# Initialize retriever at module level (loads data once at startup)
_DATA_DIR = Path(__file__).parent / "data"
GUIDE_RETRIEVER = LocalGuideRetriever(_DATA_DIR / "local_guides.json")


# Search API configuration and helpers
SEARCH_TIMEOUT = 10.0  # seconds


def _compact(text: str, limit: int = 200) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured, return None otherwise.
    
    This demonstrates graceful degradation: tools work with or without API keys.
    Students can enable real search by adding TAVILY_API_KEY or SERPAPI_API_KEY.
    """
    query = query.strip()
    if not query:
        return None

    # Try Tavily first (recommended for AI apps)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully, try next option

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully

    return None  # No search APIs configured


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use the LLM to generate a response when search APIs aren't available.
    
    This ensures tools always return useful information, even without API keys.
    """
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are a concise travel assistant."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Tools with real API calls + LLM fallback (graceful degradation pattern)
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    query = f"{destination} travel essentials weather best time top attractions etiquette language currency safety"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} essentials", summary)
    
    # LLM fallback when no search API is configured
    instruction = f"Summarize the climate, best visit time, standout sights, customs, language, currency, and safety tips for {destination}."
    return _llm_fallback(instruction)


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    query = f"{destination} travel budget average daily costs {duration}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} budget {duration}", summary)
    
    instruction = f"Outline lodging, meals, transport, activities, and extra costs for a {duration} trip to {destination}."
    return _llm_fallback(instruction)


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    focus = interests or "local culture"
    query = f"{destination} authentic local experiences {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} {focus}", summary)
    
    instruction = f"Recommend authentic local experiences in {destination} that highlight {focus}."
    return _llm_fallback(instruction)


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    query = f"{destination} day {day} itinerary highlights"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"Day {day} in {destination}", summary)
    
    instruction = f"Outline key activities for day {day} in {destination}, covering morning, afternoon, and evening."
    return _llm_fallback(instruction)


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    query = f"{destination} weather forecast travel season temperatures rainfall"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} weather", summary)
    
    instruction = f"Give a weather brief for {destination} noting season, temperatures, rainfall, humidity, and packing guidance."
    return _llm_fallback(instruction)


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance for travel planning."""
    query = f"{destination} tourist visa requirements entry rules"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} visa", summary)
    
    instruction = f"Provide a visa guidance summary for visiting {destination}, including advice to confirm with the relevant embassy."
    return _llm_fallback(instruction)


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return pricing information for attractions."""
    items = attractions or ["popular attractions"]
    focus = ", ".join(items)
    query = f"{destination} attraction ticket prices {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} attraction prices", summary)
    
    instruction = f"Share typical ticket prices and savings tips for attractions such as {focus} in {destination}."
    return _llm_fallback(instruction)


@tool
def local_customs(destination: str) -> str:
    """Return cultural etiquette and customs information."""
    query = f"{destination} cultural etiquette travel customs"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} customs", summary)
    
    instruction = f"Summarize key etiquette and cultural customs travelers should know before visiting {destination}."
    return _llm_fallback(instruction)


@tool
def hidden_gems(destination: str) -> str:
    """Return lesser-known attractions and experiences."""
    query = f"{destination} hidden gems local secrets lesser known spots"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} hidden gems", summary)
    
    instruction = f"List lesser-known attractions or experiences that feel like hidden gems in {destination}."
    return _llm_fallback(instruction)


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return travel time estimates between locations."""
    query = f"travel time {from_location} to {to_location} by {mode}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{from_location}→{to_location} {mode}", summary)
    
    instruction = f"Estimate travel time from {from_location} to {to_location} by {mode} transport."
    return _llm_fallback(instruction)


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return packing recommendations for the trip."""
    acts = ", ".join(activities or ["sightseeing"])
    query = f"what to pack for {destination} {duration} {acts}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} packing", summary)
    
    instruction = f"Suggest packing essentials for a {duration} trip to {destination} focused on {acts}."
    return _llm_fallback(instruction)


def get_city_image(destination: str, day_theme: str = "") -> str:
    """Get a city image from Unsplash API using destination keyword."""
    unsplash_key = os.getenv("UNSPLASH_API_KEY")
    
    if not unsplash_key:
        # Fallback to picsum with destination and theme-based seed for unique images per day
        import hashlib
        combined = f"{destination}_{day_theme}"
        seed = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        return f"https://picsum.photos/800/400?random={seed}"
    
    try:
        # Search for city images
        search_query = f"{destination} city"
        if day_theme:
            search_query += f" {day_theme}"
        
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {unsplash_key}"}
        params = {
            "query": search_query,
            "per_page": 1,
            "orientation": "landscape"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                image_url = data["results"][0]["urls"]["regular"]
                return image_url
        
        # Fallback if API fails
        import hashlib
        seed = int(hashlib.md5(destination.encode()).hexdigest()[:8], 16)
        return f"https://picsum.photos/800/400?random={seed}"
        
    except Exception as e:
        print(f"Unsplash API error: {e}")
        # Fallback to picsum
        import hashlib
        seed = int(hashlib.md5(destination.encode()).hexdigest()[:8], 16)
        return f"https://picsum.photos/800/400?random={seed}"


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    prompt_t = (
        "You are a research assistant.\n"
        "Gather essential information about {destination}.\n"
        "Use tools to get weather, visa, and essential info, then summarize."
    )
    vars_ = {"destination": destination}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [essential_info, weather_brief, visa_brief]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    tool_results = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["research", "info_gathering"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "research")
                current_span.set_attribute("metadata.agent_node", "research_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    # Collect tool calls and execute them
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        tool_results = tr["messages"]
        
        # Add tool results to conversation and ask LLM to synthesize
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = "Based on the above information, provide a comprehensive summary for the traveler."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call with its own prompt template
        synthesis_vars = {"destination": destination, "context": "tool_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    budget = req.get("budget", "moderate")
    prompt_t = (
        "You are a budget analyst.\n"
        "Analyze costs for {destination} over {duration} with budget: {budget}.\n"
        "Use tools to get pricing information, then provide a detailed breakdown."
    )
    vars_ = {"destination": destination, "duration": duration, "budget": budget}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [budget_basics, attraction_prices]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["budget", "cost_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "budget")
                current_span.set_attribute("metadata.agent_node", "budget_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a detailed budget breakdown for {duration} in {destination} with a {budget} budget."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"duration": duration, "destination": destination, "budget": budget}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    interests = req.get("interests", "local culture")
    travel_style = req.get("travel_style", "standard")
    
    # RAG: Retrieve curated local guides if enabled
    context_lines = []
    if ENABLE_RAG:
        retrieved = GUIDE_RETRIEVER.retrieve(destination, interests, k=3)
        if retrieved:
            context_lines.append("=== Curated Local Guides (from database) ===")
            for idx, item in enumerate(retrieved, 1):
                content = item["content"]
                source = item["metadata"].get("source", "Unknown")
                context_lines.append(f"{idx}. {content}")
                context_lines.append(f"   Source: {source}")
            context_lines.append("=== End of Curated Guides ===\n")
    
    context_text = "\n".join(context_lines) if context_lines else ""
    
    prompt_t = (
        "You are a local guide.\n"
        "Find authentic experiences in {destination} for someone interested in: {interests}.\n"
        "Travel style: {travel_style}. Use tools to gather local insights.\n"
    )
    
    # Add retrieved context to prompt if available
    if context_text:
        prompt_t += "\nRelevant curated experiences from our database:\n{context}\n"
    
    vars_ = {
        "destination": destination,
        "interests": interests,
        "travel_style": travel_style,
        "context": context_text if context_text else "No curated context available.",
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [local_flavor, local_customs, hidden_gems]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["local", "local_experiences"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "local")
                current_span.set_attribute("metadata.agent_node", "local_agent")
                if ENABLE_RAG and context_text:
                    current_span.set_attribute("metadata.rag_enabled", "true")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a curated list of authentic experiences for someone interested in {interests} with a {travel_style} approach."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"interests": interests, "travel_style": travel_style, "destination": destination}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    travel_style = req.get("travel_style", "standard")
    interests = req.get("interests", "")
    user_input = (req.get("user_input") or "").strip()
    
    # Extract duration number for generating day images
    duration_num = 1
    try:
        duration_str = duration.lower().replace("days", "").replace("day", "").strip()
        duration_num = int(duration_str) if duration_str.isdigit() else 1
    except:
        duration_num = 1
    
    prompt_parts = [
        "Create a CONCISE, VISUAL {duration} itinerary for {destination} ({travel_style}).",
        "",
        "PERSONALIZATION: Focus on interests: {interests}",
        "",
        "CRITICAL FORMATTING RULES:",
        "1. BE CONCISE - Use short, punchy descriptions (max 10-15 words per activity)",
        "2. IMAGES - For EACH day, add a hero image at the top: ![Day X](IMAGE_URL_PLACEHOLDER_DAY_X) (will be replaced with real city images)",
        "   CRITICAL: You MUST create an image placeholder for EVERY single day. If duration is 5 days, create IMAGE_URL_PLACEHOLDER_DAY_1, IMAGE_URL_PLACEHOLDER_DAY_2, IMAGE_URL_PLACEHOLDER_DAY_3, IMAGE_URL_PLACEHOLDER_DAY_4, IMAGE_URL_PLACEHOLDER_DAY_5",
        "3. EMOJIS - Use emojis heavily for visual appeal (🏛️ 🍝 🎨 🌃 ☕ 🚇 💶 📸 🗺️ ⭐)",
        "4. ACTION ITEMS - After each activity, add action links with destination context:",
        f"   - 🗺️ [Directions](https://maps.google.com/?q=[LOCATION_NAME]+{destination})",
        f"   - 🎫 [Tickets](https://www.getyourguide.com/s/?q=[ATTRACTION_NAME]+{destination})",
        f"   - 📸 [Photos](https://unsplash.com/search/photos/[LOCATION_NAME]+{destination})",
        "   IMPORTANT: Replace [LOCATION_NAME] and [ATTRACTION_NAME] with ACTUAL place names (e.g., 'Eiffel Tower', 'Colosseum')",
        "",
        "STRUCTURE:",
        "## Welcome to {destination}",
        "Start with a brief, engaging introduction about {destination} - highlight what makes it special, its unique character, and why it's worth visiting. Include 2-3 key highlights that set the destination apart.",
        "",
        "IMPORTANT: Create EXACTLY the number of days specified in the duration. If duration is '7 days', create Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7.",
        "",
        "STRUCTURE FOR EACH DAY:",
        "### Day X: [Short Theme - e.g., 'Historic Heart', 'Cultural Gems', 'Local Flavors', 'Nature & Parks', 'Art & Museums', 'Food & Markets', 'Nightlife & Entertainment']",
        "",
        "![Day X](IMAGE_URL_PLACEHOLDER_DAY_X)",
        "",
        "**☀️ Morning**",
        "- 🏛️ **[SPECIFIC ATTRACTION NAME]** - Brief 1-line description",
        "  - 💶 Cost | ⏱️ 2 hours",
        "  - 🗺️ [Directions](https://maps.google.com/?q=[SPECIFIC_ATTRACTION_NAME]+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=[SPECIFIC_ATTRACTION_NAME]+{destination})",
        "",
        "**🌤️ Afternoon**",
        "- 🍝 **[SPECIFIC RESTAURANT NAME]** - Brief description",
        "  - 💶 Cost | ⏱️ Duration",
        "  - 🗺️ [Directions](https://maps.google.com/?q=[SPECIFIC_RESTAURANT_NAME]+{destination}) | 📸 [Photos](https://unsplash.com/search/photos/[SPECIFIC_RESTAURANT_NAME]+{destination})",
        "",
        "**🌆 Evening**",
        "- 🌆 **[SPECIFIC EVENING ACTIVITY NAME]** - Brief description",
        "  - 💶 Cost | ⏱️ Duration",
        "  - 🗺️ [Directions](https://maps.google.com/?q=[SPECIFIC_EVENING_ACTIVITY]+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=[SPECIFIC_EVENING_ACTIVITY]+{destination})",
        "",
        "---",
        "",
        "CRITICAL: Create ALL days from Day 1 to Day {duration_num} where {duration_num} is the duration number. Each day should have Morning, Afternoon, and Evening sections.",
        "VERIFICATION: You MUST create exactly {duration_num} days. Count them: Day 1, Day 2, Day 3... up to Day {duration_num}.",
        "",
        "IMPORTANT NAMING RULES:",
        "- NEVER use generic names like 'Restaurant', 'Activity', 'Attraction', 'Museum', 'Park'",
        "- ALWAYS use SPECIFIC, REAL place names (e.g., 'Louvre Museum', 'Eiffel Tower', 'Café de Flore', 'Sacre-Coeur Basilica')",
        "- Research actual attractions, restaurants, and activities in {destination}",
        "- Use famous landmarks, well-known restaurants, and popular activities",
        "- Replace [SPECIFIC_ATTRACTION_NAME] with real attraction names",
        "- Replace [SPECIFIC_RESTAURANT_NAME] with real restaurant names", 
        "- Replace [SPECIFIC_EVENING_ACTIVITY] with real activity names",
        "",
        "CRITICAL LINK FORMAT:",
        "- Use EXACTLY this format for photo links: https://unsplash.com/search/photos/[PLACE_NAME]+{destination}",
        "- NEVER use the old format: https://unsplash.com/s/photos/",
        "- Example: https://unsplash.com/search/photos/Colosseum+Rome",
        "IMPORTANT: Each day MUST have its own image placed immediately after the day header. Follow this exact structure:",
        "### Day X: [Theme]",
        "",
        "![Day X](IMAGE_URL_PLACEHOLDER_DAY_X)",
        "",
        "**☀️ Morning**",
        "...",
        "**🌤️ Afternoon**",
        "...",
        "**🌆 Evening**",
        "...",
        "---",
        "",
        "---",
        "",
        "## 💰 Budget Snapshot",
        "- 🏨 Accommodation: €X/night",
        "- 🍽️ Food: €X/day",
        "- 🎫 Activities: €X/day",
        "- 🚇 Transport: €X/day",
        "**Total: €X/day**",
        "",
        "## 🎒 Quick Tips",
        "- 💡 Tip 1 (one line)",
        "- 💡 Tip 2 (one line)",
        "- 💡 Tip 3 (one line)",
        "",
        "## 📱 Useful Links",
        "- 🗺️ [Google Maps - {destination}](https://maps.google.com/?q={destination})",
        "- 🚇 [Public Transport](https://www.rome2rio.com/s/{destination})",
        "- 🎫 [Book Tours](https://www.getyourguide.com/s/?q={destination})",
        "",
        "---",
        "",
        "Agent inputs to use:",
        "Research: {research}",
        "Budget: {budget}",
        "Local: {local}",
    ]
    if user_input:
        prompt_parts.append("User input: {user_input}")
    
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "duration": duration,
        "destination": destination,
        "travel_style": travel_style,
        "duration_num": duration_num,
        "interests": interests or "general travel",
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
        "user_input": user_input,
    }
    
    # Add span attributes for better observability in Arize
    # NOTE: using_attributes must be OUTER context for proper propagation
    with using_attributes(tags=["itinerary", "final_agent"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.itinerary", "true")
                current_span.set_attribute("metadata.agent_type", "itinerary")
                current_span.set_attribute("metadata.agent_node", "itinerary_agent")
                if user_input:
                    current_span.set_attribute("metadata.user_input", user_input)
        
        # Prompt template wrapper for Arize Playground integration
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    
    # Process the content to replace image placeholders with real city images
    content = res.content
    
    # Validate that all required days are present
    missing_days = []
    for day in range(1, duration_num + 1):
        day_header = f"### Day {day}:"
        if day_header not in content:
            missing_days.append(day)
    
    if missing_days:
        print(f"⚠️ Missing days detected: {missing_days}")
        # Add missing days with basic structure
        for day in missing_days:
            day_theme = f"Day {day} Activities"
            missing_day_content = f"""

### Day {day}: {day_theme}

![Day {day}](IMAGE_URL_PLACEHOLDER_DAY_{day})

**Morning** ☀️
- 🏛️ **[Attraction]** - Explore local highlights
  - 💶 Cost varies | ⏱️ 2-3 hours
  - 🗺️ [Directions](https://maps.google.com/?q=Attraction+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=Attraction+{destination})

**Afternoon** 🌤️
- 🍝 **[Restaurant/Activity]** - Local experience
  - 💶 Cost varies | ⏱️ 1-2 hours
  - 🗺️ [Directions](https://maps.google.com/?q=Restaurant+{destination}) | 📸 [Photos](https://unsplash.com/search/photos/restaurant+{destination})

**Evening** 🌆
- 🌆 **[Evening Activity]** - Relax and enjoy
  - 💶 Cost varies | ⏱️ 2-3 hours
  - 🗺️ [Directions](https://maps.google.com/?q=Activity+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=Activity+{destination})

---
"""
            # Insert missing day content before the budget section
            budget_pos = content.find("## 💰 Budget Snapshot")
            if budget_pos != -1:
                content = content[:budget_pos] + missing_day_content + content[budget_pos:]
            else:
                content += missing_day_content
            print(f"✅ Added missing Day {day} content")

    # Replace image placeholders with actual city images
    for day in range(1, duration_num + 1):
        # Get different images for each day with day-specific themes
        day_themes = ["morning", "afternoon", "evening", "landmarks", "culture", "food", "nature", "architecture", "street", "market", "museum", "park", "beach", "mountain", "skyline", "nightlife"]
        theme = day_themes[(day - 1) % len(day_themes)]
        
        # Add day number to ensure uniqueness even with same theme
        unique_theme = f"{theme}_day{day}"
        image_url = get_city_image(destination, unique_theme)
        placeholder = f"IMAGE_URL_PLACEHOLDER_DAY_{day}"
        
        # Check if placeholder exists in content
        if placeholder in content:
            content = content.replace(placeholder, image_url)
            print(f"✅ Replaced {placeholder} with {destination} {theme} image")
        else:
            print(f"⚠️ Placeholder {placeholder} not found in content")
            # Add the image manually if placeholder is missing
            day_header = f"### Day {day}:"
            if day_header in content:
                # Insert image after the day header
                image_markdown = f"\n\n![Day {day}]({image_url})\n"
                content = content.replace(day_header, day_header + image_markdown)
                print(f"✅ Added {destination} {theme} image for Day {day}")
    
    # Fix Unsplash URLs - replace old format with new format
    content = content.replace("https://unsplash.com/s/photos/", "https://unsplash.com/search/photos/")
    
    # Final validation: count actual days in content
    actual_days = content.count("### Day ")
    print(f"📊 Generated {actual_days} days out of {duration_num} requested")
    
    if actual_days < duration_num:
        print(f"⚠️ WARNING: Only {actual_days} days generated, expected {duration_num}")
    
    return {"messages": [SystemMessage(content=content)], "final": content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_node", research_agent)
    g.add_node("budget_node", budget_agent)
    g.add_node("local_node", local_agent)
    g.add_node("itinerary_node", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research_node")
    g.add_edge(START, "budget_node")
    g.add_edge(START, "local_node")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research_node", "itinerary_node")
    g.add_edge("budget_node", "itinerary_node")
    g.add_edge("local_node", "itinerary_node")
    
    g.add_edge("itinerary_node", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/cities.json")
def serve_cities():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "cities.json")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "cities.json not found"}


@app.get("/api/cities")
async def search_cities(q: str = ""):
    """
    Free cities autocomplete API using REST Countries and Geonames
    """
    if len(q) < 2:
        return {"cities": []}
    
    try:
        # Use a free cities API - Geonames (free tier available)
        # For now, let's use a simple fallback with major cities
        major_cities = [
            {"name": "Paris", "country": "France", "region": "Europe"},
            {"name": "London", "country": "United Kingdom", "region": "Europe"},
            {"name": "Rome", "country": "Italy", "region": "Europe"},
            {"name": "Barcelona", "country": "Spain", "region": "Europe"},
            {"name": "Amsterdam", "country": "Netherlands", "region": "Europe"},
            {"name": "Berlin", "country": "Germany", "region": "Europe"},
            {"name": "Prague", "country": "Czech Republic", "region": "Europe"},
            {"name": "Vienna", "country": "Austria", "region": "Europe"},
            {"name": "Tokyo", "country": "Japan", "region": "Asia"},
            {"name": "Seoul", "country": "South Korea", "region": "Asia"},
            {"name": "Bangkok", "country": "Thailand", "region": "Asia"},
            {"name": "Singapore", "country": "Singapore", "region": "Asia"},
            {"name": "Hong Kong", "country": "Hong Kong", "region": "Asia"},
            {"name": "Dubai", "country": "UAE", "region": "Middle East"},
            {"name": "New York", "country": "United States", "region": "North America"},
            {"name": "Los Angeles", "country": "United States", "region": "North America"},
            {"name": "San Francisco", "country": "United States", "region": "North America"},
            {"name": "Toronto", "country": "Canada", "region": "North America"},
            {"name": "Sydney", "country": "Australia", "region": "Oceania"},
            {"name": "Melbourne", "country": "Australia", "region": "Oceania"},
            {"name": "Auckland", "country": "New Zealand", "region": "Oceania"},
        ]
        
        # Filter cities based on query
        filtered_cities = [
            city for city in major_cities
            if q.lower() in city["name"].lower() or q.lower() in city["country"].lower()
        ]
        
        return {"cities": filtered_cities[:8]}  # Limit to 8 results
        
    except Exception as e:
        return {"cities": [], "error": str(e)}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-trip-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    graph = build_graph()
    
    # Only include necessary fields in initial state
    # Agent outputs (research, budget, local, final) will be added during execution
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }
    
    # Add session and user tracking attributes to the trace
    session_id = req.session_id
    user_id = req.user_id
    turn_idx = req.turn_index
    
    # Build attributes for session and user tracking
    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id
    
    # Add turn_index as a custom span attribute if provided
    if turn_idx is not None and _TRACING:
        with using_attributes(**attrs_kwargs):
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("turn_index", turn_idx)
            out = graph.invoke(state)
    else:
        with using_attributes(**attrs_kwargs):
            out = graph.invoke(state)
    
    # Fix Unsplash URLs in final result
    final_result = out.get("final", "")
    if final_result:
        # Count old format URLs
        old_count = final_result.count("https://unsplash.com/s/photos/")
        
        # Replace old format with new format
        final_result = final_result.replace("https://unsplash.com/s/photos/", "https://unsplash.com/search/photos/")
        
        # Also handle any URLs without https:// prefix
        final_result = final_result.replace("unsplash.com/s/photos/", "unsplash.com/search/photos/")
        
        new_count = final_result.count("https://unsplash.com/search/photos/")
        if old_count > 0:
            print(f"🔧 Fixed {old_count} Unsplash URLs from old format to new format")
    
    return TripResponse(result=final_result, tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
