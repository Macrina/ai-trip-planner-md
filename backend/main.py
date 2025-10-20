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

# Arize instrumentation (optional)
try:
    from arize.otel import register
    
    arize_space_id = os.getenv("ARIZE_SPACE_ID")
    arize_api_key = os.getenv("ARIZE_API_KEY")
    arize_project_name = os.getenv("ARIZE_PROJECT_NAME", "ai-trip-planner")
    
    if arize_space_id and arize_api_key:
        tracer_provider = register(
            space_id=arize_space_id,
            api_key=arize_api_key,
            project_name=arize_project_name,
        )
        
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        print("✅ Arize instrumentation enabled")
    else:
        print("⚠️ Arize credentials not found - tracing disabled")
except ImportError:
    print("⚠️ Arize not installed - tracing disabled")

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

    if os.getenv("TEST_MODE") == "1":
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
        if ENABLE_RAG and self._docs and os.getenv("TEST_MODE") != "1":
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
    
    if not unsplash_key or unsplash_key == "your_unsplash_api_key_here":
        print(f"⚠️ Unsplash API key not found, using Picsum fallback for {destination}")
        # Fallback to picsum with destination and theme-based seed for unique images per day
        import hashlib
        combined = f"{destination}_{day_theme}"
        seed = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        return f"https://picsum.photos/800/400?random={seed}"
    
    try:
        # Create more specific search query based on day theme
        search_query = destination
        if day_theme:
            # Map day themes to better search terms
            theme_mapping = {
                "morning": "morning sunrise",
                "afternoon": "afternoon landmarks",
                "evening": "evening sunset",
                "landmarks": "landmarks architecture",
                "culture": "culture museums",
                "food": "food restaurants",
                "nature": "nature parks",
                "historic": "historic old town",
                "art": "art galleries",
                "markets": "markets shopping"
            }
            
            # Extract theme from day_theme (e.g., "morning_day1" -> "morning")
            theme = day_theme.split('_')[0] if '_' in day_theme else day_theme
            if theme in theme_mapping:
                search_query += f" {theme_mapping[theme]}"
            else:
                search_query += f" {theme}"
        
        print(f"🔍 Searching Unsplash for: {search_query}")
        
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {unsplash_key}"}
        params = {
            "query": search_query,
            "per_page": 3,  # Get more options
            "orientation": "landscape"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                # Use the first result
                image_url = data["results"][0]["urls"]["regular"]
                print(f"✅ Found Unsplash image for {destination}: {search_query}")
                return image_url
            else:
                print(f"⚠️ No Unsplash results for: {search_query}")
        else:
            print(f"⚠️ Unsplash API error: {response.status_code} - {response.text}")
        
        # Fallback if API fails
        import hashlib
        seed = int(hashlib.md5(destination.encode()).hexdigest()[:8], 16)
        print(f"🔄 Falling back to Picsum for {destination}")
        return f"https://picsum.photos/800/400?random={seed}"
        
    except Exception as e:
        print(f"❌ Unsplash API error: {e}")
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
    
    # Research agent execution
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
        
        # Synthesis step
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
    
    # Budget agent execution
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
        
        # Synthesis step
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
    
    # Local agent execution
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
        
        # Synthesis step
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
        # LIMIT TO MAXIMUM 5 DAYS
        if duration_num > 5:
            duration_num = 5
            duration = "5 days"
            print(f"⚠️ Duration limited to maximum 5 days")
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
        f"   - 🗺️ [Directions](https://maps.google.com/?q=ACTUAL_PLACE_NAME+{destination})",
        f"   - 🎫 [Tickets](https://www.getyourguide.com/s/?q=ACTUAL_PLACE_NAME+{destination})",
        "   - 📸 [Photos](https://unsplash.com/s/photos/ACTUAL_PLACE_NAME_LOWERCASE_WITH_HYPHENS)",
        "   CRITICAL: Replace ACTUAL_PLACE_NAME with the REAL, SPECIFIC place name from the activity above",
        "   CRITICAL FOR PHOTOS: Convert place name to lowercase and replace spaces with hyphens (e.g., 'Louvre Museum' → 'louvre-museum')",
        "   EXAMPLES: 'Eiffel Tower', 'Colosseum', 'Sagrada Familia', 'Louvre Museum', 'Times Square'",
        "",
        "STRUCTURE:",
        "## Welcome to {destination}",
        "Start with a brief, engaging introduction about {destination} - highlight what makes it special, its unique character, and why it's worth visiting. Include 2-3 key highlights that set the destination apart.",
        "",
        "",
        "🚨 CRITICAL DAY COUNT REQUIREMENT 🚨:",
        f"You MUST create EXACTLY {duration_num} days. Count them: {', '.join([f'Day {d}' for d in range(1, duration_num + 1)])}",
        f"If you generate less than {duration_num} days, the itinerary is INCOMPLETE and REJECTED.",
        f"VERIFICATION: After writing, count your days. You should have {duration_num} sections starting with '### Day X:'",
        "",
        "STRUCTURE FOR EACH DAY:",
        "### Day X: [Short Theme - e.g., 'Historic Heart', 'Cultural Gems', 'Local Flavors', 'Nature & Parks', 'Art & Museums', 'Food & Markets', 'Nightlife & Entertainment']",
        "",
        "![Day X](IMAGE_URL_PLACEHOLDER_DAY_X)",
        "",
        "**☀️ Morning**",
        "- 🏛️ **[REAL ATTRACTION NAME]** - Detailed description MINIMUM 150 characters (up to 200 characters) explaining what makes this place special, unique features, and what visitors can expect to experience",
        "  - 💵 $15-25 | ⏱️ 2 hours",
        "  - 🗺️ [Directions](https://maps.google.com/?q=REAL_ATTRACTION_NAME+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=REAL_ATTRACTION_NAME+{destination})",
        "",
        "**🌤️ Afternoon**",
        "- 🍝 **[REAL RESTAURANT NAME]** - Detailed description MINIMUM 150 characters (up to 200 characters) highlighting cuisine type, signature dishes, ambiance, and why it's worth visiting for food lovers",
        "  - 💵 $20-40 | ⏱️ 1.5 hours",
        f"  - 🗺️ [Directions](https://maps.google.com/?q=REAL_RESTAURANT_NAME+{destination}) | 📸 [Photos](https://unsplash.com/s/photos/real-restaurant-name-paris)",
        "",
        "**🌆 Evening**",
        "- 🌆 **[REAL EVENING ACTIVITY NAME]** - Detailed description MINIMUM 150 characters (up to 200 characters) describing the atmosphere, what activities are available, and what makes this evening spot memorable",
        "  - 💵 $25-50 | ⏱️ 2-3 hours",
        "  - 🗺️ [Directions](https://maps.google.com/?q=REAL_EVENING_ACTIVITY+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=REAL_EVENING_ACTIVITY+{destination})",
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
        "- Replace REAL_ATTRACTION_NAME with the actual attraction name from the activity above",
        "- Replace REAL_RESTAURANT_NAME with the actual restaurant name from the activity above", 
        "- Replace REAL_EVENING_ACTIVITY with the actual activity name from the activity above",
        "- CRITICAL: The place name in the link MUST match the place name in the activity title",
        "",
        "📝 CRITICAL DESCRIPTION LENGTH REQUIREMENT:",
        "- ⚠️ ABSOLUTE MINIMUM: Each activity description MUST be at least 150 characters (count the characters!)",
        "- IDEAL RANGE: 160-200 characters for optimal detail",
        "- ❌ ANY description under 150 characters is REJECTED and UNACCEPTABLE",
        "- ❌ REJECTED: Short descriptions like 'Visit X' or 'See Y' or 'Explore Z' (TOO SHORT!)",
        "- ✅ REQUIRED: Detailed, engaging descriptions that explain WHY the place is special",
        "- Descriptions should be detailed and informative, not brief one-liners",
        "- Include specific details about what makes the place special, unique features, atmosphere, or what visitors can expect",
        "- Example PERFECT description (180 chars): 'Marvel at Gaudí's unfinished masterpiece featuring intricate facades, colorful stained glass, and towering spires. A UNESCO World Heritage site that blends Gothic and Art Nouveau styles.'",
        "- Example BAD description (too short): 'Visit Gaudí's famous church' (only 28 chars - TOO SHORT! REJECTED!)",
        "- ✅ Before you write each description, COUNT the characters to ensure it meets the 150+ character minimum",
        "- ✅ Write 2-3 full sentences for each activity description",
        "",
        "💵 CRITICAL COST ESTIMATION REQUIREMENT (REALISTIC USD PRICES):",
        "- ALL costs MUST be in USD with realistic price ranges based on actual market prices",
        "- Use ACTUAL NUMBERS format: 💵 $15-25, 💵 $30-50, 💵 $8-12",
        "- ❌ NEVER use placeholders like: $$, $XX-YY, $X-Y, Varies, TBD, Cost",
        "- ❌ REJECTED: Any response with generic cost placeholders will be REJECTED",
        "- ✅ REQUIRED: Every activity MUST have a specific USD price range with real numbers",
        "- Research typical prices for the specific activity type and destination",
        "",
        "PRICE GUIDELINES BY ACTIVITY TYPE:",
        "Museums/Attractions:",
        "  - Free attractions (parks, churches, viewpoints): 💵 Free",
        "  - Small museums/local sites: 💵 $5-15",
        "  - Major museums/attractions: 💵 $15-30",
        "  - Premium experiences (skip-the-line, guided tours): 💵 $30-60",
        "",
        "Restaurants/Meals:",
        "  - Street food/casual cafes: 💵 $8-15",
        "  - Mid-range restaurants: 💵 $20-40",
        "  - Upscale dining: 💵 $50-100",
        "  - Fine dining/Michelin stars: 💵 $100-200",
        "",
        "Evening Activities:",
        "  - Free activities (sunset walks, night markets): 💵 Free",
        "  - Shows/performances: 💵 $25-60",
        "  - Bars/clubs entry: 💵 $10-30",
        "  - Special experiences (rooftop bars, boat tours): 💵 $30-80",
        "",
        "⚠️ ADJUST for destination cost of living:",
        "  - Budget destinations (Southeast Asia, Eastern Europe): -30-50% from base prices",
        "  - Mid-range destinations (Southern Europe, Latin America): Use base prices",
        "  - Expensive destinations (Western Europe, Japan, Scandinavia): +30-50% from base prices",
        "  - Premium destinations (Switzerland, Iceland, Monaco): +50-100% from base prices",
        "",
        "✅ Examples of GOOD cost estimates:",
        "  - 'Louvre Museum, Paris' → 💵 $18-22 (major museum in expensive city)",
        "  - 'Pho street vendor, Hanoi' → 💵 $2-4 (street food in budget destination)",
        "  - 'Tapas bar, Barcelona' → 💵 $15-30 (casual dining in mid-range city)",
        "  - 'Tokyo Skytree' → 💵 $25-35 (premium attraction in expensive city)",
        "",
        "CRITICAL LINK FORMAT:",
        "- Use EXACTLY this format for photo links: https://unsplash.com/s/photos/[place-name-lowercase-with-hyphens]",
        "- Convert place names to lowercase and replace spaces with hyphens",
        "- Example: 'Colosseum Rome' → https://unsplash.com/s/photos/colosseum-rome",
        "- Example: 'Louvre Museum Paris' → https://unsplash.com/s/photos/louvre-museum-paris",
        "",
        "CONCRETE EXAMPLE:",
        "If the activity is: - 🏛️ **Sagrada Familia** - Visit Gaudi's masterpiece",
        "Then the links should be:",
        f"  - 🗺️ [Directions](https://maps.google.com/?q=Sagrada+Familia+{destination})",
        f"  - 🎫 [Tickets](https://www.getyourguide.com/s/?q=Sagrada+Familia+{destination})",
        "  - 📸 [Photos](https://unsplash.com/s/photos/sagrada-familia-barcelona)",
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
        "",
        "🚨 FINAL QUALITY CHECK BEFORE SUBMITTING:",
        "1. ✅ Every activity has a description with AT LEAST 150 characters",
        "2. ✅ Every activity has a SPECIFIC USD price (e.g., 💵 $15-25) - NO PLACEHOLDERS",
        "3. ✅ All place names are SPECIFIC and REAL (no generic names)",
        "4. ✅ All Unsplash photo links use correct format with lowercase and hyphens",
        "5. ✅ Generated EXACTLY {duration_num} days with complete Morning, Afternoon, and Evening sections",
        "",
        "⛔ DO NOT SUBMIT if any activity has:",
        "- Generic costs like $$, $XX-YY, Varies, TBD, or Cost",
        "- Short descriptions under 150 characters",
        "- Generic names like 'Restaurant' or 'Museum'",
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
    
    # Itinerary agent execution
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
        print(f"⚠️ Missing days detected: {missing_days}. Attempting retry...")
        
        # RETRY: Generate missing days with focused prompt
        retry_prompt = f"""You are creating an itinerary for {destination} for {duration_num} days.
You already generated some days, but Days {', '.join(map(str, missing_days))} are MISSING.

CRITICAL: Generate ONLY these specific days: {', '.join([f'Day {d}' for d in missing_days])}

EXACT FORMAT for EACH missing day:

### Day X: [Creative Theme]

![Day X](IMAGE_URL_PLACEHOLDER_DAY_X)

**☀️ Morning**
- 🏛️ **[Specific Attraction]** - Detailed description MINIMUM 150 characters (up to 200 chars) explaining what makes this special
  - 💵 $15-30 | ⏱️ 2-3 hours  
  - 🗺️ [Directions](https://maps.google.com/?q=Attraction+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=Attraction+{destination})

**🌤️ Afternoon**
- 🍝 **[Specific Restaurant]** - Detailed description MINIMUM 150 characters (up to 200 chars) explaining cuisine, dishes, and atmosphere
  - 💵 $20-45 | ⏱️ 1-2 hours
  - 🗺️ [Directions](https://maps.google.com/?q=Restaurant+{destination}) | 📸 [Photos](https://unsplash.com/s/photos/restaurant-{destination.lower().replace(' ', '-')})

**🌆 Evening**
- 🎭 **[Specific Activity]** - Detailed description MINIMUM 150 characters (up to 200 chars) describing the experience and what to expect
  - 💵 $25-50 | ⏱️ 2-3 hours
  - 🗺️ [Directions](https://maps.google.com/?q=Activity+{destination}) | 🎫 [Tickets](https://www.getyourguide.com/s/?q=Activity+{destination})

---

⚠️ CRITICAL: ALL activity descriptions MUST be at least 150 characters. Count them!
⚠️ CRITICAL: ALL costs MUST be realistic USD prices with ACTUAL NUMBERS (💵 $15-25, NOT $XX-YY or $$)!
⚠️ CRITICAL: Research actual prices for {destination} - NO PLACEHOLDERS ALLOWED!

Context: {vars_['research'][:200]}
Local tips: {vars_['local'][:200]}

MUST GENERATE: {len(missing_days)} day(s) - {', '.join([f'Day {d}' for d in missing_days])}"""

        try:
            print(f"🔄 Retrying to generate days: {missing_days}")
            retry_res = llm.invoke([SystemMessage(content=retry_prompt)])
            retry_content = retry_res.content
            
            # Verify retry success
            still_missing = [d for d in missing_days if f"### Day {d}:" not in retry_content]
            
            if not still_missing:
                # Success! Insert retry content
                budget_pos = content.find("## 💰 Budget Snapshot")
                if budget_pos != -1:
                    content = content[:budget_pos] + "\n\n" + retry_content + "\n\n" + content[budget_pos:]
                else:
                    content += "\n\n" + retry_content
                print(f"✅ Retry successful! Generated missing days: {missing_days}")
            else:
                print(f"⚠️ Retry partial success. Still missing: {still_missing}")
                raise Exception(f"Still missing days: {still_missing}")
                
        except Exception as e:
            # Fallback: Use basic template
            print(f"⚠️ Retry failed: {e}. Using fallback template for: {missing_days}")
        for day in missing_days:
            missing_day_content = f"""

### Day {day}: Explore {destination}

![Day {day}](IMAGE_URL_PLACEHOLDER_DAY_{day})

**☀️ Morning**
- 🏛️ **Local Highlights** - Discover the fascinating treasures and iconic landmarks that make {destination} unique. Explore historical sites, architectural wonders, and cultural hotspots that showcase the city's rich heritage and vibrant character. Perfect for photography enthusiasts and history lovers alike.
  - 💵 $15-30 | ⏱️ 2-3 hours
  - 🗺️ [Directions](https://maps.google.com/?q={destination}+attractions) | 🎫 [Tickets](https://www.getyourguide.com/s/?q={destination})

**🌤️ Afternoon**
- 🍝 **Traditional Cuisine** - Savor authentic local flavors and traditional dishes that define {destination}'s culinary scene. Experience the unique tastes, fresh ingredients, and time-honored cooking techniques that make this destination a food lover's paradise. Enjoy regional specialties in a welcoming atmosphere.
  - 💵 $20-45 | ⏱️ 1-2 hours
  - 🗺️ [Directions](https://maps.google.com/?q={destination}+restaurants) | 📸 [Photos](https://unsplash.com/s/photos/{destination.lower().replace(' ', '-')}-food)

**🌆 Evening**
- 🎭 **Cultural Experience** - Experience {destination} by night with its vibrant entertainment scene, illuminated landmarks, and lively atmosphere. Discover local nightlife, evening performances, or simply stroll through beautifully lit streets while soaking in the city's unique after-dark charm and energy.
  - 💵 $25-50 | ⏱️ 2-3 hours
  - 🗺️ [Directions](https://maps.google.com/?q={destination}+nightlife) | 🎫 [Tickets](https://www.getyourguide.com/s/?q={destination})

---
"""
            budget_pos = content.find("## 💰 Budget Snapshot")
            if budget_pos != -1:
                content = content[:budget_pos] + missing_day_content + content[budget_pos:]
            else:
                content += missing_day_content
                print(f"✅ Added fallback content for Day {day}")

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
    
    # Fix Unsplash URLs - ensure correct format (keep /s/photos/ format)
    # Note: /s/photos/ is the correct working format, not /search/photos/
    
    # Convert Unsplash URLs from + signs to hyphens
        import re
        
    def fix_unsplash_url(match):
        """Convert Unsplash URLs to use hyphens instead of plus signs"""
        url = match.group(0)
        # Handle both /s/photos/ and /search/photos/ formats
        if '/s/photos/' in url:
            base = 'https://unsplash.com/s/photos/'
            path = url.split('/s/photos/')[-1].rstrip(')')
        elif '/search/photos/' in url:
            base = 'https://unsplash.com/s/photos/'
            path = url.split('/search/photos/')[-1].rstrip(')')
        else:
            return url
        
        # Convert to lowercase and replace + with -
        path = path.lower().replace('+', '-')
        return f'{base}{path}'
    
    # Fix all Unsplash URLs in content (both formats)
    content = re.sub(r'https://unsplash\.com/(?:s|search)/photos/[^\s\)]+', fix_unsplash_url, content)
    
    # Extra pass: catch any remaining /search/photos/ URLs
    content = content.replace('unsplash.com/search/photos/', 'unsplash.com/s/photos/')
    
    # Extra pass: fix any + signs in Unsplash URLs that might have been missed
    def replace_plus_in_unsplash(match):
        url = match.group(0)
        if '+' in url:
            parts = url.split('/s/photos/')
            if len(parts) == 2:
                fixed_path = parts[1].lower().replace('+', '-')
                return f'https://unsplash.com/s/photos/{fixed_path}'
        return url
    
    content = re.sub(r'https://unsplash\.com/s/photos/[^\s\)]+', replace_plus_in_unsplash, content)
    
    # Debug: Count fixed URLs
    unsplash_urls = re.findall(r'https://unsplash\.com/s/photos/[^\s\)]+', content)
    if unsplash_urls:
        print(f"🔍 Unsplash URLs after fix: {unsplash_urls[:3]}...")  # Show first 3
    
    # Post-process to fix generic placeholders in action links
    try:
        # Fix generic placeholders in Google Maps links
        maps_pattern = r'https://maps\.google\.com/\?q=([^+]+)\+([^)]+)'
        def fix_maps_link(match):
            place_name = match.group(1)
            destination = match.group(2)
            # If place name is generic, use a more specific search
            if place_name.lower() in ['attraction', 'restaurant', 'activity', 'restaurant/activity', 'evening activity']:
                return f"https://maps.google.com/?q={destination}+attractions"
            return match.group(0)
        content = re.sub(maps_pattern, fix_maps_link, content)
        
        # Fix generic placeholders in GetYourGuide links
        tickets_pattern = r'https://www\.getyourguide\.com/s/\?q=([^+]+)\+([^)]+)'
        def fix_tickets_link(match):
            place_name = match.group(1)
            destination = match.group(2)
            # If place name is generic, use a more specific search
            if place_name.lower() in ['attraction', 'restaurant', 'activity', 'restaurant/activity', 'evening activity']:
                return f"https://www.getyourguide.com/s/?q={destination}+tours"
            return match.group(0)
        content = re.sub(tickets_pattern, fix_tickets_link, content)
        
        # Fix generic placeholders in Unsplash photo links
        photos_pattern = r'https://unsplash\.com/search/photos/([^+]+)\+([^)]+)'
        def fix_photos_link(match):
            place_name = match.group(1)
            destination = match.group(2)
            # If place name is generic, use a more specific search
            if place_name.lower() in ['attraction', 'restaurant', 'activity', 'restaurant/activity', 'evening activity']:
                return f"https://unsplash.com/search/photos/{destination}+city"
            return match.group(0)
        content = re.sub(photos_pattern, fix_photos_link, content)
        
        print(f"🔧 Post-processed action links to fix generic placeholders")
    except Exception as e:
        print(f"⚠️ Error in post-processing: {e}")
        # Continue without post-processing
    
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


@app.get("/api/background-image")
def get_background_image():
    """Get a random travel background image from Unsplash"""
    try:
        unsplash_access_key = os.getenv("UNSPLASH_API_KEY")
        if unsplash_access_key:
            # Use official Unsplash API
            url = "https://api.unsplash.com/photos/random"
            params = {
                "query": "travel destination landscape",
                "orientation": "landscape",
                "client_id": unsplash_access_key
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                image_url = data.get("urls", {}).get("full", data.get("urls", {}).get("regular"))
                if image_url:
                    return {"url": image_url}
        
        # Fallback to a reliable placeholder service
        return {"url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=1920&h=1080&fit=crop"}
    except Exception as e:
        print(f"Error fetching background image: {e}")
        # Fallback image
        return {"url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=1920&h=1080&fit=crop"}


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
    
    # Execute the graph
    out = graph.invoke(state)
    
    # Fix Unsplash URLs in final result
    final_result = out.get("final", "")
    if final_result:
        import re
        
        def fix_unsplash_url_final(match):
            """Convert Unsplash URLs to use hyphens instead of plus signs"""
            url = match.group(0)
            # Handle both /s/photos/ and /search/photos/ formats
            if '/s/photos/' in url:
                base = 'https://unsplash.com/s/photos/'
                path = url.split('/s/photos/')[-1].rstrip(')')
            elif '/search/photos/' in url:
                base = 'https://unsplash.com/s/photos/'
                path = url.split('/search/photos/')[-1].rstrip(')')
            else:
                return url
            
            # Convert to lowercase and replace + with -
            path = path.lower().replace('+', '-')
            return f'{base}{path}'
        
        # Fix all Unsplash URLs (convert + to - and use /s/photos/ format)
        final_result = re.sub(r'https://unsplash\.com/(?:s|search)/photos/[^\s\)]+', fix_unsplash_url_final, final_result)
        
        # Extra pass: catch any remaining /search/photos/ URLs
        final_result = final_result.replace('unsplash.com/search/photos/', 'unsplash.com/s/photos/')
        
        # Extra pass: fix any + signs in Unsplash URLs
        def replace_plus_final(match):
            url = match.group(0)
            if '+' in url:
                parts = url.split('/s/photos/')
                if len(parts) == 2:
                    fixed_path = parts[1].lower().replace('+', '-')
                    return f'https://unsplash.com/s/photos/{fixed_path}'
            return url
        
        final_result = re.sub(r'https://unsplash\.com/s/photos/[^\s\)]+', replace_plus_final, final_result)
        
        unsplash_count = final_result.count("https://unsplash.com/s/photos/")
        # Show sample URLs for debugging
        sample_urls = re.findall(r'https://unsplash\.com/s/photos/[^\s\)]+', final_result)
        if unsplash_count > 0:
            print(f"✅ Fixed {unsplash_count} Unsplash URLs")
            if sample_urls:
                print(f"   Sample: {sample_urls[0]}")
    
    return TripResponse(result=final_result, tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    
    # Always use port 8000 (no auto port switching)
    PORT = 8000
    print(f"🚀 Starting backend on port {PORT}")
    print(f"📍 URL: http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
