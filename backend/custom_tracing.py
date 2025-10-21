"""
Custom tracing utilities for AI Trip Planner

Provides helpers for adding custom spans, attributes, and events
to monitor fallbacks, constraints, and quality metrics.
"""

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import time
import re


def get_current_span():
    """Get the current active span."""
    return trace.get_current_span()


def add_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add an event to the current span."""
    span = get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes or {})


def set_attribute(key: str, value: Any):
    """Set an attribute on the current span."""
    span = get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def set_status(status_code: StatusCode, description: Optional[str] = None):
    """Set the status of the current span."""
    span = get_current_span()
    if span and span.is_recording():
        span.set_status(Status(status_code, description))


# ============================================================================
# FALLBACK MONITORING
# ============================================================================

@contextmanager
def trace_rag_retrieval(destination: str, interests: Optional[str] = None, k: int = 3):
    """
    Trace RAG retrieval with fallback monitoring.
    
    Usage:
        with trace_rag_retrieval("Tokyo", "food, art", k=3) as tracer:
            try:
                results = vector_search()
                tracer.success("vector", len(results))
            except Exception:
                results = keyword_search()
                tracer.fallback("keyword", len(results))
    """
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("rag_retrieval") as span:
        span.set_attribute("rag.destination", destination)
        span.set_attribute("rag.interests", interests or "")
        span.set_attribute("rag.k", k)
        
        class RagTracer:
            def success(self, method: str, count: int):
                span.set_attribute("rag.search_method", method)
                span.set_attribute("rag.results_count", count)
                span.set_attribute("rag.fallback_triggered", False)
                add_event(f"RAG {method} search successful", {"count": count})
            
            def fallback(self, method: str, count: int):
                span.set_attribute("rag.search_method", method)
                span.set_attribute("rag.results_count", count)
                span.set_attribute("rag.fallback_triggered", True)
                add_event(f"RAG fallback to {method}", {"count": count})
            
            def failed(self):
                span.set_attribute("rag.search_method", "none")
                span.set_attribute("rag.results_count", 0)
                span.set_attribute("rag.fallback_triggered", True)
                add_event("RAG retrieval failed", {"error": "no_results"})
                set_status(StatusCode.ERROR, "No retrieval results")
        
        yield RagTracer()


@contextmanager
def trace_search_api(query: str):
    """
    Trace search API with fallback monitoring.
    
    Usage:
        with trace_search_api("Tokyo travel info") as tracer:
            try:
                result = tavily_search(query)
                tracer.success("tavily", len(result))
            except:
                result = llm_fallback(query)
                tracer.fallback("llm", len(result))
    """
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("search_api_call") as span:
        span.set_attribute("search.query", query)
        span.set_attribute("search.query_length", len(query))
        
        class SearchTracer:
            def success(self, api: str, response_length: int):
                span.set_attribute("search.api_used", api)
                span.set_attribute("search.response_length", response_length)
                span.set_attribute("search.fallback_triggered", False)
                add_event(f"{api} API successful", {"length": response_length})
            
            def fallback(self, fallback_type: str, response_length: int):
                span.set_attribute("search.api_used", fallback_type)
                span.set_attribute("search.response_length", response_length)
                span.set_attribute("search.fallback_triggered", True)
                add_event(f"Fallback to {fallback_type}", {"length": response_length})
            
            def failed(self):
                span.set_attribute("search.api_used", "none")
                span.set_attribute("search.response_length", 0)
                span.set_attribute("search.fallback_triggered", True)
                add_event("All search methods failed")
                set_status(StatusCode.ERROR, "Search failed")
        
        yield SearchTracer()


@contextmanager
def trace_itinerary_generation(destination: str, duration_num: int):
    """
    Trace itinerary generation with retry/fallback monitoring.
    
    Usage:
        with trace_itinerary_generation("Tokyo", 3) as tracer:
            content = llm.generate()
            actual_days = count_days(content)
            
            if actual_days < duration_num:
                tracer.missing_days(actual_days, [2, 3])
                content = retry_generate()
                tracer.retry_success(3)
            else:
                tracer.success(actual_days)
    """
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("itinerary_generation") as span:
        span.set_attribute("itinerary.destination", destination)
        span.set_attribute("itinerary.days_requested", duration_num)
        
        class ItineraryTracer:
            def success(self, days_generated: int):
                span.set_attribute("itinerary.days_generated", days_generated)
                span.set_attribute("itinerary.missing_days", "[]")
                span.set_attribute("itinerary.retry_triggered", False)
                span.set_attribute("itinerary.fallback_template_used", False)
                add_event("Itinerary generated successfully", {
                    "days": days_generated,
                    "complete": True
                })
            
            def missing_days(self, days_generated: int, missing: List[int]):
                span.set_attribute("itinerary.days_generated", days_generated)
                span.set_attribute("itinerary.missing_days", str(missing))
                add_event("Missing days detected", {
                    "generated": days_generated,
                    "missing": str(missing)
                })
            
            def retry_success(self, final_days: int):
                span.set_attribute("itinerary.days_generated", final_days)
                span.set_attribute("itinerary.retry_triggered", True)
                span.set_attribute("itinerary.fallback_template_used", False)
                add_event("Retry successful", {"final_days": final_days})
            
            def fallback_template(self, days_using_template: List[int]):
                span.set_attribute("itinerary.retry_triggered", True)
                span.set_attribute("itinerary.fallback_template_used", True)
                span.set_attribute("itinerary.template_days", str(days_using_template))
                add_event("Using fallback template", {
                    "days": str(days_using_template),
                    "reason": "retry_failed"
                })
        
        yield ItineraryTracer()


# ============================================================================
# CONSTRAINT VALIDATION
# ============================================================================

def validate_itinerary_constraints(content: str, duration_num: int) -> Dict[str, Any]:
    """
    Validate itinerary output against all constraints.
    
    Returns dict with validation results:
    {
        "day_count": {"expected": 3, "actual": 3, "passed": True},
        "description_length": {"violations": 2, "passed": False},
        "pricing": {"placeholder_count": 0, "passed": True},
        "place_names": {"generic_count": 1, "passed": False}
    }
    """
    results = {}
    
    # 1. Day count validation
    actual_days = content.count("### Day ")
    results["day_count"] = {
        "expected": duration_num,
        "actual": actual_days,
        "passed": actual_days == duration_num
    }
    
    # 2. Description length validation (150+ chars)
    # Match activity descriptions between emoji and next separator
    activity_pattern = r'-\s+[🏛️🍝🌆🎭🎨🏨].*?(?=\n\s+-\s+💵|\n\n|\Z)'
    activities = re.findall(activity_pattern, content, re.DOTALL)
    
    short_descriptions = []
    for activity in activities:
        # Extract just the description part (before the cost line)
        desc_match = re.match(r'-\s+[🏛️🍝🌆🎭🎨🏨]\s+\*\*([^*]+)\*\*\s+-\s+([^-\n]+)', activity, re.DOTALL)
        if desc_match:
            description = desc_match.group(2).strip()
            if len(description) < 150:
                short_descriptions.append({
                    "name": desc_match.group(1),
                    "length": len(description)
                })
    
    results["description_length"] = {
        "min_required": 150,
        "violations": len(short_descriptions),
        "short_descriptions": short_descriptions[:5],  # First 5 examples
        "passed": len(short_descriptions) == 0
    }
    
    # 3. Pricing validation (no placeholders)
    placeholder_patterns = [
        r'💵\s*\$X+',           # $XX, $XXX
        r'💵\s*\$\$(?!\d)',     # $$ (but not part of actual price)
        r'💵\s*Varies',
        r'💵\s*TBD',
        r'💵\s*Cost',
        r'💵\s*\$\d+-\$\d+',    # Should be hyphen not dash
    ]
    
    placeholders = []
    for pattern in placeholder_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        placeholders.extend(matches)
    
    results["pricing"] = {
        "placeholder_count": len(placeholders),
        "examples": placeholders[:5],
        "passed": len(placeholders) == 0
    }
    
    # 4. Place names validation (no generic names)
    generic_patterns = [
        r'\*\*Restaurant\*\*(?!\s+\w)',  # "Restaurant" not followed by name
        r'\*\*Museum\*\*(?!\s+\w)',
        r'\*\*Activity\*\*',
        r'\*\*Attraction\*\*',
        r'\*\*Local\s+\w+\*\*',  # "Local Highlights", etc.
    ]
    
    generic_names = []
    for pattern in generic_patterns:
        matches = re.findall(pattern, content)
        generic_names.extend(matches)
    
    results["place_names"] = {
        "generic_count": len(generic_names),
        "examples": generic_names[:5],
        "passed": len(generic_names) == 0
    }
    
    return results


def trace_validation_results(results: Dict[str, Any]):
    """Add validation results as span attributes and events."""
    span = get_current_span()
    if not span or not span.is_recording():
        return
    
    # Add all validation results as attributes
    for category, data in results.items():
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"validation.{category}.{key}", value)
    
    # Add events for failures
    failures = []
    for category, data in results.items():
        if not data.get("passed", True):
            failures.append(category)
            add_event(f"Validation failed: {category}", data)
    
    # Overall validation status
    all_passed = len(failures) == 0
    span.set_attribute("validation.all_passed", all_passed)
    span.set_attribute("validation.failure_count", len(failures))
    
    if not all_passed:
        set_status(StatusCode.ERROR, f"Validation failures: {', '.join(failures)}")
    
    return all_passed


# ============================================================================
# SESSION & REQUEST TRACKING
# ============================================================================

def trace_request_metadata(req):
    """Add request metadata to current span."""
    span = get_current_span()
    if not span or not span.is_recording():
        return
    
    # Core request parameters
    span.set_attribute("request.destination", req.destination)
    span.set_attribute("request.duration", req.duration)
    span.set_attribute("request.budget", req.budget or "not_specified")
    span.set_attribute("request.interests", req.interests or "not_specified")
    span.set_attribute("request.travel_style", req.travel_style or "not_specified")
    
    # Session tracking (if provided)
    if req.session_id:
        span.set_attribute("session.id", req.session_id)
    if req.user_id:
        span.set_attribute("session.user_id", req.user_id)
    if req.turn_index is not None:
        span.set_attribute("session.turn_index", req.turn_index)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@contextmanager
def trace_agent_execution(agent_type: str):
    """
    Trace agent execution with timing.
    
    Usage:
        with trace_agent_execution("research"):
            result = research_agent(state)
    """
    tracer = trace.get_tracer(__name__)
    start_time = time.time()
    
    with tracer.start_as_current_span(f"{agent_type}_agent_execution") as span:
        span.set_attribute("agent.type", agent_type)
        
        try:
            yield span
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            span.set_attribute("agent.execution_time_ms", execution_time_ms)
            add_event(f"{agent_type} agent completed", {
                "duration_ms": execution_time_ms
            })


def trace_tool_calls(tool_calls: List[Dict[str, Any]]):
    """Add tool call metrics to current span."""
    span = get_current_span()
    if not span or not span.is_recording():
        return
    
    span.set_attribute("tools.total_calls", len(tool_calls))
    
    # Count by agent
    agents = {}
    for call in tool_calls:
        agent = call.get("agent", "unknown")
        agents[agent] = agents.get(agent, 0) + 1
    
    for agent, count in agents.items():
        span.set_attribute(f"tools.{agent}_calls", count)
    
    # List tool names
    tool_names = [call.get("tool", "unknown") for call in tool_calls]
    span.set_attribute("tools.names", ", ".join(tool_names))

