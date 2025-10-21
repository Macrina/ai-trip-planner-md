# Custom Tracing - Implementation Guide

## Quick Start

### 1. Import the custom tracing module
```python
import custom_tracing as ct
```

### 2. Add traces at key points (see below)

---

## 📍 Implementation Locations

### **A. RAG Retriever Fallback** (Line ~200-230)

**Current code:**
```python
def retrieve(self, destination: str, interests: Optional[str], *, k: int = 3):
    if not ENABLE_RAG or self.is_empty:
        return []

    # Use vector search if available, otherwise fall back to keywords
    if not self._vectorstore:
        return self._keyword_fallback(destination, interests, k=k)

    query = destination
    if interests:
        query = f"{destination} with interests {interests}"
    
    try:
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
        docs = retriever.invoke(query)
    except Exception:
        return self._keyword_fallback(destination, interests, k=k)
```

**Add tracing:**
```python
def retrieve(self, destination: str, interests: Optional[str], *, k: int = 3):
    import custom_tracing as ct
    
    with ct.trace_rag_retrieval(destination, interests, k) as tracer:
        if not ENABLE_RAG or self.is_empty:
            tracer.failed()
            return []

        # Use vector search if available
        if not self._vectorstore:
            results = self._keyword_fallback(destination, interests, k=k)
            tracer.fallback("keyword", len(results))
            return results

        query = destination
        if interests:
            query = f"{destination} with interests {interests}"
        
        try:
            # Vector search
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
            
            # Format results
            top_docs = docs[:k]
            results = [...]  # existing code
            
            tracer.success("vector", len(results))
            return results
            
        except Exception:
            # Fallback to keyword
            results = self._keyword_fallback(destination, interests, k=k)
            tracer.fallback("keyword", len(results))
            return results
```

---

### **B. Search API Fallback** (Line ~294-356)

**Current code:**
```python
def _search_api(query: str) -> Optional[str]:
    query = query.strip()
    if not query:
        return None

    # Try Tavily first
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            # ... Tavily API call ...
            if combined:
                return _compact(combined)
        except Exception:
            pass  # Fail gracefully

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            # ... SerpAPI call ...
            if combined:
                return _compact(combined)
        except Exception:
            pass

    return None  # No search APIs configured
```

**Add tracing:**
```python
def _search_api(query: str) -> Optional[str]:
    import custom_tracing as ct
    
    query = query.strip()
    if not query:
        return None
    
    with ct.trace_search_api(query) as tracer:
        # Try Tavily first
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                # ... Tavily API call ...
                if combined:
                    result = _compact(combined)
                    tracer.success("tavily", len(result))
                    return result
            except Exception:
                pass  # Try next API

        # Try SerpAPI as fallback
        serp_key = os.getenv("SERPAPI_API_KEY")
        if serp_key:
            try:
                # ... SerpAPI call ...
                if combined:
                    result = _compact(combined)
                    tracer.fallback("serpapi", len(result))
                    return result
            except Exception:
                pass
        
        # No APIs available
        tracer.failed()
        return None
```

---

### **C. LLM Fallback Tracking** (Line ~359-371)

**Add tracing when this function is called:**
```python
def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    import custom_tracing as ct
    
    # Log that we're using LLM fallback
    ct.add_event("Using LLM fallback", {
        "instruction_length": len(instruction),
        "has_context": context is not None
    })
    ct.set_attribute("fallback.type", "llm")
    
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    
    response = llm.invoke([
        SystemMessage(content="You are a concise travel assistant."),
        HumanMessage(content=prompt),
    ])
    
    result = _compact(response.content)
    ct.set_attribute("fallback.response_length", len(result))
    
    return result
```

---

### **D. Itinerary Generation with Retry** (Line ~762-1054)

**Add tracing around validation logic:**
```python
def itinerary_agent(state: TripState) -> TripState:
    import custom_tracing as ct
    
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    
    # Extract duration number
    duration_num = 1
    try:
        duration_str = duration.lower().replace("days", "").replace("day", "").strip()
        duration_num = int(duration_str) if duration_str.isdigit() else 1
        if duration_num > 5:
            duration_num = 5
            duration = "5 days"
    except:
        duration_num = 1
    
    with ct.trace_itinerary_generation(destination, duration_num) as tracer:
        # ... prompt construction (lines 783-954) ...
        
        # Itinerary agent execution
        res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
        content = res.content
        
        # Validate that all required days are present
        missing_days = []
        for day in range(1, duration_num + 1):
            day_header = f"### Day {day}:"
            if day_header not in content:
                missing_days.append(day)
        
        if missing_days:
            tracer.missing_days(duration_num - len(missing_days), missing_days)
            
            # RETRY: Generate missing days
            try:
                retry_res = llm.invoke([SystemMessage(content=retry_prompt)])
                retry_content = retry_res.content
                
                still_missing = [d for d in missing_days if f"### Day {d}:" not in retry_content]
                
                if not still_missing:
                    content += "\n\n" + retry_content
                    tracer.retry_success(duration_num)
                else:
                    raise Exception(f"Still missing: {still_missing}")
                    
            except Exception as e:
                # Fallback: Use template
                tracer.fallback_template(missing_days)
                for day in missing_days:
                    # ... add fallback template content ...
                    pass
        else:
            # All days generated successfully
            actual_days = content.count("### Day ")
            tracer.success(actual_days)
        
        # ... rest of processing (image replacement, etc.) ...
        
        return {"messages": [SystemMessage(content=content)], "final": content}
```

---

### **E. Request-Level Tracing** (Line ~1311-1374)

**Add tracing at API endpoint:**
```python
@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    import custom_tracing as ct
    from opentelemetry import trace
    
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("plan_trip_request") as span:
        # Add request metadata
        ct.trace_request_metadata(req)
        
        # Build and execute graph
        graph = build_graph()
        state = {
            "messages": [],
            "trip_request": req.model_dump(),
            "tool_calls": [],
        }
        
        # Execute the graph
        out = graph.invoke(state)
        
        # Add tool call metrics
        tool_calls = out.get("tool_calls", [])
        ct.trace_tool_calls(tool_calls)
        
        # Validate final result
        final_result = out.get("final", "")
        if final_result:
            # Extract duration for validation
            duration_num = 1
            try:
                duration_str = req.duration.lower().replace("days", "").strip()
                duration_num = int(duration_str) if duration_str.isdigit() else 1
            except:
                pass
            
            # Run validation
            validation_results = ct.validate_itinerary_constraints(
                final_result, 
                duration_num
            )
            
            # Add validation results to trace
            ct.trace_validation_results(validation_results)
            
            # ... rest of URL fixing code ...
        
        return TripResponse(result=final_result, tool_calls=tool_calls)
```

---

## 🧪 Testing Custom Traces

### Test Script

Create `backend/test_custom_tracing.py`:

```python
#!/usr/bin/env python3
"""Test custom tracing implementation."""

import os
from dotenv import load_dotenv
load_dotenv()

# Setup Arize
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    api_key=os.getenv("ARIZE_API_KEY"),
    project_name="ai-trip-planner-custom-traces",
    batch=False,
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Test custom traces
import custom_tracing as ct

print("Testing custom tracing...")

# 1. Test RAG retrieval trace
with ct.trace_rag_retrieval("Paris", "food, art", k=3) as tracer:
    # Simulate vector search success
    tracer.success("vector", 3)

# 2. Test search API trace
with ct.trace_search_api("Paris travel info") as tracer:
    # Simulate API fallback
    tracer.fallback("llm", 200)

# 3. Test itinerary generation trace
with ct.trace_itinerary_generation("Paris", 3) as tracer:
    # Simulate missing days
    tracer.missing_days(2, [3])
    # Simulate retry success
    tracer.retry_success(3)

# 4. Test validation
content = """
### Day 1: Test
- 🏛️ **Eiffel Tower** - A short description
  - 💵 $XX-YY | ⏱️ 2 hours
### Day 2: Test
- 🍝 **Restaurant** - Another description
  - 💵 $20-30 | ⏱️ 1 hour
"""

results = ct.validate_itinerary_constraints(content, 3)
ct.trace_validation_results(results)

print("✅ Custom tracing test complete!")
print("Check Arize dashboard: https://app.arize.com/")
print("Project: ai-trip-planner-custom-traces")
```

### Run Test

```bash
cd backend
source ../venv312/bin/activate
python test_custom_tracing.py
```

---

## 📊 Viewing Traces in Arize

### 1. Go to Dashboard
https://app.arize.com/ → Project: `ai-trip-planner`

### 2. Filter by Custom Attributes

**Fallback monitoring:**
```
rag.fallback_triggered = true
search.fallback_triggered = true
itinerary.retry_triggered = true
```

**Validation failures:**
```
validation.all_passed = false
validation.day_count.passed = false
validation.description_length.violations > 0
```

### 3. Create Custom Metrics

**Fallback Rate Dashboard:**
- RAG Fallback % = `COUNT WHERE rag.fallback_triggered / COUNT(rag_retrieval)`
- API Fallback % = `COUNT WHERE search.fallback_triggered / COUNT(search_api_call)`
- Retry Rate % = `COUNT WHERE itinerary.retry_triggered / COUNT(itinerary_generation)`

**Quality Dashboard:**
- Validation Pass Rate = `COUNT WHERE validation.all_passed / COUNT(plan_trip_request)`
- Avg Description Length Violations = `AVG(validation.description_length.violations)`
- Pricing Error Rate = `COUNT WHERE validation.pricing.passed = false / COUNT`

---

## 🎯 Next Steps

1. ✅ Review this guide
2. ✅ Implement traces at each location (A-E above)
3. ✅ Run test script to verify traces work
4. ✅ Generate real trip requests and check Arize
5. ✅ Create dashboards for key metrics
6. ✅ Set up alerts for high fallback rates or quality issues

---

## 💡 Tips

- **Start small:** Implement A (RAG fallback) first, test, then add more
- **Check traces:** Use console logging (`log_to_console=True`) initially
- **Iterate:** Add more attributes as you discover what's useful
- **Alert wisely:** Set thresholds based on real data, not guesses

