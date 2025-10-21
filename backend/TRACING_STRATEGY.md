# AI Trip Planner - Tracing Strategy

## Architecture Overview

```
User Request → API Endpoint
    ↓
LangGraph Multi-Agent System
    ├── Research Agent (parallel)
    ├── Budget Agent (parallel)
    ├── Local Agent (parallel) ← RAG Retriever
    └── Itinerary Agent (sequential)
        ↓
Response Validation & Fixing
    ↓
Final Output
```

---

## 🎯 Key Tracing Points

### 1. **Fallback Mechanisms** (CRITICAL)

#### **A. RAG Retriever Fallback**
Location: `LocalGuideRetriever.retrieve()` (line 185-231)

**What to trace:**
- ✅ Vector search attempted
- ⚠️ Vector search failed → keyword fallback triggered
- ❌ Both methods failed → empty results

**Why:** Monitor RAG effectiveness and fallback frequency

**Trace attributes:**
```python
span.set_attribute("rag.search_method", "vector" | "keyword" | "none")
span.set_attribute("rag.results_count", len(results))
span.set_attribute("rag.fallback_triggered", True/False)
```

---

#### **B. Search API Fallback**
Location: `_search_api()` → `_llm_fallback()` (line 294-371)

**What to trace:**
- ✅ Tavily API success
- ⚠️ Tavily failed → SerpAPI attempted
- ⚠️ Both APIs failed → LLM fallback triggered
- ❌ All methods failed

**Why:** Track external API reliability and fallback usage

**Trace attributes:**
```python
span.set_attribute("search.api_used", "tavily" | "serpapi" | "llm_fallback" | "none")
span.set_attribute("search.fallback_triggered", True/False)
span.set_attribute("search.query", query)
span.set_attribute("search.response_length", len(response))
```

---

#### **C. Itinerary Generation Fallback**
Location: `itinerary_agent()` (line 962-1054)

**What to trace:**
- ⚠️ Missing days detected
- 🔄 Retry attempt triggered
- ✅ Retry successful
- ❌ Retry failed → template fallback used

**Why:** Monitor LLM reliability in generating complete itineraries

**Trace attributes:**
```python
span.set_attribute("itinerary.days_requested", duration_num)
span.set_attribute("itinerary.days_generated", actual_days)
span.set_attribute("itinerary.missing_days", missing_days)
span.set_attribute("itinerary.retry_triggered", True/False)
span.set_attribute("itinerary.fallback_template_used", True/False)
```

---

### 2. **Constraint Validation** (QUALITY)

#### **A. Day Count Constraint**
Location: `itinerary_agent()` (line 963-968)

**Constraint:** Generate exactly N days as requested (max 5)

**What to trace:**
```python
span.set_attribute("validation.day_count.expected", duration_num)
span.set_attribute("validation.day_count.actual", actual_days)
span.set_attribute("validation.day_count.passed", actual_days == duration_num)
```

---

#### **B. Description Length Constraint**
Location: Prompt validation (line 847-857)

**Constraint:** Each activity description ≥ 150 characters

**What to trace:**
```python
# Parse final output and count
import re
activities = re.findall(r'-\s+[🏛️🍝🌆].*?-\s+💵', content, re.DOTALL)
short_descriptions = [a for a in activities if len(a) < 150]

span.set_attribute("validation.description_length.min_required", 150)
span.set_attribute("validation.description_length.violations", len(short_descriptions))
span.set_attribute("validation.description_length.passed", len(short_descriptions) == 0)
```

---

#### **C. Pricing Constraint**
Location: Prompt validation (line 859-897)

**Constraint:** Realistic USD prices (not placeholders like $$, $XX)

**What to trace:**
```python
# Parse for placeholder patterns
placeholders = re.findall(r'💵\s*(\$X+|\$\$|Varies|TBD)', content)

span.set_attribute("validation.pricing.placeholder_count", len(placeholders))
span.set_attribute("validation.pricing.passed", len(placeholders) == 0)
```

---

#### **D. Real Place Names Constraint**
Location: Prompt validation (line 836-844)

**Constraint:** No generic names like "Restaurant", "Museum", "Activity"

**What to trace:**
```python
# Parse for generic names
generic_patterns = ['**Restaurant**', '**Museum**', '**Activity**', '**Attraction**']
generic_count = sum(content.count(p) for p in generic_patterns)

span.set_attribute("validation.place_names.generic_count", generic_count)
span.set_attribute("validation.place_names.passed", generic_count == 0)
```

---

### 3. **Performance Metrics** (OPTIMIZATION)

#### **A. Agent Execution Time**
**What to trace:**
```python
span.set_attribute("agent.execution_time_ms", execution_time)
span.set_attribute("agent.type", "research" | "budget" | "local" | "itinerary")
```

---

#### **B. Tool Call Success Rate**
**What to trace:**
```python
span.set_attribute("tools.total_calls", len(tool_calls))
span.set_attribute("tools.successful_calls", successful)
span.set_attribute("tools.failed_calls", failed)
span.set_attribute("tools.success_rate", successful / total)
```

---

### 4. **User Experience** (CONTEXT)

#### **A. Session Tracking**
Location: `TripRequest` model (line 72-74)

**What to trace:**
```python
span.set_attribute("session.id", req.session_id)
span.set_attribute("session.user_id", req.user_id)
span.set_attribute("session.turn_index", req.turn_index)
```

---

#### **B. Request Parameters**
**What to trace:**
```python
span.set_attribute("request.destination", req.destination)
span.set_attribute("request.duration", req.duration)
span.set_attribute("request.budget", req.budget)
span.set_attribute("request.interests", req.interests)
span.set_attribute("request.travel_style", req.travel_style)
```

---

## 🔧 Implementation Priority

### **Phase 1: Fallback Monitoring** (HIGH PRIORITY)
- [ ] Add spans for RAG retriever fallback
- [ ] Add spans for search API fallback
- [ ] Add spans for itinerary retry/fallback
- [ ] Add events when fallbacks trigger

### **Phase 2: Quality Validation** (MEDIUM PRIORITY)
- [ ] Add constraint validation spans
- [ ] Parse final output for violations
- [ ] Log validation failures as events

### **Phase 3: Performance & Context** (LOW PRIORITY)
- [ ] Add timing metrics
- [ ] Add session tracking
- [ ] Add custom metadata

---

## 📊 Custom Metrics to Track

### **Arize Dashboard Queries**

1. **Fallback Rate:**
   ```
   COUNT(spans WHERE fallback_triggered = true) / COUNT(spans)
   ```

2. **Constraint Violations:**
   ```
   COUNT(spans WHERE validation.*.passed = false)
   ```

3. **Average Days Generated vs Requested:**
   ```
   AVG(itinerary.days_generated - itinerary.days_requested)
   ```

4. **API Success Rate:**
   ```
   COUNT(spans WHERE search.api_used != "llm_fallback") / COUNT(spans)
   ```

---

## 🎯 Success Criteria

✅ **Fallback monitoring:** See which fallbacks trigger most frequently  
✅ **Constraint compliance:** Track % of requests that meet all constraints  
✅ **Quality regression:** Alert when validation failures spike  
✅ **Performance baseline:** Establish normal execution times per agent  

---

## 📝 Next Steps

1. Implement Phase 1 tracing (fallback monitoring)
2. Deploy and collect data for 1 week
3. Analyze patterns in Arize dashboard
4. Tune prompts based on constraint violations
5. Implement Phases 2-3 based on findings

