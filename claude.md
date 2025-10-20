# Claude/LLM Integration Notes

## Current LLM Setup
- **Model**: gpt-3.5-turbo (OpenAI)
- **Framework**: LangChain + LangGraph
- **Architecture**: Multi-agent system with parallel execution
- **Temperature**: 0.7
- **Max Tokens**: 1500

## Multi-Agent System

### Research Agent
- **Tools**: `essential_info`, `weather_brief`, `visa_brief`
- **Purpose**: Gather destination information, weather, visa requirements
- **Output**: Comprehensive travel research summary

### Budget Agent  
- **Tools**: `budget_basics`, `attraction_prices`
- **Purpose**: Analyze costs and provide budget breakdown
- **Output**: Detailed cost estimates by category

### Local Agent
- **Tools**: `local_flavor`, `local_customs`, `hidden_gems`
- **Purpose**: Find authentic experiences and local recommendations
- **Features**: Uses RAG for curated local guides from `local_guides.json`
- **Output**: Curated list of local experiences

### Itinerary Agent
- **Purpose**: Synthesizes all agent outputs into day-by-day itinerary
- **Format**: Markdown with emojis, images, action links
- **Features**: 
  - Hero images for each day (Unsplash/Picsum)
  - Real place names with Google Maps/GetYourGuide/Unsplash links
  - Realistic USD pricing
  - Minimum 150-character activity descriptions

## Observability & Monitoring

### Arize Integration
- **Platform**: Arize AI (https://app.arize.com)
- **Space ID**: `U3BhY2U6MjA1Ok9SZXY=`
- **Project Name**: `ai-trip-planner`
- **Instrumentation**: 
  - `arize-phoenix>=4.0.0`
  - `openinference-instrumentation-openai>=0.1.39`
  - `openinference-instrumentation-langchain>=0.1.0`

### What's Traced
- All OpenAI API calls (model, tokens, messages)
- LangChain operations
- Tool invocations
- Agent workflows

### Setup
```python
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    space_id="U3BhY2U6MjA1Ok9SZXY=",
    api_key="[YOUR_API_KEY]",
    project_name="ai-trip-planner",
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

## Prompt Engineering Patterns

### Tool Fallback Strategy
- Primary: Web search APIs (Tavily, SerpAPI)
- Fallback: LLM generation when APIs unavailable
- Pattern: `_search_api()` → `_llm_fallback()`

### Output Formatting
- Use structured prompts with clear formatting rules
- Enforce markdown with specific emoji/link patterns
- Set character minimums for quality (150+ chars per activity)
- Include validation and retry logic for missing content

### Context Management
- Limit context to 400 chars per agent output
- Use RAG to inject relevant local guides
- Compact API responses to stay within token limits

## Common Issues & Solutions

### Issue: Generic Activity Names
**Solution**: Prompt explicitly requires real place names, includes examples, rejects generic placeholders

### Issue: Incomplete Itineraries
**Solution**: Validate day count, retry with focused prompt for missing days, fallback template

### Issue: API Quota Limits
**Solution**: Graceful degradation with LLM fallbacks, no hard failures

### Issue: Unsplash Photo Links Breaking
**Solution**: Convert to lowercase, replace spaces with hyphens, use `/s/photos/` format

## RAG System

### Local Guides Database
- **File**: `backend/data/local_guides.json`
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Store**: In-memory (LangChain)
- **Retrieval**: Top-k similarity search with keyword fallback
- **Feature Flag**: `ENABLE_RAG=1` (opt-in)

### Usage Example
```python
results = GUIDE_RETRIEVER.retrieve(
    destination="Paris",
    interests="food, art",
    k=3
)
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional for enhanced features
TAVILY_API_KEY=...
WEATHER_API_KEY=...
UNSPLASH_API_KEY=...

# Arize (optional)
# Configured in code, not env vars

# Features
ENABLE_RAG=1
TEST_MODE=0
```

## Key Learnings

### What Works Well
- ✅ Parallel agent execution speeds up response time
- ✅ Structured prompts with examples improve output quality
- ✅ Graceful degradation keeps app working without external APIs
- ✅ Validation + retry logic handles LLM inconsistencies
- ✅ RAG provides authentic local recommendations

### Challenges
- ⚠️ LLMs sometimes skip days in multi-day itineraries
- ⚠️ Generic placeholder names need explicit rejection in prompts
- ⚠️ Cost estimation requires detailed guidance per destination
- ⚠️ Arize gRPC export shows cosmetic errors (StatusCode.INTERNAL)

### Best Practices
1. **Always include examples** in prompts for complex formats
2. **Validate LLM outputs** before returning to users
3. **Set explicit minimums** (character counts, required fields)
4. **Use fallbacks** at every external dependency point
5. **Test with TEST_MODE=1** for development without API costs

## Future Improvements

### Short Term
- [ ] Add retry logic with exponential backoff for API calls
- [ ] Improve cost estimation with regional pricing data
- [ ] Add user feedback loop to improve recommendations
- [ ] Create automated tests for agent outputs

### Long Term
- [ ] Add Claude support (Anthropic) as alternative LLM
- [ ] Implement streaming responses for real-time updates
- [ ] Add conversation memory for follow-up questions
- [ ] Build evaluation framework for itinerary quality
- [ ] Add A/B testing for different prompt strategies

## Testing

### Manual Tests
```bash
# Test Arize tracing
python backend/test_arize_trace.py

# Debug trace creation
python backend/test_arize_debug.py

# Test in mock mode (no API costs)
TEST_MODE=1 python backend/main.py
```

### What to Test
- [ ] All agents produce valid outputs
- [ ] Links are properly formatted and specific
- [ ] Images load correctly for each day
- [ ] Budget calculations are realistic
- [ ] Fallbacks work when APIs are disabled

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Arize Phoenix Docs](https://docs.arize.com/phoenix)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

**Last Updated**: October 2025  
**Maintainer**: AI Trip Planner Team

