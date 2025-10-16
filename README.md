# AI Trip Planner

A **production-ready multi-agent system** built for learning and customization. This repo demonstrates three essential AI engineering patterns that students can study, modify, and adapt for their own use cases.

## What You'll Learn

- 🤖 **Multi-Agent Orchestration**: 4 specialized agents running in parallel using LangGraph
- 🔍 **RAG (Retrieval-Augmented Generation)**: Vector search over curated data with fallback strategies
- 🌐 **API Integration**: Real-time web search with graceful degradation (LLM fallback)
- 📊 **Observability**: Production tracing with Arize for debugging and evaluation
- 🛠️ **Composable Architecture**: Easily adapt from "trip planner" to your own agent system

**Perfect for:** Students learning to build, evaluate, and deploy agentic AI systems.

## ✨ Recent Improvements

### 🎨 Enhanced User Interface
- **Modern Tabbed Interface**: Day-by-day itinerary navigation with smooth transitions
- **Action Buttons**: Beautiful, color-coded buttons for directions, tickets, and photos
- **Responsive Design**: Works perfectly on desktop and mobile devices
- **Accordion Sidebar**: Expandable sections for budget, tips, and useful links
- **Smart Autocomplete**: Destination input with static database + API fallback

### 🖼️ Visual Enhancements
- **Destination Images**: Real city photos for each day using Unsplash API
- **Dynamic Image Loading**: Unique images per day based on destination and theme
- **Fallback System**: Graceful degradation to Picsum when Unsplash unavailable
- **Image Optimization**: Proper sizing and loading for web performance

### 🎯 Improved Content Quality
- **Specific Place Names**: No more generic "Restaurant/Activity" - uses real landmarks
- **Working Photo Links**: Fixed Unsplash URLs to display actual destination photos
- **Enhanced Budget Display**: Automatic total calculation with visual breakdown
- **Better Action Links**: All external links open in new tabs with proper styling

### 🛠️ Technical Improvements
- **Tailwind CSS Integration**: Local build system replacing CDN for better performance
- **Node.js Tooling**: Added npm scripts, live-server, prettier, and eslint
- **Static File Serving**: FastAPI now serves frontend assets efficiently
- **Post-Processing**: Automatic URL format correction and content validation

## Key Features

### 🎯 Smart Itinerary Generation
- **Multi-Agent System**: Research, Budget, Local, and Itinerary agents work in parallel
- **Real-time Data**: Weather, web search, local recommendations with API fallbacks
- **RAG System**: Local guide database with web search fallback for authentic experiences
- **Specific Recommendations**: Real landmarks and restaurants instead of generic suggestions

### 🎨 Modern User Interface
- **Tabbed Navigation**: Day-by-day itinerary with smooth transitions and modern design
- **Action Buttons**: Color-coded buttons for directions (blue), tickets (green), and photos (purple)
- **Responsive Design**: Works perfectly on desktop and mobile with full-height layout
- **Smart Autocomplete**: Destination input with static database + API fallback
- **Accordion Sidebar**: Expandable sections for Budget, Tips, and Useful Links

### 🖼️ Visual Experience
- **Destination Images**: Real city photos for each day using Unsplash API
- **Dynamic Loading**: Unique images per day based on destination and theme
- **Fallback System**: Graceful degradation to Picsum when Unsplash unavailable
- **Enhanced Budget**: Automatic total calculation with visual breakdown and gradient styling

### 🛠️ Developer Experience
- **Node.js Integration**: npm scripts, live-server, prettier, eslint for modern development
- **Tailwind CSS**: Local build system with PostCSS for optimized styling
- **Static File Serving**: FastAPI serves frontend assets efficiently
- **Hot Reload**: Development server with automatic CSS rebuilding

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                             │
│                    (destination, duration, interests)            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Endpoint      │
                    │   + Session Tracking    │
                    │   + Static File Serving │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LangGraph Workflow    │
                    │   (Parallel Execution)  │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼─────┐           ┌─────▼──────┐         ┌──────▼─────┐
   │ Research │           │   Budget   │         │   Local    │
   │  Agent   │           │   Agent    │         │   Agent    │
   └────┬─────┘           └─────┬──────┘         └──────┬─────┘
        │                        │                        │
        │ Tools:                 │ Tools:                 │ Tools + RAG:
        │ • essential_info       │ • budget_basics        │ • local_flavor
        │ • weather_brief        │ • attraction_prices    │ • hidden_gems
        │ • visa_brief           │                        │ • Vector search
        │                        │                        │   (90+ guides)
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                            ┌────▼─────┐
                            │Itinerary │
                            │  Agent   │
                            │(Synthesis)│
                            │+ Images  │
                            └────┬─────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Final Itinerary       │
                    │   + Tool Call Metadata  │
                    │   + Day Images          │
                    │   + Action Buttons      │
                    └─────────────────────────┘

All agents, tools, and LLM calls → Arize Observability Platform
```

## Learning Paths

### 🎓 Beginner Path
1. **Setup & Run** (15 min)
   - Clone repo, configure `.env` with OpenAI key
   - Start server: `./start.sh`
   - Test API: `python "test scripts/test_api.py"`

2. **Observe & Understand** (30 min)
   - Make a few trip planning requests
   - View traces in Arize dashboard
   - Understand agent execution flow and tool calls

3. **Experiment with Prompts** (30 min)
   - Modify agent prompts in `backend/main.py`
   - Change tool descriptions
   - See how it affects outputs

### 🚀 Intermediate Path
1. **Enable Advanced Features** (20 min)
   - Set `ENABLE_RAG=1` to use vector search
   - Add `TAVILY_API_KEY` for real-time web search
   - Compare results with/without these features

2. **Add Custom Data** (45 min)
   - Add your own city to `backend/data/local_guides.json`
   - Test RAG retrieval with your data
   - Understand fallback strategies

3. **Create a New Tool** (1 hour)
   - Add a new tool (e.g., `restaurant_finder`)
   - Integrate it into an agent
   - Test and trace the new tool calls

### 💪 Advanced Path
1. **Change the Domain** (2-3 hours)
   - Use Cursor AI to help transform the system
   - Example: Change from "trip planner" to "PRD generator"
   - Modify state, agents, and tools for your use case

2. **Add a New Agent** (2 hours)
   - Create a 5th agent (e.g., "activities planner")
   - Update the LangGraph workflow
   - Test parallel vs sequential execution

3. **Implement Evaluations** (2 hours)
   - Use `test scripts/synthetic_data_gen.py` as a base
   - Create evaluation criteria for your domain
   - Set up automated evals in Arize

## Common Use Cases (Built by Students)

Students have successfully adapted this codebase for:

- **📝 PR Description Generator**
  - Agents: Code Analyzer, Context Gatherer, Description Writer
  - Replaces travel tools with GitHub API calls
  - Used by tech leads to auto-generate PR descriptions

- **🎯 Customer Support Analyst**
  - Agents: Ticket Classifier, Knowledge Base Search, Response Generator
  - RAG over support docs instead of local guides
  - Routes tickets and drafts responses

- **🔬 Research Assistant**
  - Agents: Web Searcher, Academic Search, Citation Manager, Synthesizer
  - Web search for papers + RAG over personal library
  - Generates research summaries with citations

- **📱 Content Planning System**
  - Agents: SEO Researcher, Social Media Planner, Blog Scheduler
  - Tools for keyword research, trend analysis
  - Creates cross-platform content calendars

- **🏗️ Architecture Review Agent**
  - Agents: Code Scanner, Pattern Detector, Best Practices Checker
  - RAG over architecture docs
  - Reviews PRs for architectural concerns

**💡 Your Turn**: Use Cursor AI to help you adapt this system for your domain!

## Quickstart

1) Requirements
- Python 3.10+ (Docker optional)

2) Configure environment
- Copy `backend/.env.example` to `backend/.env`.
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`.
- Optional: `ARIZE_SPACE_ID` and `ARIZE_API_KEY` for tracing.

3) Install dependencies
```bash
# Install Node.js and nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
nvm use --lts

# Install project dependencies
npm install
npm run install-deps  # Installs Python dependencies

# Alternative: Direct Python installation
cd backend
uv pip install -r requirements.txt   # faster, deterministic installs
# If uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh
# Fallback: pip install -r requirements.txt
```

4) Run
```bash
# make sure you are back in the root directory of ai-trip-planner
cd ..

# Option 1: Using npm scripts (recommended)
npm start                       # starts backend on 8000; serves minimal UI at '/'

# Option 2: Using start.sh script
./start.sh                      # starts backend on 8000; serves minimal UI at '/'

# Option 3: Direct Python command
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
 - Minimal UI: http://localhost:8000/

Docker (optional)
```bash
docker-compose up --build
```

## Project Structure
- `backend/`: FastAPI app (`main.py`), LangGraph agents, tracing hooks, static file serving.
- `frontend/`: Complete UI with `index.html`, compiled `styles.css`, and `cities.json` for autocomplete.
- `src/`: Tailwind CSS source files (`input.css`) for local build system.
- `optional/airtable/`: Airtable integration (optional, not on critical path).
- `test scripts/`: `test_api.py`, `synthetic_data_gen.py` for quick checks/evals.
- Root: `start.sh`, `docker-compose.yml`, `README.md`, `package.json`, `tailwind.config.js`, `postcss.config.js`.

## Development Commands

### Backend Development
- Backend (dev): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- API smoke test: `python "test scripts"/test_api.py`
- Synthetic evals: `python "test scripts"/synthetic_data_gen.py --base-url http://localhost:8000 --count 12`

### Frontend Development
- Build CSS: `npm run build-css` (watch mode) or `npm run build-css-prod` (production)
- Start dev server: `npm start` (starts backend + serves frontend)
- Frontend only: `live-server frontend --port=3000 --open=/index.html`

## API
- POST `/plan-trip` → returns a generated itinerary with images and action buttons.
  Example body:
  ```json
  {"destination":"Tokyo, Japan","duration":"7 days","budget":"$2000","interests":"food, culture"}
  ```
- GET `/health` → simple status.
- GET `/cities.json` → autocomplete data for destination input.
- GET `/static/*` → serves frontend assets (CSS, images, etc.).

## Notes on Tracing (Optional)
- If `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set, OpenInference exports spans for agents/tools/LLM calls. View at https://app.arize.com.

## Optional Features

### RAG: Vector Search for Local Guides

The local agent can use vector search to retrieve curated local experiences from a database of 90+ real-world recommendations:

- **Enable**: Set `ENABLE_RAG=1` in your `.env` file
- **Requirements**: Requires `OPENAI_API_KEY` for embeddings
- **Data**: Uses curated experiences from `backend/data/local_guides.json`
- **Benefits**: Provides grounded, cited recommendations with sources
- **Learning**: Great example of production RAG patterns with fallback strategies

When disabled (default), the local agent uses LLM-generated responses.

See `RAG.md` for detailed documentation.

### Web Search: Real-Time Tool Data

Tools can call real web search APIs (Tavily or SerpAPI) for up-to-date travel information:

- **Enable**: Add `TAVILY_API_KEY` or `SERPAPI_API_KEY` to your `.env` file
- **Benefits**: Real-time data for weather, attractions, prices, customs, etc.
- **Fallback**: Without API keys, tools automatically fall back to LLM-generated responses
- **Learning**: Demonstrates graceful degradation and multi-tier fallback patterns

Recommended: Tavily (free tier: 1000 searches/month) - https://tavily.com

### City Images: Real Destination Photos

The itinerary agent can fetch real city images from Unsplash for each day of your trip:

- **Enable**: Add `UNSPLASH_API_KEY` to your `.env` file
- **Benefits**: Beautiful, relevant city photos for each day of your itinerary
- **Fallback**: Without API key, uses destination-based random images from Picsum
- **Learning**: Demonstrates API integration with graceful fallback patterns

Get your free Unsplash API key: https://unsplash.com/developers

## Development Tools

### Node.js Setup
This project now includes Node.js tooling for enhanced development:

- **nvm**: Node Version Manager for switching between Node.js versions
- **npm**: Package manager for JavaScript dependencies
- **live-server**: Local development server for frontend
- **prettier**: Code formatter for consistent styling

### Available npm Scripts
```bash
npm start          # Start the FastAPI backend server
npm run dev        # Start development server with auto-reload
npm run setup      # Initial project setup (installs all dependencies)
npm run install-deps # Install Python dependencies only
npm test           # Run Python tests
npm run lint       # Run Python linting
npm run format     # Format Python code with black
```

### Development Workflow
```bash
# Start backend
npm start

# In another terminal, start frontend development server
live-server frontend --port=3000 --open=/index.html
```

## Next Steps

1. **🎯 Start Simple**: Get it running, make some requests, view traces
2. **🔍 Explore Code**: Read through `backend/main.py` to understand patterns
3. **🛠️ Modify Prompts**: Change agent behaviors to see what happens
4. **🚀 Enable Features**: Try RAG and web search
5. **💡 Build Your Own**: Use Cursor to transform it into your agent system

## Troubleshooting

- **401/empty results**: Verify `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `backend/.env`
- **No traces**: Ensure Arize credentials are set and reachable
- **Port conflicts**: Stop existing services on 3000/8000 or change ports
- **RAG not working**: Check `ENABLE_RAG=1` and `OPENAI_API_KEY` are both set
- **Slow responses**: Web search APIs may timeout; LLM fallback will handle it
- **Layout broken**: Ensure `npm run build-css-prod` has been run to generate `frontend/styles.css`
- **Images not loading**: Check `UNSPLASH_API_KEY` is set, or images will fallback to Picsum
- **Generic suggestions**: Ensure backend is running latest code with specific naming rules
- **Button links not working**: Check that markdown renderer is properly configured
- **Node.js issues**: Use `nvm use` to switch to correct Node.js version (LTS)

## Deploy on Render
- This repo includes `render.yaml`. Connect your GitHub repo in Render and deploy as a Web Service.
- Render will run: `pip install -r backend/requirements.txt` and `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and optional Arize vars in the Render dashboard.
