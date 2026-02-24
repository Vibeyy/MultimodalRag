# Streamlit UI Guide

## Starting the Streamlit UI

```cmd
# Install Streamlit (if not already installed)
pip install streamlit==1.31.1

# Start the UI
python scripts\start_streamlit.py
```

The UI will be available at: `http://localhost:8501`

## Features

### 💬 Chat Page

**Interactive document Q&A with conversation history**

**Features:**
- Real-time question answering
- Conversation history (last 5 queries)
- Expandable query details with metadata
- Citation display with sources
- Hallucination detection results
- Latency tracking

**Settings (Sidebar):**
- **Retrieval Top-K**: Adjust number of chunks (1-20)
- **Query Expansion**: Enable automatic query variants
- **Hallucination Detection**: Validate response grounding

**Example Flow:**
1. Type your question in the input field
2. Click "🚀 Ask"
3. View answer with citations
4. Check metadata in expandable section
5. Review conversation history

**Metrics Displayed:**
- Retrieved chunks count
- Query latency
- Hallucination score (if enabled)
- Error count

### 📁 Ingest Documents Page

**Upload and process documents with progress tracking**

**Features:**
- Multi-file upload (drag & drop supported)
- Tag management
- Collection recreation option (⚠️ deletes all data)
- Real-time progress bar
- Detailed ingestion results
- Ingestion history (last 5 jobs)

**Supported Formats:**
- PDF (.pdf) - Text + OCR
- Images (.png, .jpg, .jpeg) - OCR
- Text (.txt, .md) - Direct

**Workflow:**
1. Upload files (drag & drop or browse)
2. Add tags (optional, comma-separated)
3. Check "Recreate Collection" if needed (⚠️ WARNING)
4. Click "📥 Start Ingestion"
5. Monitor progress bar
6. Review results and errors

**Results:**
- Documents processed
- Total chunks created
- Text vs. Image chunk breakdown
- Error messages (if any)

### 📊 System Status Page

**Monitor vector store health and statistics**

**Features:**
- Vector store connection status
- Collection name and chunk count
- Vector configuration details
- Session statistics
- Chat and ingestion metrics

**Information Displayed:**

**Vector Store:**
- Status: 🟢 Connected / 🔴 Disconnected
- Collection name
- Total chunks stored

**Vector Configuration:**
- Text vector: 1024d, Cosine distance
- Image vector: 768d, Cosine distance

**Session Statistics:**
- Total chat queries
- Average query latency
- Ingestion jobs completed
- Total chunks ingested

**Actions:**
- 🔄 Refresh: Update all statistics

### 🧪 Evaluation Page

**Run RAGAS evaluation on test cases**

**Features:**
- JSON test case editor
- Evaluation metrics display
- Tips for CLI evaluation

**Metrics:**
- Retrieval: Precision, Recall, F1, MRR, NDCG
- Generation: Answer relevancy, Faithfulness
- Overall: Hallucination rate, Latency

**Note:** For full evaluation, use CLI:
```cmd
python scripts\evaluate_rag.py data\test_cases.json
```

## Sidebar Features

### Navigation
- 💬 Chat
- 📁 Ingest Documents
- 📊 System Status
- 🧪 Evaluation

### Settings (Persistent)
- **Retrieval Top-K** (1-20): Controls number of chunks
- **Query Expansion**: Generate query variants
- **Hallucination Detection**: Validate responses

### Quick Actions
- **🔄 Refresh Status**: Clear cache and reload
- **🗑️ Clear Chat History**: Reset conversation

## Why Streamlit?

**Streamlit Advantages for RAG Systems:**
- ✅ Persistent conversation history via session state
- ✅ Professional multi-page architecture
- ✅ Custom CSS styling and layouts
- ✅ Better state management
- ✅ Progress bars and real-time updates
- ✅ Expandable sections and accordions
- ✅ Metric cards with custom styling
- ✅ Production-ready deployment options

## Tips & Best Practices

### Chat Tips

1. **Use Conversation History**: Review previous Q&A before asking
2. **Expand Details**: Click expanders for full metadata
3. **Adjust Top-K**: Lower for focused, higher for comprehensive
4. **Enable Query Expansion**: Better for complex questions
5. **Check Hallucination**: Use for factual, critical queries

### Ingestion Tips

1. **Batch Upload**: Select multiple files at once
2. **Use Tags**: Organize with meaningful tags (e.g., "research-2024")
3. **Monitor Progress**: Watch progress bar for large uploads
4. **Check History**: Review previous ingestion jobs
5. **⚠️ Recreate Warning**: Only use when necessary (deletes all data!)

### Performance

**Chat Latency:**
- Simple query: ~2-3 seconds
- With expansion: ~4-6 seconds
- With hallucination check: ~5-8 seconds

**Ingestion Speed:**
- Text file (10KB): ~1-2 seconds
- PDF (10 pages): ~5-10 seconds
- PDF with images (50 pages): ~30-60 seconds (OCR dependent)

## Customization

### Change Port

Edit [start_streamlit.py](../scripts/start_streamlit.py):

```python
subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(app_path),
    "--server.port=8501",  # Change this
    "--server.address=0.0.0.0",
])
```

### Change Theme

Edit [streamlit_app.py](../src/multimodal_rag/ui/streamlit_app.py) config:

```python
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded",  # or "collapsed"
)
```

Or use `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Custom CSS

Edit the CSS in [streamlit_app.py](../src/multimodal_rag/ui/streamlit_app.py):

```python
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    /* Add your custom styles */
</style>
""", unsafe_allow_html=True)
```

## Troubleshooting

### UI Won't Start

**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```cmd
pip install streamlit==1.31.1
```

### Connection Failed

**Error**: "Qdrant not connected"

**Solution**:
```cmd
docker-compose ps
docker-compose restart qdrant
```

### Session State Issues

**Problem**: History not persisting

**Solution**: Streamlit uses session state - don't use browser refresh. Use "🔄 Refresh Status" button instead.

### Slow Performance

**Problem**: UI feels sluggish

**Solutions**:
- Use `@st.cache_resource` for component initialization (already implemented)
- Reduce Top-K value
- Disable query expansion for faster queries
- Check Docker resource allocation

## Keyboard Shortcuts

Streamlit shortcuts:
- **C**: Clear cache
- **R**: Rerun script
- **H**: Hide/show sidebar
- **?**: Show keyboard shortcuts

## Mobile Support

Streamlit is responsive and works on mobile, but best experience is on desktop browsers.

## Integration

Run alongside other services:

```cmd
# Terminal 1: API
python scripts\start_api.py

# Terminal 2: Streamlit UI
python scripts\start_streamlit.py
```

Access:
- **Streamlit UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Advanced Features

### Session State Variables

Available in `st.session_state`:
- `chat_history`: List of Q&A pairs
- `ingestion_history`: List of ingestion jobs
- `top_k`: Current retrieval setting
- `expand_query`: Query expansion flag
- `check_hallucination`: Hallucination detection flag

### Caching

Components are cached using `@st.cache_resource`:
- Vector store initialization
- RAG agent initialization
- Embedding models

**Clear cache**: Use "🔄 Refresh Status" button in sidebar

## Production Deployment

For production deployment:

1. **Streamlit Cloud** (Free):
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Deploy (free tier available)

2. **Docker**:
   ```dockerfile
   FROM python:3.11
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["streamlit", "run", "src/multimodal_rag/ui/streamlit_app.py"]
   ```

3. **Self-hosted**:
   - Use systemd or supervisor
   - Nginx reverse proxy
   - SSL certificate (Let's Encrypt)

## Why Streamlit for RAG?

**When to use Streamlit:**
- ✅ Production applications
- ✅Production Ready

**Streamlit Features for Production:**
- ✅ Multi-page architecture for complex apps
- ✅ Custom layouts and styling
- ✅ Persistent session state
- ✅ Professional appearance
- ✅ Easy deployment (Streamlit Cloud, Docker, self-hosted)
- ✅ Real-time updates and progress tracking
- ✅ Built-in state management
- ✅ Mobile-responsive design