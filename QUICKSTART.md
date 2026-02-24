# 🚀 Quick Start Guide

Get started with the Multimodal RAG Assistant in 10 minutes!

## Prerequisites Checklist

- [ ] Anaconda or Miniconda installed
- [ ] Docker Desktop installed and running
- [ ] Google Gemini API key ([Get free key](https://makersuite.google.com/app/apikey))
- [ ] Langfuse Cloud account ([Sign up free](https://cloud.langfuse.com/))

## Step-by-Step Setup

### 1. Create Conda Environment (2 min)

```cmd
# Navigate to project directory
cd "C:\Users\aakhilesh\OneDrive - Altimetrik Corp\Desktop\New folder"

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate multimodal_rag
```

### 2. Install Dependencies (3 min)

```cmd
# Install Python packages
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
pip check
```

### 3. Configure Environment (1 min)

```cmd
# Copy environment template
copy .env.example .env

# Edit .env file and add your Gemini API key
notepad .env
```

**Required: Add your API keys to `.env`:**
```env

# Langfuse Cloud keys from https://cloud.langfuse.com/
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
```

### 4. Start Docker Services (2 min)

```cmd
# Start Qdrant and Redis
docker-compose up -d

# Verify all services are running
docker-compose ps
```

Expected output:
```
NAME                          STATUS
multimodal_rag_qdrant         Up (healthy)
multimodal_rag_redis          Up (healthy)
```

### 5. Test Installation (1 min)

```cmd
# Test environment configuration
python -c "from multimodal_rag.utils.config import validate_environment; validate_environment(); print('✓ Environment valid!')"
```

## 🎯 First RAG Query (1 min)

### Ingest Sample Document

```cmd
# Create a sample text file
echo "Artificial Intelligence is transforming industries worldwide." > data\raw\sample.txt

# Ingest it (Note: For actual PDFs, place them in data\raw\)
python scripts\ingest_documents.py data\raw\ --tags demo test
```

### Query the System

```cmd
python scripts\query_rag.py "What is artificial intelligence?" --top-k 3
```

Expected output:
```
==============================================================
Answer:
==============================================================

Artificial Intelligence is transforming industries worldwide 
[Source: sample.txt, Page: 1].

Citations (1):
  • sample.txt, Page 1

Metadata:
  Tokens: ~245
  Latency: 1234.56ms
  Has citations: True
==============================================================
```

## 📚 Next Steps

### Process Real PDFs

```cmd
# Place PDFs in data/raw/
# Example: data/raw/company_report.pdf

# Ingest all PDFs
python scripts\ingest_documents.py data\raw\ --tags reports q4-2024

# Query with hallucination detection
python scripts\query_rag.py "What were the Q4 revenue figures?" --check-hallucination
```

### Advanced Features

**Query Expansion:**
```cmd
python scripts\query_rag.py "machine learning applications" --expand-query --top-k 5
```

**Custom Collection:**
```cmd
# Create dedicated collection
python scripts\ingest_documents.py data\raw\finance\ --collection finance_docs --tags finance

# Query specific collection
python scripts\query_rag.py "quarterly earnings" --collection finance_docs
```

## 🔍 Access Dashboards

### Qdrant Dashboard
- URL: http://localhost:6333/dashboard
- View: Vector collections, search, and stats

### Langfuse Cloud Dashboard
- URL: https://cloud.langfuse.com/
- View: Query traces, metrics, token usage, observability

## 🧪 Run Tests

```cmd
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_ingestion.py -v

# Run with coverage
pytest tests/ --cov=src/multimodal_rag --cov-report=html
```

## ❓ Troubleshooting

### Error: "Missing GEMINI_API_KEY"
**Solution:** Edit `.env` and add your API keys:
- Gemini: https://makersuite.google.com/app/apikey
- Langfuse: https://cloud.langfuse.com/ (Settings → API Keys)

### Error: "Connection refused" (Qdrant)
**Solution:** 
```cmd
docker-compose restart qdrant
docker-compose ps
```

### Error: "Module not found"
**Solution:**
```cmd
pip install -e .
python -c "import multimodal_rag; print('✓ Import successful')"
```

### EasyOCR Installation Issues
**Solution:**
```cmd
pip uninstall easyocr opencv-python -y
pip install easyocr --no-deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless
```

## 📊 Verify Everything Works

```cmd
# 1. Check environment
python -c "from multimodal_rag.utils.config import get_config; c=get_config(); print(f'✓ Config loaded: {c.gemini_model}')"

# 2. Check Docker services
docker-compose ps

# 3. Check Qdrant connection
python -c "from multimodal_rag.retrieval.vector_store import QdrantStore; s=QdrantStore(); print('✓ Qdrant connected')"

# 4. Check embedder
python -c "from multimodal_rag.ingestion.embedder import TextEmbedder; e=TextEmbedder(); print('✓ Embedder ready')"

# 5. Run a test
pytest tests/test_ingestion.py::test_semantic_chunker_initialization -v
```

## 🎓 Learn More

- [Full README](README.md) - Complete documentation
- [Architecture](docs/ARCHITECTURE.md) - System design (to be created)
- [Coding Standards](coding_standars.md) - Development guidelines

## 🆘 Get Help

If you encounter issues:

1. Check logs: `docker-compose logs qdrant redis`
2. Restart services: `docker-compose restart`
3. Rebuild environment: `conda env remove -n multimodal_rag; conda env create -f environment.yml`
4. Check API keys: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Gemini:', os.getenv('GEMINI_API_KEY')[:20] + '...'); print('Langfuse:', os.getenv('LANGFUSE_PUBLIC_KEY')[:15] + '...')"`

---

**Ready to build amazing RAG applications!** 🎉
