# 🤖 Multimodal RAG App

A production-ready Retrieval-Augmented Generation (RAG) application that processes PDFs with **OpenAI Vision API** to extract text from both regular content and images/diagrams.

## ✨ Features

- 📄 **Smart PDF Processing**: Hybrid extraction that tries traditional text first, only uses Vision API when needed (90% cost savings!)
- 💰 **Cost-Optimized**: Built-in page limits and cost estimation for large documents/books
- 🧠 **General Knowledge Fallback**: ChatGPT-like behavior - answers from your documents OR general knowledge when needed
- 🔍 **Hybrid Search**: Combines dense vector search with semantic matching
- 💬 **Accurate Responses**: Uses OpenAI GPT-4o for high-quality answer generation with citations
- 🌐 **Clean UI**: Built with Streamlit for easy interaction
- ☁️ **Cloud Ready**: Deploy for free on Streamlit Cloud + Qdrant Cloud

### 🚀 NEW: Large PDF Support!

Process **500-page books** for **$0.50** instead of $5!
- ✅ Smart hybrid extraction (text-first, Vision fallback)
- ✅ Automatic page limits to prevent huge costs
- ✅ Cost estimation before processing
- ✅ Process specific page ranges

📖 **See**: [LARGE_PDF_GUIDE.md](LARGE_PDF_GUIDE.md) | [LARGE_PDF_QUICK.md](LARGE_PDF_QUICK.md)

### 🆕 NEW: General Knowledge Mode!

Answer questions even when documents don't have the info - just like ChatGPT!
- ✅ Automatic fallback to general knowledge when retrieval fails
- ✅ Clear UI indicators showing answer source (📚 docs vs 💡 general)
- ✅ Configurable strict mode for compliance use cases

📖 **See**: [GENERAL_KNOWLEDGE_MODE.md](GENERAL_KNOWLEDGE_MODE.md)

## 🚀 Quick Start (Local)

### ⚡ **START HERE**: [STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md) ⭐

Complete walkthrough from zero to deployed app in 30 minutes!

---

### Prerequisites
- Python 3.11+
- OpenAI API key
- Docker (for local Qdrant) OR Qdrant Cloud account

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/multimodal-rag-app.git
cd multimodal-rag-app

# Install dependencies
pip install -r requirements.txt

# Install Poppler (for PDF processing)
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
# Add poppler/bin to PATH
# Linux: sudo apt-get install poppler-utils
# Mac: brew install poppler

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Run Locally

```bash
# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run Streamlit app
streamlit run streamlit_lazy.py
```

Visit `http://localhost:8501`

## ☁️ Deploy for FREE

**Total Cost**: $0/month + OpenAI usage (~$1-10/month for light use)

### Quick Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/multimodal-rag-app.git
   git push -u origin main
   ```

2. **Set up Qdrant Cloud** (Free tier: 1GB)
   - Sign up: https://cloud.qdrant.io/
   - Create cluster (free tier)
   - Copy URL and API key

3. **Deploy on Streamlit Cloud** (Free tier: unlimited public apps)
   - Go to: https://share.streamlit.io/
   - Click "New app"
   - Select your repo
   - Configure secrets (see below)
   - Deploy!

4. **Configure Secrets** in Streamlit Cloud:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   OPENAI_MODEL = "gpt-4o-mini"
   OPENAI_VISION_MODEL = "gpt-4o-mini"
   OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
   QDRANT_URL = "https://xxxxx.aws.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "your-qdrant-key"
   QDRANT_COLLECTION_NAME = "multimodal_rag"
   ```

📖 **Detailed Guide**: See [FREE_HOSTING_GUIDE.md](FREE_HOSTING_GUIDE.md)
✅ **Checklist**: See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

## 📋 How It Works

1. **Upload PDF**: User uploads a PDF document
2. **Vision Processing**: Each PDF page is converted to an image and processed with GPT-4 Vision
3. **Text Extraction**: Vision API extracts ALL text (regular + text in images/charts)
4. **Chunking**: Text is split into semantic chunks
5. **Embedding**: Chunks are embedded using OpenAI embeddings
6. **Storage**: Vectors stored in Qdrant vector database
7. **Query**: User asks a question
8. **Retrieval**: System finds relevant chunks using hybrid search
9. **Generation**: GPT-4o generates an answer with citations

## 🔑 Why Use OpenAI Vision for PDFs?

Traditional OCR often **misses text** in:
- ❌ Scanned documents
- ❌ Charts and diagrams
- ❌ Tables with complex layouts
- ❌ Images with embedded text
- ❌ Handwritten notes (in some cases)

**OpenAI Vision API** solves this by:
- ✅ Understanding context and layout
- ✅ Extracting text from any visual element
- ✅ Handling complex documents
- ✅ Better accuracy than traditional OCR

## 💰 Cost Estimate

### Free Tier Components:
- Streamlit Cloud: **Free** (unlimited public apps)
- Qdrant Cloud: **Free** (1GB storage, ~100k vectors)
- GitHub: **Free** (public repositories)

### Pay-as-you-go (OpenAI):
- **GPT-4o**: $2.50/1M input tokens, $10/1M output tokens
- **GPT-4o-mini**: 60% cheaper than GPT-4o
- **Vision**: ~$0.01 per image
- **Embeddings**: ~$0.02/1M tokens

**Example Usage**:
```
10-page PDF processing:
  - Vision (10 images): $0.10
  - Embeddings: $0.001
  Total: ~$0.10 one-time

100 queries/month:
  - GPT-4o-mini: ~$0.50-1.00
  Total: ~$0.50-1.00/month

TOTAL: ~$1-5/month for moderate usage
```

## 📁 Project Structure

```
MultimodalRag/
├── src/multimodal_rag/          # Core application
│   ├── ingestion/               # PDF & image processing
│   │   ├── pdf_processor.py    # Vision-based PDF processing
│   │   ├── image_processor.py  # Vision-based image processing
│   │   └── embedder.py          # OpenAI embeddings
│   ├── retrieval/               # Vector search
│   ├── generation/              # Response generation
│   └── utils/                   # Configuration & logging
├── streamlit_lazy.py            # Streamlit UI
├── requirements.txt             # Dependencies
├── packages.txt                 # System packages (poppler)
├── .streamlit/config.toml       # Streamlit config
├── FREE_HOSTING_GUIDE.md        # Deployment guide
└── DEPLOYMENT_CHECKLIST.md      # Deployment checklist
```

## 🔧 Configuration

Edit `.env` file:

```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini  # Use mini for lower costs

# Qdrant - Local
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Qdrant - Cloud (for production)
# QDRANT_URL=https://xxxxx.aws.cloud.qdrant.io:6333
# QDRANT_API_KEY=your-key

# Settings
CHUNK_SIZE=512
TOP_K_RETRIEVAL=10
```

## 🛠️ Development

### Run Tests
```bash
pytest tests/
```

### Check Code Quality
```bash
# Format
black src/

# Lint
pylint src/
```

## 📚 Documentation

**🌟 START HERE**:
- **[STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md)** - ⭐ Complete setup guide (0 to deployed in 30 min)

**New Features**:
- **[GENERAL_KNOWLEDGE_MODE.md](GENERAL_KNOWLEDGE_MODE.md)** - 🆕 ChatGPT-like fallback mode guide
- **[SMART_RETRIEVAL_GUIDE.md](SMART_RETRIEVAL_GUIDE.md)** - 🆕 Confidence-based smart retrieval

**Large PDF Processing**:
- [LARGE_PDF_GUIDE.md](LARGE_PDF_GUIDE.md) - Complete guide for processing large PDFs cost-effectively
- [LARGE_PDF_QUICK.md](LARGE_PDF_QUICK.md) - Quick reference for large PDF settings

**Deployment**:
- [QDRANT_CLOUD_SETUP.md](QDRANT_CLOUD_SETUP.md) - 🆕 Quick Qdrant Cloud setup (auto-creates collections!)
- [FREE_HOSTING_GUIDE.md](FREE_HOSTING_GUIDE.md) - Complete free hosting guide
- [DEPLOY_NOW.md](DEPLOY_NOW.md) - Quick 5-minute deployment guide
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Deployment checklist

**Other**:
- [OPENAI_MIGRATION_GUIDE.md](OPENAI_MIGRATION_GUIDE.md) - Migration from Gemini to OpenAI

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- OpenAI for Vision and GPT-4 APIs
- Qdrant for vector database
- Streamlit for UI framework

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/multimodal-rag-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/multimodal-rag-app/discussions)

---

**Live Demo**: [Your deployed app URL]

**Made with ❤️ using OpenAI, Qdrant, and Streamlit**
