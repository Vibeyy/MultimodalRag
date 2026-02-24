# API Documentation

## FastAPI Endpoints

Base URL: `http://localhost:8000/api/v1`

### Query Endpoint

**POST** `/query`

Query the RAG system with automatic retrieval and generation.

**Request Body:**
```json
{
  "query": "What is artificial intelligence?",
  "top_k": 5,
  "expand_query": false,
  "check_hallucination": false,
  "filters": null
}
```

**Response:**
```json
{
  "answer": "Artificial Intelligence (AI) is the simulation...",
  "citations": [
    {
      "source_file": "sample.txt",
      "page_num": 1,
      "chunk_index": 0
    }
  ],
  "retrieved_chunks": 5,
  "hallucination_score": null,
  "is_hallucinated": null,
  "metadata": {}
}
```

### Ingest Endpoint

**POST** `/ingest`

Ingest documents into the vector store.

**Request Body:**
```json
{
  "file_paths": ["data/raw/report.pdf", "data/raw/image.png"],
  "tags": ["research", "2024"],
  "recreate_collection": false
}
```

**Response:**
```json
{
  "document_count": 2,
  "total_chunks": 150,
  "text_chunks": 75,
  "image_chunks": 75,
  "processing_time_ms": 45000.5,
  "errors": []
}
```

### Search Endpoint

**POST** `/search`

Search for chunks without generation (retrieval only).

**Request Body:**
```json
{
  "query": "artificial intelligence",
  "limit": 10,
  "filters": {"tags": ["research"]}
}
```

**Response:**
```json
{
  "chunks": [
    {
      "id": "doc123_chunk_0",
      "text": "Artificial Intelligence (AI) is...",
      "score": 0.95,
      "chunk_type": "text",
      "metadata": {
        "source_file": "sample.txt",
        "page_num": 1
      }
    }
  ],
  "total_found": 10
}
```

### Status Endpoint

**GET** `/status`

Get system status and statistics.

**Response:**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "collection_name": "multimodal_rag",
  "total_documents": 0,
  "total_chunks": 150,
  "vector_config": {
    "text": {
      "size": 1024,
      "distance": "Cosine"
    },
    "image": {
      "size": 768,
      "distance": "Cosine"
    }
  }
}
```

### Delete Collection Endpoint

**DELETE** `/collection`

Delete the vector collection (⚠️ use with caution!).

**Response:**
```json
{
  "message": "Collection deleted successfully"
}
```

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message here"
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad Request
- `500`: Internal Server Error

## Authentication

Currently no authentication required. Add API keys or OAuth2 for production deployment.

## Rate Limiting

Gemini API has 60 requests/minute free tier limit. The system includes automatic rate limiting.

## Examples

### Python Client

```python
import requests

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "What is AI?",
        "top_k": 5,
        "expand_query": True,
    }
)
result = response.json()
print(result["answer"])
```

### cURL

```bash
# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "top_k": 5}'

# Status
curl http://localhost:8000/api/v1/status
```

### JavaScript

```javascript
// Query
const response = await fetch('http://localhost:8000/api/v1/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: 'What is AI?',
    top_k: 5
  })
});
const result = await response.json();
console.log(result.answer);
```
