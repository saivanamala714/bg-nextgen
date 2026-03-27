# PDF Chat RAG Backend

A production-ready FastAPI backend for PDF chat with Retrieval-Augmented Generation (RAG). This application enables users to ask questions about PDF documents with high accuracy, excellent fuzzy word handling, and fast responses.

## 🚀 Features

- **High Accuracy RAG**: Hybrid search combining vector similarity and keyword matching
- **Fast Responses**: Optimized embeddings and streaming responses
- **Cost Effective**: Local PDF embeddings with OpenAI query embeddings
- **Production Ready**: Async endpoints, error handling, logging, and Docker support
- **Scalable**: Supabase Postgres with pgvector for vector storage
- **Chat Logging**: Store all interactions for future training/fine-tuning

## 🏗️ Architecture

```
pdf-chat-rag-backend/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── routers/
│   │   ├── chat.py             # Chat endpoints
│   │   └── ingest.py           # Ingestion endpoints
│   ├── core/
│   │   ├── config.py           # Settings management
│   │   └── supabase_client.py  # Database client
│   ├── services/
│   │   ├── rag_service.py      # Main RAG logic
│   │   └── embedding_service.py # Embedding generation
│   └── models/
│       └── schemas.py          # Pydantic models
├── local_ingest/
│   ├── ingest.py               # Local PDF processing
│   └── README.md               # Ingestion instructions
├── requirements.txt
├── Dockerfile
├── .env.example
├── README.md
└── supabase_schema.sql         # Database schema
```

## 🛠️ Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Vector DB**: Supabase Postgres + pgvector
- **Local Embeddings**: BAAI/bge-m3 via HuggingFace (1024 dim)
- **Query Embeddings**: OpenAI text-embedding-3-small
- **LLM**: Groq llama-3.3-70b-versatile with streaming
- **PDF Processing**: PyMuPDF + RecursiveCharacterTextSplitter
- **Deployment**: Docker + Render.com/Railway ready

## 📋 Prerequisites

- Python 3.11+
- Supabase account (free tier works)
- OpenAI API key
- Groq API key
- A PDF file to process

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd pdf-chat-rag-backend

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` with your API keys:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Setup Supabase Database

1. Create a new Supabase project
2. Go to SQL Editor and run the `supabase_schema.sql` script
3. Enable the vector extension (included in schema)
4. Get your project URL and service role key from Settings > API

### 4. Process Your PDF

```bash
# Place your PDF in local_ingest/
cp /path/to/your/document.pdf local_ingest/your-document.pdf

# Run local ingestion
python local_ingest/ingest.py
```

This creates `local_ingest/embeddings_to_upload.json` with processed chunks.

### 5. Upload Embeddings

Start the server first:

```bash
uvicorn app.main:app --reload
```

Then upload the embeddings:

```bash
# Method 1: Upload JSON file
curl -X POST http://localhost:8000/ingest/json-file \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@local_ingest/embeddings_to_upload.json'

# Method 2: Direct JSON upload
curl -X POST http://localhost:8000/ingest/precomputed \
     -H 'Content-Type: application/json' \
     -d @local_ingest/embeddings_to_upload.json
```

### 6. Start Chatting

```bash
# Regular chat
curl -X POST http://localhost:8000/chat \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is this document about?"}'

# Streaming chat
curl -X POST http://localhost:8000/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"question": "Summarize the key points"}'
```

## 📡 API Endpoints

### Chat Endpoints

- `POST /chat` - Regular chat response
- `POST /chat/stream` - Streaming chat (SSE)
- `GET /chat/history/{session_id}` - Get chat history
- `DELETE /chat/history/{session_id}` - Clear chat history

### Ingestion Endpoints

- `POST /ingest/precomputed` - Upload precomputed embeddings
- `POST /ingest/json-file` - Upload embeddings via file
- `GET /ingest/status` - Get ingestion statistics
- `DELETE /ingest/clear` - Clear all data (⚠️ destructive)

### Utility Endpoints

- `GET /` - API information
- `GET /health` - Health check with service status
- `GET /docs` - Interactive API documentation

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | Required |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Required |
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `MAX_CONTEXT_CHUNKS` | Max document chunks in context | 8 |
| `TEMPERATURE` | LLM temperature (0-1) | 0.1 |
| `CHUNK_SIZE` | PDF chunk size | 800 |
| `CHUNK_OVERLAP` | PDF chunk overlap | 150 |

### RAG Parameters

- **Embedding Models**: BGE-M3 (local) + text-embedding-3-small (queries)
- **Chunk Size**: 800 characters with 150 overlap
- **Vector Dimension**: 1024 (BGE-M3)
- **Similarity Threshold**: 0.7 (configurable)
- **Hybrid Search**: 70% vector + 30% keyword weighting

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t pdf-chat-rag-backend .

# Run container
docker run -p 8000:8000 --env-file .env pdf-chat-rag-backend
```

### Deploy to Render.com

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set build command: `docker build -t pdf-chat-rag-backend .`
4. Set start command: `docker run -p $PORT:$PORT --env-file .env pdf-chat-rag-backend`
5. Add environment variables in Render dashboard

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Returns service status including:
- Supabase connectivity
- Embedding service status
- OpenAI/Groq API status
- Overall system health

### Logging

The application uses structured logging with configurable levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages only

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_chat.py
```

## 🔒 Security

- Row Level Security (RLS) enabled on Supabase tables
- Service role keys used for admin operations
- Input validation with Pydantic models
- CORS configuration for production
- No sensitive data in logs

## 💰 Cost Optimization

- **Local Embeddings**: $0 recurring cost for PDF processing
- **Query Embeddings**: ~$0.00002 per query (OpenAI)
- **LLM Calls**: ~$0.0005 per 1K tokens (Groq)
- **Database**: Free tier Supabase supports ~500MB data

Estimated monthly cost for moderate usage: <$5

## 🚨 Troubleshooting

### Common Issues

1. **Embedding Dimension Mismatch**
   - Ensure BGE-M3 model loads correctly
   - Check vector dimension in Supabase schema (1024)

2. **Supabase Connection Failed**
   - Verify URL and service role key
   - Check network connectivity
   - Ensure schema is properly installed

3. **Memory Issues with Large PDFs**
   - Process in smaller batches
   - Increase chunk size to reduce number of chunks
   - Use machine with more RAM

4. **Slow Response Times**
   - Check Supabase HNSW index is created
   - Reduce `MAX_CONTEXT_CHUNKS`
   - Optimize PDF chunking strategy

### Debug Mode

Enable debug logging:

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Verify all environment variables are set
4. Ensure Supabase schema is properly installed
5. Test API endpoints using the `/docs` interface

## 🔄 Updates and Maintenance

- Regularly update dependencies: `pip install -r requirements.txt --upgrade`
- Monitor Supabase usage and upgrade if needed
- Backup chat logs periodically for training data
- Update embedding models as improved versions become available

---

**Built with ❤️ using FastAPI, Supabase, and modern RAG techniques**
# bg-nextgen
