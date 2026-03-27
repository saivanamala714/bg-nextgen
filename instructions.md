PROJECT INSTRUCTION: Build a Pure Python FastAPI Backend for PDF Chat RAG
Project Name: pdf-chat-rag-backend
Goal:
Create a backend-only FastAPI application (no frontend/UI code) that answers questions from a single static 1.1 MB PDF with high accuracy, excellent fuzzy word handling, and fast responses. The PDF rarely changes.
Core Requirements:

High answer correctness + minimal hallucination
Strong handling of fuzzy/synonyms/technical terms
Fast response time (low latency)
Store every question + answer + context for future training/fine-tuning
Pre-compute PDF embeddings locally (free)

Tech Stack:

Framework: FastAPI + Uvicorn
Vector DB: Supabase Postgres + pgvector (use supabase Python client)
Local Embeddings (PDF):BAAI/bge-m3 via HuggingFace (1024 dim)
Query Embeddings: OpenAI text-embedding-3-small
LLM: Groq llama-3.3-70b-versatile with streaming
RAG Framework: LangChain (simple chains) or pure code — keep it clean and lightweight
PDF Processing: PyMuPDF or pdfplumber + RecursiveCharacterTextSplitter
Deployment Target: Render.com or Railway (free tier friendly for Python FastAPI)

1. Project Structure
textpdf-chat-rag-backend/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── routers/
│   │   ├── chat.py
│   │   └── ingest.py            # Optional: precomputed upload
│   ├── core/
│   │   ├── config.py            # Settings with Pydantic
│   │   └── supabase_client.py
│   ├── services/
│   │   ├── rag_service.py       # Main RAG logic
│   │   └── embedding_service.py
│   └── models/
│       └── schemas.py           # Pydantic models
├── local_ingest/
│   ├── ingest.py                # One-time local embedding script
│   └── your-document.pdf        # User places PDF here
├── requirements.txt
├── .env
├── Dockerfile                   # For deployment
├── README.md
└── supabase_schema.sql          # SQL to run in Supabase
2. Local Ingestion Script (local_ingest/ingest.py)
Create a script that:

Loads the 1.1 MB PDF
Splits with RecursiveCharacterTextSplitter (chunk_size=800, chunk_overlap=150)
Embeds using BAAI/bge-m3 (normalize embeddings)
Saves records as embeddings_to_upload.json (list of dicts with chunk_text, embedding as list, metadata)

3. Supabase Schema (supabase_schema.sql)
Generate full SQL:

Enable vector extension
pdf_chunks table: id, pdf_name, chunk_text, embedding vector(1024), metadata jsonb, created_at
HNSW cosine index on embedding
GIN full-text index on chunk_text
match_documents RPC function with hybrid search (vector + keyword using RRF or weighted) — return top 8 results with similarity
chat_logs table: id, session_id uuid, question, answer, context_used text, metadata jsonb, created_at, feedback jsonb

4. Environment Variables (.env)
List all required:

SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
GROQ_API_KEY
OPENAI_API_KEY

5. Core Features to Implement
POST /chat

Input: { "question": str, "history": list[dict] optional, "session_id": str optional }
Steps:
Generate query embedding with OpenAI text-embedding-3-small
Call Supabase hybrid match_documents (vector + keyword for fuzzy robustness)
Build strong system prompt: "Answer ONLY using the provided context. If not found, say 'I could not find information about that in the document.' Be concise and factual."
Call Groq with streaming (temperature=0.1, model=llama-3.3-70b-versatile)
Return streaming response (text/event-stream)
After full answer is generated, asynchronously log to chat_logs table (question, full answer, context_used, metadata including latency)


POST /ingest/precomputed

Accepts JSON body (the array from local_ingest)
Inserts into pdf_chunks using service role key
Return success + count

6. Additional Requirements

Use async where possible (async def endpoints)
Proper error handling and logging (use logging module)
Limit history to last 6-8 messages for speed
Keep context clean in prompt (include page numbers from metadata)
Add max_tokens reasonable limit
Include health check endpoint /health
Use Pydantic v2 models for request/response
Make code clean with good comments and type hints
Support streaming properly with StreamingResponse

7. Deployment Notes (in README)

How to run locally: uvicorn app.main:app --reload
Dockerfile for Render/Railway/Fly.io
How to run local ingest → upload via /ingest/precomputed
Recommended deployment: Render.com (free web service) or Railway ($5 credit)

8. Cost Optimization

Local embeddings = $0 recurring for PDF
Hybrid retrieval for better accuracy without extra cost
Low temperature + limited history for faster + cheaper Groq calls
Chat logging is lightweight

Generate the complete project with all files, well-commented code, and production-ready structure.
Start by creating the folder structure and all Python files.

Copy everything above and feed it to your IDE.
After generation, you can:

Run the local ingest.py
Deploy on Render.com (easiest for Python FastAPI — free tier spins down after inactivity but is cheap)
Or Railway / Fly.io