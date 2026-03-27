# Quick Setup Guide

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Set Up Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Required environment variables:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Your Supabase service role key
- `GROQ_API_KEY` - Your Groq API key
- `OPENAI_API_KEY` - Your OpenAI API key

## 3. Set Up Supabase Database
Run the SQL commands in `supabase_schema.sql` in your Supabase SQL editor.

## 4. Process Your PDF
1. Place your PDF file in `local_ingest/your-document.pdf`
2. Run the ingestion script:
   ```bash
   cd local_ingest
   python ingest.py
   ```
3. Upload the generated embeddings via the API.

## 5. Run the Application
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## 6. Test the API
- Health check: `GET http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- Chat endpoint: `POST http://localhost:8000/chat`

## Troubleshooting
- If you get import errors, make sure all dependencies are installed
- Check the logs for detailed error messages
- Verify your environment variables are set correctly
