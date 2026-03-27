-- PDF Chat RAG Backend - Supabase Schema (Transaction-Safe)
-- 
-- This SQL script sets up the database schema for the PDF Chat RAG application.
-- Run this script in your Supabase SQL editor to create the necessary tables and functions.
-- 
-- Features:
-- - Vector extension for embeddings
-- - PDF chunks table with HNSW vector index
-- - Full-text search capabilities
-- - Hybrid search function (vector + keyword)
-- - Chat logging with session management
-- - Optimized indexes for performance

-- Enable necessary extensions (safe for transactions)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create PDF chunks table
CREATE TABLE IF NOT EXISTS pdf_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_name TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1024) NOT NULL,  -- BGE-M3 embedding dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT pdf_chunks_pdf_name_check CHECK (length(pdf_name) > 0),
    CONSTRAINT pdf_chunks_chunk_text_check CHECK (length(chunk_text) > 0)
);

-- Create chat logs table
CREATE TABLE IF NOT EXISTS chat_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context_used TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    feedback JSONB DEFAULT '{}',
    
    -- Constraints
    CONSTRAINT chat_logs_question_check CHECK (length(question) > 0),
    CONSTRAINT chat_logs_answer_check CHECK (length(answer) > 0)
);

-- Create HNSW vector index for similarity search
-- This index is optimized for cosine similarity search
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding_hnsw 
ON pdf_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create additional performance indexes
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_pdf_name ON pdf_chunks(pdf_name);
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_created_at ON pdf_chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON chat_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON chat_logs(created_at);

-- Create full-text search index for keyword search
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_chunk_text_gin 
ON pdf_chunks 
USING gin(chunk_text gin_trgm_ops);

-- ===============================================================
-- Hybrid Search Function
-- ===============================================================

-- Create the match_documents function for hybrid search
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    query_text TEXT DEFAULT '',
    match_count INTEGER DEFAULT 8,
    similarity_threshold REAL DEFAULT 0.7
)
RETURNS TABLE (
    chunk_text TEXT,
    similarity DOUBLE PRECISION,
    metadata JSONB,
    pdf_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pdf_chunks.chunk_text,
        1 - (pdf_chunks.embedding <=> query_embedding) AS similarity,
        pdf_chunks.metadata,
        pdf_chunks.pdf_name
    FROM pdf_chunks
    WHERE 1 - (pdf_chunks.embedding <=> query_embedding) > similarity_threshold
    ORDER BY pdf_chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ===============================================================
-- Row Level Security (RLS)
-- ===============================================================

-- Enable RLS on tables
ALTER TABLE pdf_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_logs ENABLE ROW LEVEL SECURITY;

-- Create policies for pdf_chunks (read-only for public)
CREATE POLICY "Enable read access for all users" ON pdf_chunks
    FOR SELECT USING (true);

CREATE POLICY "Enable insert access for all users" ON pdf_chunks
    FOR INSERT WITH CHECK (true);

-- Create policies for chat_logs (read/write for public)
CREATE POLICY "Enable read access for all users" ON chat_logs
    FOR SELECT USING (true);

CREATE POLICY "Enable insert access for all users" ON chat_logs
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable update access for all users" ON chat_logs
    FOR UPDATE USING (true);

-- ===============================================================
-- Sample Data Test (Optional - Remove in Production)
-- ===============================================================

-- Insert a test chunk to verify everything works
INSERT INTO pdf_chunks (pdf_name, chunk_text, embedding, metadata)
VALUES (
    'test.pdf',
    'This is a test document chunk for verification.',
    array_fill(0.1, ARRAY[1024])::vector(1024),
    '{"page_number": 1, "test": true}')
ON CONFLICT DO NOTHING;

-- ===============================================================
-- Verification Queries (Uncomment to test)
-- ===============================================================

-- Test the match_documents function
-- SELECT * FROM match_documents(
--     array_fill(0.1, ARRAY[1024])::vector(1024),
--     'test document',
--     5,
--     0.5
-- );

-- Check table structures
-- \d pdf_chunks
-- \d chat_logs

-- Count records
-- SELECT COUNT(*) FROM pdf_chunks;
-- SELECT COUNT(*) FROM chat_logs;

-- Check if vector extension is enabled
-- SELECT extname FROM pg_extension WHERE extname = 'vector';
