-- PDF Chat RAG Backend - Supabase Schema
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

-- Enable necessary extensions
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
    session_id UUID NOT NULL,
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

-- Create GIN index for full-text search on chunk text
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_chunk_text_gin 
ON pdf_chunks 
USING gin (chunk_text gin_trgm_ops);

-- Create GIN index on metadata for JSONB queries
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_metadata_gin 
ON pdf_chunks 
USING gin (metadata);

-- Create index on pdf_name for document filtering
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_pdf_name 
ON pdf_chunks (pdf_name);

-- Create index on session_id for chat logs
CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id 
ON chat_logs (session_id);

-- Create index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at 
ON chat_logs (created_at DESC);

-- Create hybrid search function
-- This function combines vector similarity and keyword matching
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    query_text TEXT DEFAULT '',
    match_count INTEGER DEFAULT 8,
    similarity_threshold FLOAT DEFAULT 0.7,
    pdf_name_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    pdf_name TEXT,
    chunk_text TEXT,
    embedding vector(1024),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    similarity FLOAT,
    text_rank FLOAT
) AS $$
DECLARE
    vector_weight FLOAT := 0.7;  -- Weight for vector similarity
    text_weight FLOAT := 0.3;    -- Weight for text matching
BEGIN
    RETURN QUERY
    WITH vector_search AS (
        SELECT 
            pc.id,
            pc.pdf_name,
            pc.chunk_text,
            pc.embedding,
            pc.metadata,
            pc.created_at,
            1 - (pc.embedding <=> query_embedding) as vector_similarity
        FROM pdf_chunks pc
        WHERE 
            (pdf_name_filter IS NULL OR pc.pdf_name = pdf_name_filter)
            AND (1 - (pc.embedding <=> query_embedding)) >= similarity_threshold
        ORDER BY pc.embedding <=> query_embedding
        LIMIT match_count * 2  -- Get more candidates for better hybrid results
    ),
    text_search AS (
        SELECT 
            pc.id,
            pc.pdf_name,
            pc.chunk_text,
            pc.embedding,
            pc.metadata,
            pc.created_at,
            CASE 
                WHEN query_text != '' THEN 
                    ts_rank_cd(
                        to_tsvector('english', pc.chunk_text),
                        plainto_tsquery('english', query_text)
                    )
                ELSE 0
            END as text_similarity
        FROM pdf_chunks pc
        WHERE 
            (pdf_name_filter IS NULL OR pc.pdf_name = pdf_name_filter)
            AND (query_text = '' OR pc.chunk_text % query_text)
        ORDER BY text_similarity DESC
        LIMIT match_count * 2
    ),
    combined_results AS (
        SELECT 
            COALESCE(vs.id, ts.id) as id,
            COALESCE(vs.pdf_name, ts.pdf_name) as pdf_name,
            COALESCE(vs.chunk_text, ts.chunk_text) as chunk_text,
            COALESCE(vs.embedding, ts.embedding) as embedding,
            COALESCE(vs.metadata, ts.metadata) as metadata,
            COALESCE(vs.created_at, ts.created_at) as created_at,
            COALESCE(vs.vector_similarity, 0) as vector_similarity,
            COALESCE(ts.text_similarity, 0) as text_similarity
        FROM vector_search vs
        FULL OUTER JOIN text_search ts ON vs.id = ts.id
    )
    SELECT 
        id,
        pdf_name,
        chunk_text,
        embedding,
        metadata,
        created_at,
        vector_similarity as similarity,
        text_similarity as text_rank
    FROM combined_results
    WHERE 
        (vector_similarity > 0 OR text_similarity > 0)
    ORDER BY 
        (vector_weight * vector_similarity + text_weight * text_similarity) DESC,
        vector_similarity DESC,
        text_similarity DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for chat statistics
CREATE OR REPLACE FUNCTION get_chat_statistics(
    session_id_param UUID DEFAULT NULL
)
RETURNS TABLE (
    total_sessions BIGINT,
    total_questions BIGINT,
    avg_response_time FLOAT,
    most_recent_chat TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(*) as total_questions,
        AVG((metadata->>'response_time_seconds')::FLOAT) as avg_response_time,
        MAX(created_at) as most_recent_chat
    FROM chat_logs
    WHERE 
        (session_id_param IS NULL OR session_id = session_id_param);
END;
$$ LANGUAGE plpgsql;

-- Create function for document statistics
CREATE OR REPLACE FUNCTION get_document_statistics()
RETURNS TABLE (
    pdf_name TEXT,
    chunk_count BIGINT,
    total_characters BIGINT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pc.pdf_name,
        COUNT(*) as chunk_count,
        SUM(length(pc.chunk_text)) as total_characters,
        MIN(pc.created_at) as created_at
    FROM pdf_chunks pc
    GROUP BY pc.pdf_name
    ORDER BY chunk_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Enable Row Level Security (RLS)
ALTER TABLE pdf_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_logs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- For pdf_chunks: Allow read access to everyone, write access to service role only
CREATE POLICY "Enable read access for all users on pdf_chunks"
    ON pdf_chunks FOR SELECT
    USING (true);

CREATE POLICY "Enable insert for service role on pdf_chunks"
    ON pdf_chunks FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Enable update for service role on pdf_chunks"
    ON pdf_chunks FOR UPDATE
    USING (true);

CREATE POLICY "Enable delete for service role on pdf_chunks"
    ON pdf_chunks FOR DELETE
    USING (true);

-- For chat_logs: Allow read/write access based on session, full access to service role
CREATE POLICY "Enable read access for own sessions on chat_logs"
    ON chat_logs FOR SELECT
    USING (true);  -- In production, you might want to restrict this

CREATE POLICY "Enable insert for all users on chat_logs"
    ON chat_logs FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Enable update for service role on chat_logs"
    ON chat_logs FOR UPDATE
    USING (true);

CREATE POLICY "Enable delete for service role on chat_logs"
    ON chat_logs FOR DELETE
    USING (true);

-- Create updated_at trigger (optional but useful)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add updated_at columns if you want them
-- ALTER TABLE pdf_chunks ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
-- ALTER TABLE chat_logs ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Create triggers for updated_at (if columns exist)
-- CREATE TRIGGER update_pdf_chunks_updated_at 
--     BEFORE UPDATE ON pdf_chunks 
--     FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- CREATE TRIGGER update_chat_logs_updated_at 
--     BEFORE UPDATE ON chat_logs 
--     FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON pdf_chunks TO authenticated;
GRANT ALL ON chat_logs TO authenticated;
GRANT SELECT ON pdf_chunks TO anon;
GRANT SELECT ON chat_logs TO anon;

-- Create view for easy access to recent chats
CREATE OR REPLACE VIEW recent_chats AS
SELECT 
    session_id,
    question,
    answer,
    created_at,
    (metadata->>'response_time_seconds')::FLOAT as response_time_seconds
FROM chat_logs
ORDER BY created_at DESC
LIMIT 100;

-- Create view for document overview
CREATE OR REPLACE VIEW document_overview AS
SELECT 
    pdf_name,
    COUNT(*) as chunk_count,
    SUM(length(chunk_text)) as total_characters,
    AVG(length(chunk_text)) as avg_chunk_length,
    MIN(created_at) as first_ingested,
    MAX(created_at) as last_ingested
FROM pdf_chunks
GROUP BY pdf_name
ORDER BY chunk_count DESC;

-- Add comments for documentation
COMMENT ON TABLE pdf_chunks IS 'Stores PDF text chunks with vector embeddings for semantic search';
COMMENT ON TABLE chat_logs IS 'Stores chat interactions for analytics and training';
COMMENT ON COLUMN pdf_chunks.embedding IS 'Vector embedding using BGE-M3 model (1024 dimensions)';
COMMENT ON COLUMN chat_logs.session_id IS 'UUID to group conversations by session';
COMMENT ON COLUMN chat_logs.context_used IS 'The document context used to generate the answer';
COMMENT ON COLUMN chat_logs.feedback IS 'User feedback and ratings for the response';
COMMENT ON FUNCTION match_documents IS 'Hybrid search combining vector similarity and text matching';

-- Performance optimization settings
-- These settings help with vector search performance
ALTER SYSTEM SET hnsw.ef_search = 64;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Create sample data (optional - for testing)
-- This is commented out but can be useful for initial testing
/*
INSERT INTO pdf_chunks (pdf_name, chunk_text, embedding, metadata) VALUES
('sample.pdf', 'This is a sample chunk of text for testing purposes.', 
 '[0.1,0.2,0.3]'::vector(1024), 
 '{"page_number": 1, "chunk_index": 0}'),
('sample.pdf', 'Another sample chunk with different content.', 
 '[0.4,0.5,0.6]'::vector(1024), 
 '{"page_number": 1, "chunk_index": 1}');
*/

COMMIT;

-- Verification queries
-- Run these to verify the setup:

-- 1. Check tables exist
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('pdf_chunks', 'chat_logs');

-- 2. Check indexes exist
-- SELECT indexname FROM pg_indexes WHERE tablename IN ('pdf_chunks', 'chat_logs');

-- 3. Test the match_documents function (requires actual data)
-- SELECT * FROM match_documents('[0.1,0.2,0.3]'::vector(1024), 'sample query', 5, 0.5);

-- 4. Check RLS policies
-- SELECT policyname, tablename FROM pg_policies WHERE tablename IN ('pdf_chunks', 'chat_logs');
