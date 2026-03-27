from supabase import create_client, Client
from typing import Dict, Any, List, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Supabase client wrapper with optimized operations."""
    
    def __init__(self):
        if settings.test_mode:
            logger.warning("Running in test mode - Supabase client disabled")
            self.client = None
            return
            
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Supabase connection on initialization."""
        if settings.test_mode:
            return
            
        try:
            # Try a simple health check instead of querying a specific table
            response = self.client.rpc('get_server_version', {}).execute()
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            # Don't fail startup, just log a warning
            logger.warning(f"Could not test Supabase connection: {e}")
            logger.info("Supabase client initialized but connection not verified")
    
    async def search_documents(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = settings.max_context_chunks
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using vector similarity and keyword matching.
        Calls the match_documents RPC function in Supabase.
        """
        if settings.test_mode:
            logger.warning("Test mode: Returning mock search results")
            return [{
                'chunk_text': 'This is a mock document chunk for testing.',
                'similarity': 0.8,
                'metadata': {'page_number': 1},
                'pdf_name': 'test.pdf'
            }]
        
        try:
            logger.info(f"Searching documents with query: '{query_text}'")
            logger.info(f"Query embedding dimensions: {len(query_embedding)}")
            logger.info(f"First 5 embedding values: {query_embedding[:5]}")
            logger.info(f"Last 5 embedding values: {query_embedding[-5:]}")
            
            # Prepare the RPC call parameters
            rpc_params = {
                'query_embedding': query_embedding,
                'query_text': query_text,
                'match_count': limit,
                'similarity_threshold': 0.3  # Force lower threshold for testing
            }
            
            logger.info(f"RPC call parameters: {rpc_params}")
            
            response = self.client.rpc(
                'match_documents',
                rpc_params
            ).execute()
            
            results = response.data if response.data else []
            logger.info(f"Found {len(results)} matching documents")
            
            if results:
                logger.info(f"First result similarity: {results[0].get('similarity', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []   
    async def log_chat_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        context_used: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log chat interaction to the chat_logs table.
        Returns the ID of the inserted record.
        """
        if settings.test_mode:
            logger.warning("Test mode: Mock logging chat interaction")
            return "mock-log-id"
            
        try:
            chat_data = {
                'session_id': session_id,
                'question': question,
                'answer': answer,
                'context_used': context_used,
                'metadata': metadata or {}
            }
            
            response = self.client.table('chat_logs').insert(chat_data).execute()
            
            if response.data:
                log_id = response.data[0]['id']
                logger.info(f"Logged chat interaction with ID: {log_id}")
                return log_id
            else:
                logger.error("Failed to log chat interaction")
                raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error logging chat interaction: {e}")
            raise
    
    async def insert_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Insert PDF chunks into the database.
        Returns the number of successfully inserted chunks.
        """
        if settings.test_mode:
            logger.warning(f"Test mode: Mock inserting {len(chunks)} chunks")
            return len(chunks)
            
        try:
            # Process in batches to avoid payload size limits
            batch_size = 100
            inserted_count = 0
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                response = self.client.table('pdf_chunks').insert(batch).execute()
                
                if response.data:
                    inserted_count += len(response.data)
                    logger.info(f"Inserted batch {i//batch_size + 1}: {len(response.data)} chunks")
                else:
                    logger.error(f"Failed to insert batch {i//batch_size + 1}")
            
            logger.info(f"Successfully inserted {inserted_count} chunks total")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            raise
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = settings.max_history_messages
    ) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        if settings.test_mode:
            logger.warning("Test mode: Returning empty chat history")
            return []
            
        try:
            response = (self.client.table('chat_logs')
                       .select('question, answer, created_at')
                       .eq('session_id', session_id)
                       .order('created_at', desc=False)
                       .limit(limit)
                       .execute())
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []


# Global Supabase client instance
supabase_client = SupabaseClient()
