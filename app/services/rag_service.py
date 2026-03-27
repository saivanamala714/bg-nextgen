from typing import List, Dict, Any, AsyncGenerator, Optional
import openai
import json
import logging
import time
from datetime import datetime

from ..core.config import settings
from ..core.supabase_client import supabase_client
from ..services.embedding_service import embedding_service
from ..models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service for handling chat with PDF documents."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.llm_client = None
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client with fallback options."""
        try:
            # Try Groq first
            if settings.groq_api_key and settings.groq_api_key != "your_groq_api_key":
                from groq import Groq
                self.llm_client = Groq(api_key=settings.groq_api_key)
                logger.info("Using Groq LLM client")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
        
        try:
            # Fallback to OpenAI
            if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key":
                self.llm_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("Using OpenAI LLM client as fallback")
                return
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        logger.error("No LLM client available - will return raw context")
    
    def _build_system_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Build system prompt with context and instructions."""
        context_text = "\n\n".join([
            f"[Page {chunk.get('metadata', {}).get('page_number', 'Unknown')}]: {chunk['chunk_text']}"
            for chunk in context_chunks
        ])
        
        system_prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided document context.

CONTEXT:
{context_text}

INSTRUCTIONS:
1. Answer the user's question using ONLY the information from the context above
2. If the information is not found in the context, say "I could not find information about that in the document."
3. Be concise, factual, and accurate
4. Include page numbers when referencing specific information
5. Do not make up information or go beyond the provided context
6. If you're unsure about something, acknowledge the limitation

Question: {{user_question}}"""

        return system_prompt
    
    def _format_chat_history(self, history: List[ChatMessage]) -> List[Dict[str, str]]:
        """Format chat history for the LLM."""
        formatted_history = []
        
        # Limit history to prevent context overflow
        limited_history = history[-settings.max_history_messages:] if history else []
        
        for message in limited_history:
            formatted_history.append({
                "role": message.role,
                "content": message.content
            })
        
        return formatted_history
    
    async def retrieve_relevant_context(
        self,
        query_embedding: List[float],
        query_text: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks using hybrid search.
        """
        try:
            # Handle dimension mismatch between OpenAI (1536) and local model (1024)
            # For now, we'll use the query_text for keyword matching and adjust the search
            logger.info(f"Retrieving context for query: {query_text[:100]}...")
            
            # Search using Supabase hybrid search
            context_chunks = await supabase_client.search_documents(
                query_embedding=query_embedding,
                query_text=query_text,
                limit=settings.max_context_chunks
            )
            
            if not context_chunks:
                logger.warning("No relevant context found")
                return []
            
            logger.info(f"Retrieved {len(context_chunks)} context chunks")
            return context_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    async def generate_response_stream(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response using Groq LLM.
        """
        try:
            if not context_chunks:
                yield "I could not find relevant information in the document to answer your question."
                return
            
            # Build system prompt
            system_prompt = self._build_system_prompt(context_chunks)
            
            # Format chat history
            formatted_history = self._format_chat_history(history or [])
            
            # Prepare messages for Groq
            messages = [
                {"role": "system", "content": system_prompt.replace("{user_question}", question)}
            ]
            messages.extend(formatted_history)
            
            logger.info(f"Generating response with {len(messages)} messages")
            
            # Generate streaming response
            stream = self.groq_client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield "I apologize, but I encountered an error while generating a response. Please try again."
    
    async def generate_response(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Generate non-streaming response (for internal use).
        """
        try:
            response_parts = []
            async for chunk in self.generate_response_stream(question, context_chunks, history):
                response_parts.append(chunk)
            
            return "".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    async def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with retrieved context.
        """
        try:
            if not context_chunks:
                logger.warning("No context chunks provided for answer generation")
                return {
                    "answer": "I could not find relevant information in the document to answer your question.",
                    "sources": [],
                    "context_used": ""
                }
            
            logger.info(f"Generating answer for question: '{question}'")
            logger.info(f"Using {len(context_chunks)} context chunks")
            
            # Check if LLM client is available
            if not self.llm_client:
                logger.warning("No LLM client available - returning raw context")
                context_text = "\n\n---\n\n".join([
                    f"Page {chunk.get('metadata', {}).get('page_number', '?')}: {chunk['chunk_text']}"
                    for chunk in context_chunks
                ])
                return {
                    "answer": f"Found {len(context_chunks)} relevant chunks from the document:\n\n{context_text}",
                    "sources": [
                        {
                            "chunk_text": chunk["chunk_text"],
                            "similarity": chunk["similarity"],
                            "metadata": chunk["metadata"],
                            "pdf_name": chunk["pdf_name"]
                        }
                        for chunk in context_chunks
                    ],
                    "context_used": context_text
                }
            
            # Prepare context
            context_text = "\n\n".join([
                f"[Page {chunk.get('metadata', {}).get('page_number', '?')}]: {chunk['chunk_text']}"
                for chunk in context_chunks
            ])
            
            # Generate prompt
            prompt = f"""Based on the following context from the Bhagavad Gita, please answer the question concisely and clearly.

Context:
{context_text}

Question: {question}

Please provide a helpful answer based only on the context provided. If the context doesn't contain enough information, please say so. Keep your answer focused and informative."""

            logger.info("Calling LLM API for answer generation...")
            
            # Try different LLM clients
            response = None
            try:
                # Try Groq
                if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                    response = self.llm_client.chat.completions.create(
                        model=settings.llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context from the Bhagavad Gita. Be accurate and provide clear, concise answers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=settings.temperature,
                        max_tokens=settings.max_tokens
                    )
                    answer = response.choices[0].message.content
                    logger.info(f"Groq LLM Response: {answer[:200]}...")  # Log first 200 chars
                else:
                    raise Exception("Groq client not available")
            except Exception as e:
                logger.warning(f"Groq API failed: {e}")
                # Try OpenAI
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context from the Bhagavad Gita. Be accurate and provide clear, concise answers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=settings.temperature,
                        max_tokens=settings.max_tokens
                    )
                    answer = response.choices[0].message.content
                except Exception as e2:
                    logger.error(f"Both LLM APIs failed: {e2}")
                    raise e2
            
            logger.info("Successfully generated answer")
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "chunk_text": chunk["chunk_text"],
                        "similarity": chunk["similarity"],
                        "metadata": chunk["metadata"],
                        "pdf_name": chunk["pdf_name"]
                    }
                    for chunk in context_chunks
                ],
                "context_used": context_text
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to raw context
            context_text = "\n\n---\n\n".join([
                f"Page {chunk.get('metadata', {}).get('page_number', '?')}: {chunk['chunk_text']}"
                for chunk in context_chunks
            ])
            return {
                "answer": f"I encountered an error generating a concise answer, but here are the relevant passages I found:\n\n{context_text}",
                "sources": [
                    {
                        "chunk_text": chunk["chunk_text"],
                        "similarity": chunk["similarity"],
                        "metadata": chunk["metadata"],
                        "pdf_name": chunk["pdf_name"]
                    }
                    for chunk in context_chunks
                ],
                "context_used": context_text
            }
    
    async def chat(
        self,
        question: str,
        session_id: str,
        history: Optional[List[ChatMessage]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete chat pipeline: retrieve context, generate response, log interaction.
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await embedding_service.generate_query_embedding(question)
            
            # Step 2: Retrieve relevant context
            context_chunks = await self.retrieve_relevant_context(
                query_embedding=query_embedding,
                query_text=question
            )
            
            # Step 3: Generate response
            response = await self.generate_answer(question, context_chunks, session_id)
            
            # Step 4: Prepare context text for logging
            context_text = "\n\n".join([
                chunk['chunk_text'] for chunk in context_chunks
            ])
            
            # Step 5: Log interaction asynchronously
            log_metadata = {
                **(metadata or {}),
                'response_time_seconds': time.time() - start_time,
                'context_chunks_count': len(context_chunks),
                'query_embedding_dimension': len(query_embedding)
            }
            
            # Don't wait for logging to complete
            import asyncio
            asyncio.create_task(
                supabase_client.log_chat_interaction(
                    session_id=session_id,
                    question=question,
                    answer=response["answer"],
                    context_used=context_text,
                    metadata=log_metadata
                )
            )
            
            # Step 6: Return response with metadata
            return {
                'answer': response["answer"],
                'session_id': session_id,
                'sources': response["sources"],
                'metadata': {
                    'response_time_seconds': time.time() - start_time,
                    'context_chunks_used': len(context_chunks),
                    'has_context': len(context_chunks) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat pipeline: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your request. Please try again.",
                'session_id': session_id,
                'sources': [],
                'metadata': {
                    'error': str(e),
                    'response_time_seconds': time.time() - start_time
                }
            }


# Global RAG service instance
rag_service = RAGService()
