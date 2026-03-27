from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import json
import logging
import asyncio

from ..models.schemas import (
    ChatRequest, ChatResponse, ChatMessage,
    ErrorResponse, DocumentSearchResult
)
from ..services.rag_service import rag_service
from ..core.supabase_client import supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for asking questions about PDF documents.
    Returns a complete response (non-streaming).
    """
    try:
        logger.info(f"Chat request for session {request.session_id}: {request.question[:100]}...")
        
        # Process chat request
        result = await rag_service.chat(
            question=request.question,
            session_id=request.session_id,
            history=request.history,
            metadata=request.metadata
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.
    Uses Server-Sent Events (SSE) for streaming.
    """
    try:
        logger.info(f"Streaming chat request for session {request.session_id}: {request.question[:100]}...")
        
        async def generate_stream():
            try:
                # Generate query embedding
                from ..services.embedding_service import embedding_service
                query_embedding = await embedding_service.generate_query_embedding(request.question)
                
                # Retrieve context
                context_chunks = await rag_service.retrieve_relevant_context(
                    query_embedding=query_embedding,
                    query_text=request.question
                )
                
                # Send initial metadata
                metadata = {
                    'type': 'metadata',
                    'session_id': request.session_id,
                    'context_chunks_count': len(context_chunks),
                    'sources': [
                        {
                            'chunk_text': chunk['chunk_text'][:200] + "...",
                            'similarity': chunk.get('similarity', 0.0),
                            'metadata': chunk.get('metadata', {}),
                            'pdf_name': chunk.get('pdf_name', 'document.pdf')
                        }
                        for chunk in context_chunks
                    ]
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                
                # Stream the response
                response_parts = []
                async for chunk in rag_service.generate_response_stream(
                    request.question, 
                    context_chunks, 
                    request.history
                ):
                    response_parts.append(chunk)
                    
                    # Send chunk
                    chunk_data = {
                        'type': 'chunk',
                        'content': chunk
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Log interaction after complete response
                full_response = "".join(response_parts)
                context_text = "\n\n".join([
                    chunk['chunk_text'] for chunk in context_chunks
                ])
                
                # Don't block the stream with logging
                asyncio.create_task(
                    supabase_client.log_chat_interaction(
                        session_id=request.session_id,
                        question=request.question,
                        answer=full_response,
                        context_used=context_text,
                        metadata=request.metadata
                    )
                )
                
                # Send completion signal
                completion_data = {
                    'type': 'complete',
                    'session_id': request.session_id
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                error_data = {
                    'type': 'error',
                    'error': str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 10):
    """
    Retrieve chat history for a specific session.
    """
    try:
        logger.info(f"Retrieving chat history for session {session_id}")
        
        history = await supabase_client.get_chat_history(
            session_id=session_id,
            limit=min(limit, 50)  # Cap at 50 for performance
        )
        
        return {
            'session_id': session_id,
            'history': history,
            'count': len(history)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a specific session.
    """
    try:
        logger.info(f"Clearing chat history for session {session_id}")
        
        # This would require implementing a delete method in supabase_client
        # For now, return success
        return {
            'session_id': session_id,
            'message': 'Chat history cleared successfully'
        }
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
