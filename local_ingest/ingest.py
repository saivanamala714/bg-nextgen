#!/usr/bin/env python3
"""
Local PDF ingestion script for PDF Chat RAG Backend.

This script processes a PDF file locally, generates embeddings using BGE-M3,
and saves the results to a JSON file for later upload to the database.

Usage:
    python local_ingest/ingest.py

Prerequisites:
    - Place your PDF file in local_ingest/your-document.pdf
    - Install requirements: pip install -r requirements.txt
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber  # Using pdfplumber instead of PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PDF_FILENAME = "your-document.pdf"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
OUTPUT_FILENAME = "embeddings_to_upload.json"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


class LocalPDFIngestor:
    """Local PDF processing and embedding generation."""
    
    def __init__(self):
        self.embedding_model = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the BGE-M3 embedding model."""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using pdfplumber.
        Returns list of pages with text and metadata.
        """
        try:
            logger.info(f"Extracting text from: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            pages_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text and text.strip():  # Only add non-empty pages
                        pages_data.append({
                            'page_number': page_num + 1,
                            'text': text.strip(),
                            'metadata': {
                                'page_number': page_num + 1,
                                'pdf_filename': os.path.basename(pdf_path)
                            }
                        })
            
            logger.info(f"Extracted text from {len(pages_data)} pages")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def split_text_into_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter.
        """
        try:
            logger.info("Splitting text into chunks...")
            
            all_chunks = []
            
            for page_data in pages_data:
                # Split the page text into chunks
                chunks = self.text_splitter.split_text(page_data['text'])
                
                for i, chunk_text in enumerate(chunks):
                    chunk_data = {
                        'chunk_text': chunk_text,
                        'metadata': {
                            **page_data['metadata'],
                            'chunk_index': i,
                            'total_chunks_on_page': len(chunks)
                        }
                    }
                    all_chunks.append(chunk_data)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
            raise
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks using BGE-M3.
        """
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            # Extract text from chunks
            texts = [chunk['chunk_text'] for chunk in chunks]
            
            # Generate embeddings in batches
            logger.info("Processing embeddings (this may take a while)...")
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,  # Important for cosine similarity
                show_progress_bar=True,
                batch_size=32
            )
            
            # Combine chunks with embeddings
            chunks_with_embeddings = []
            for i, chunk in enumerate(chunks):
                embedding_list = embeddings[i].tolist()
                
                # Validate dimension
                if len(embedding_list) != EMBEDDING_DIMENSION:
                    logger.warning(f"Embedding dimension mismatch: {len(embedding_list)} vs {EMBEDDING_DIMENSION}")
                    # Pad or truncate if necessary
                    if len(embedding_list) < EMBEDDING_DIMENSION:
                        embedding_list.extend([0.0] * (EMBEDDING_DIMENSION - len(embedding_list)))
                    else:
                        embedding_list = embedding_list[:EMBEDDING_DIMENSION]
                
                chunk_with_embedding = {
                    'chunk_text': chunk['chunk_text'],
                    'embedding': embedding_list,
                    'metadata': chunk['metadata']
                }
                chunks_with_embeddings.append(chunk_with_embedding)
            
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            return chunks_with_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def save_embeddings_to_file(self, chunks_with_embeddings: List[Dict[str, Any]], output_path: str):
        """
        Save chunks with embeddings to JSON file.
        """
        try:
            logger.info(f"Saving embeddings to: {output_path}")
            
            # Prepare data for upload
            upload_data = []
            pdf_name = PDF_FILENAME
            
            for chunk in chunks_with_embeddings:
                upload_chunk = {
                    'chunk_text': chunk['chunk_text'],
                    'embedding': chunk['embedding'],
                    'metadata': chunk['metadata'],
                    'pdf_name': pdf_name
                }
                upload_data.append(upload_chunk)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(upload_data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Saved {len(upload_data)} chunks to {output_path} ({file_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Error saving embeddings to file: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, output_path: str) -> bool:
        """
        Complete PDF processing pipeline.
        """
        try:
            logger.info("Starting PDF processing pipeline...")
            
            # Step 1: Extract text from PDF
            pages_data = self.extract_text_from_pdf(pdf_path)
            
            if not pages_data:
                logger.error("No text extracted from PDF")
                return False
            
            # Step 2: Split text into chunks
            chunks = self.split_text_into_chunks(pages_data)
            
            if not chunks:
                logger.error("No chunks created from text")
                return False
            
            # Step 3: Generate embeddings
            chunks_with_embeddings = self.generate_embeddings(chunks)
            
            # Step 4: Save to file
            self.save_embeddings_to_file(chunks_with_embeddings, output_path)
            
            logger.info("PDF processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in PDF processing pipeline: {e}")
            return False


def main():
    """Main function to run the ingestion process."""
    # Set up paths
    script_dir = Path(__file__).parent
    pdf_path = script_dir / PDF_FILENAME
    output_path = script_dir / OUTPUT_FILENAME
    
    print("=" * 60)
    print("PDF Chat RAG - Local Ingestion Script")
    print("=" * 60)
    print(f"PDF file: {pdf_path}")
    print(f"Output file: {output_path}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("=" * 60)
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"\n❌ ERROR: PDF file not found!")
        print(f"Please place your PDF file at: {pdf_path}")
        print(f"Current file: {PDF_FILENAME}")
        return
    
    # Run ingestion
    ingestor = LocalPDFIngestor()
    success = ingestor.process_pdf(str(pdf_path), str(output_path))
    
    if success:
        print("\n✅ SUCCESS: PDF processing completed!")
        print(f"📁 Output saved to: {output_path}")
        print("\nNext steps:")
        print("1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("2. Upload the embeddings using the API:")
        print(f"   curl -X POST http://localhost:8000/ingest/json-file \\")
        print(f"        -H 'Content-Type: multipart/form-data' \\")
        print(f"        -F 'file=@{output_path}'")
        print("\nOr use the precomputed endpoint:")
        print("   curl -X POST http://localhost:8000/ingest/precomputed \\")
        print("        -H 'Content-Type: application/json' \\")
        print(f"        -d @{output_path}")
    else:
        print("\n❌ FAILED: PDF processing encountered errors!")
        print("Please check the logs above for details.")


if __name__ == "__main__":
    main()
