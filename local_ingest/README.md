# Place your PDF file here
# 
# Instructions:
# 1. Copy your PDF file to this directory
# 2. Rename it to 'your-document.pdf' (or update the PDF_FILENAME in ingest.py)
# 3. Run the ingestion script: python local_ingest/ingest.py
# 
# The script will:
# - Extract text from the PDF
# - Split it into chunks (800 chars with 150 overlap)
# - Generate embeddings using BGE-M3 model
# - Save embeddings to embeddings_to_upload.json
#
# Example:
# cp /path/to/your/document.pdf local_ingest/your-document.pdf
# python local_ingest/ingest.py
