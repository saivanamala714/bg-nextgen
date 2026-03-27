#!/usr/bin/env python3
"""
Convert batch embeddings to the correct API format.
"""

import json
import sys
from pathlib import Path

def convert_batch_to_api_format(input_file: str, output_file: str, pdf_name: str = "your-document.pdf"):
    """Convert batch file to API format."""
    
    # Load the batch file (array of chunks)
    with open(input_file, 'r') as f:
        chunks = json.load(f)
    
    # Create the API request format
    api_request = {
        "pdf_name": pdf_name,
        "chunks": chunks
    }
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(api_request, f, indent=2)
    
    print(f"✅ Converted {len(chunks)} chunks to API format")
    print(f"📁 Output: {output_file}")
    print(f"📊 Size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_batch.py <input_batch_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace(".json", "_api.json")
    convert_batch_to_api_format(input_file, output_file)
