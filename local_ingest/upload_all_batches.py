#!/usr/bin/env python3
"""
Automated script to upload all batches to the API.
"""

import json
import requests
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    
    return len(chunks)

def upload_batch(api_file: str, base_url: str = "http://localhost:8000"):
    """Upload a single batch to the API."""
    
    try:
        with open(api_file, 'r') as f:
            data = json.load(f)
        
        response = requests.post(
            f"{base_url}/ingest/precomputed",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "file": api_file,
                "inserted_count": result.get("inserted_count", 0),
                "message": result.get("message", "")
            }
        else:
            return {
                "success": False,
                "file": api_file,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "file": api_file,
            "error": str(e)
        }

def main():
    """Main upload process."""
    
    batches_dir = Path("batches")
    if not batches_dir.exists():
        print("❌ Batches directory not found!")
        return
    
    # Get all batch files
    batch_files = sorted(batches_dir.glob("batch_*.json"))
    api_files = []
    
    print(f"🔄 Found {len(batch_files)} batch files")
    
    # Step 1: Convert all batches to API format
    print("\n📝 Converting batches to API format...")
    for i, batch_file in enumerate(batch_files, 1):
        if "_api.json" in batch_file.name:
            continue  # Skip already converted files
            
        api_file = batch_file.with_name(batch_file.name.replace(".json", "_api.json"))
        
        try:
            chunk_count = convert_batch_to_api_format(str(batch_file), str(api_file))
            api_files.append(api_file)
            print(f"  ✅ {i}/{len(batch_files)}: {batch_file.name} → {api_file.name} ({chunk_count} chunks)")
        except Exception as e:
            print(f"  ❌ {i}/{len(batch_files)}: Failed to convert {batch_file.name}: {e}")
    
    print(f"\n📊 Converted {len(api_files)} files")
    
    # Step 2: Upload all batches
    print("\n🚀 Starting upload process...")
    
    total_inserted = 0
    successful_uploads = 0
    failed_uploads = 0
    
    # Upload sequentially to avoid overwhelming the server
    for i, api_file in enumerate(sorted(api_files), 1):
        print(f"  📤 Uploading {i}/{len(api_files)}: {api_file.name}...", end=" ")
        
        result = upload_batch(str(api_file))
        
        if result["success"]:
            total_inserted += result["inserted_count"]
            successful_uploads += 1
            print(f"✅ {result['inserted_count']} chunks")
        else:
            failed_uploads += 1
            print(f"❌ {result['error']}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Summary
    print(f"\n🎉 Upload Complete!")
    print(f"  ✅ Successful uploads: {successful_uploads}/{len(api_files)}")
    print(f"  ❌ Failed uploads: {failed_uploads}")
    print(f"  📊 Total chunks inserted: {total_inserted}")
    
    if failed_uploads == 0:
        print(f"\n🎯 All {len(api_files)} batches uploaded successfully!")
        print(f"💡 You can now test the chat functionality!")
    else:
        print(f"\n⚠️  {failed_uploads} batches failed. Check the errors above.")

if __name__ == "__main__":
    main()
