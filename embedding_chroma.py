import os
import time
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Configuration ---
EMBEDDING_MODEL = "qwen3-embedding:8b"
SPLITS_DIR = "cuda_quantum_full_docs/splits"
CHROMA_DB_DIR = "cuda_quantum_chroma_db"
BATCH_SIZE = 50  # Process 50 chunks at a time for stability

def embed_all_chunks_to_chroma():
    """Reads all chunks and builds/updates the ChromaDB vector store in batches."""
    if not os.path.exists(SPLITS_DIR):
        print(f"[!] Error: Directory {SPLITS_DIR} not found. Please run the crawler first.")
        return

    # Initialize Embedding model
    print(f"[*] Initializing Embedding Model: {EMBEDDING_MODEL}")
    embedding_func = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Gather all chunk files
    chunk_files = sorted([f for f in os.listdir(SPLITS_DIR) if f.endswith('.txt')])
    total_files = len(chunk_files)
    
    if total_files == 0:
        print("[!] No chunks found to embed.")
        return

    print(f"[*] Found {total_files} chunks. Starting batch processing...")
    
    documents = []
    
    # Load all files into memory as LangChain Document objects
    for filename in chunk_files:
        filepath = os.path.join(SPLITS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract the original source filename (remove _chunk_X)
            source_name = filename.split("_chunk_")[0] + ".txt"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "chunk_file": filename,
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            documents.append(doc)

    # Process in batches
    vectorstore = None
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        print(f"[*] Processing batch {i//BATCH_SIZE + 1}/{(len(documents)-1)//BATCH_SIZE + 1}...")
        
        if vectorstore is None:
            # First batch initializes the store
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_func,
                persist_directory=CHROMA_DB_DIR,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # Subsequent batches are added to the existing store
            vectorstore.add_documents(batch)
            
    print(f"[SUCCESS] Knowledge base built and saved to: {CHROMA_DB_DIR}")

    # --- Verification Test ---
    print("\n" + "="*50)
    print("RUNNING SEARCH VERIFICATION")
    print("="*50)
    query = "What is a cudaq kernel?"
    print(f"Test Query: {query}")
    results = vectorstore.similarity_search(query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}] Source: {doc.metadata.get('source')}")
        print("-" * 30)
        # Show first 200 chars
        snippet = doc.page_content[:200].replace('\n', ' ') + "..."
        print(snippet)
        print("-" * 30)

if __name__ == "__main__":
    start_time = time.time()
    embed_all_chunks_to_chroma()
    duration = time.time() - start_time
    print(f"\n[*] Total time elapsed: {duration:.2f} seconds")