import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize Embedding model
model = "qwen3-embedding:8b" 
embedding = OllamaEmbeddings(model=model)

# Define split text file path and ChromaDB storage path
splits_dir = "cuda_quantum_full_docs/splits"
chroma_db_dir = "cuda_quantum_chroma_db"

def embed_all_chunks_to_chroma():
    if not os.path.exists(splits_dir):
        print(f"Error: Directory {splits_dir} not found. Please ensure text chunks are generated.")
        return

    # Read chunk files
    files = [f for f in os.listdir(splits_dir) if f.endswith('.txt')]
    print(f"Scanned {len(files)} text chunks, preparing to write to database...")
    
    test_files = files[:]
    texts_to_embed = []
    metadatas = []
    
    # Load all file contents
    for filename in test_files:
        filepath = os.path.join(splits_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            texts_to_embed.append(f.read())
            source_name = filename.split("_chunk_")[0] + ".txt"
            # Create metadata to facilitate tracing back to source (including full chunk filename and original filename)
            metadatas.append({
                "source": source_name,
                "chunk_file": filename
            })
            
    print(f"\nProgress: Converting {len(texts_to_embed)} texts to vectors and writing to ChromaDB...")
    
    # Execute vector conversion and save
    vectorstore = Chroma.from_texts(
        texts=texts_to_embed,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=chroma_db_dir,
        # Cosine Similarity only calculates the angle between vectors, meaning even if the user query is short and the database document is long, as long as their semantic direction is consistent, they can be accurately matched.
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Completed: All data has been converted and saved to {chroma_db_dir}.")
    
    print("\n--- Running example query test ---")
    query = "What is a qubit?"
    print(f"Query Question: {query}")
    results = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}] Source: {doc.metadata.get('source')}")
        print("-" * 40)
        print(doc.page_content[:] + "...")
        print("-" * 40)


if __name__ == "__main__":
    embed_all_chunks_to_chroma()