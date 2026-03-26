import os


from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser




# Must use the "exact same" Embedding model as when creating the database
embedding_model = "qwen3-embedding:8b"
embedding = OllamaEmbeddings(model=embedding_model)


chroma_db_dir = "cuda_quantum_chroma_db"


# Load the existing ChromaDB
vectorstore = Chroma(
    persist_directory=chroma_db_dir,
    embedding_function=embedding
)


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)


llm = ChatOllama(model="qwen3:14b-q4_K_M")

template = """Act as a professional NVIDIA CUDA-Q assistant. Use the following pieces of retrieved context to answer the question. 
If the answer is not contained within the text, say you don't know, but try to use all relevant details provided.

[Context]
{context}

[Question]
{question}

Answer concisely and professionally.
"""
prompt = ChatPromptTemplate.from_template(template)



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



while True:
    user_input = input("Please enter your question (enter 'q' to quit): ").strip()
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Ending conversation.")
        break
    if not user_input:
        continue
        
    print("Retrieving and reranking with the local BGE model...")
    
    # Execute retrieval and reranking for display and for the chain
    # Note: rag_chain also calls the retriever, but we do it manually here to show sources
    retrieved_docs = retriever.invoke(user_input)
    
    print("\n[Retrieved Relevant Materials (Top 6)]")
    for i, doc in enumerate(retrieved_docs[:6], 1):
        source = doc.metadata.get("source", "Unknown Source")
        chunk_file = doc.metadata.get("chunk_file", "Unknown Chunk File")
        score = doc.metadata.get("relevance_score", "N/A")
        
        print(f"--- Reference {i} (Rerank Score: {score} | Source: {source} | Chunk File: {chunk_file}) ---")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print("\n")
            
    print("="*50)
    print("Generating answer from LLM...")
    
    # Actually call the RAG chain
    response = rag_chain.invoke(user_input)
    
    print("\n[AI Assistant Answer]")
    print(response)
    print("\n" + "="*50 + "\n")