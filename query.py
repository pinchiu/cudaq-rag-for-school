import os


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
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
    #mmr  algorithm for diversity context
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

'''
llm = ChatOllama(model="")

template = """Act as a professional AI assistant. Please answer the question "only based on" the reference materials provided below.
If the answer cannot be found in the reference materials, answer directly: "Based on the information currently provided, I did not find any relevant information."

[Reference Materials]
{context}

[User Question]
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
'''


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


'''
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
'''


while True:
    user_input = input("Please enter your question (enter 'q' to quit): ").strip()
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Ending conversation.")
        break
    if not user_input:
        continue
        
    print("Retrieving and reranking with the local BGE model...")
    
    # Execute retrieval and reranking
    retrieved_docs = retriever.invoke(user_input)
    
    print("\n[Retrieved Relevant Materials (Top 4)]")
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        chunk_file = doc.metadata.get("chunk_file", "Unknown Chunk File")
        # Extract the score assigned by the reranker model
        score = doc.metadata.get("relevance_score", "N/A")
        
        print(f"--- Reference {i} (Rerank Score: {score} | Source: {source} | Chunk File: {chunk_file}) ---")
        print(doc.page_content)
        print("\n")
            
    print("="*50 + "\n")