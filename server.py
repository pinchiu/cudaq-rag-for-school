from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import sys

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cudaq-rag")

logger.info("Starting CUDA-Q RAG API initialization...")

try:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    logger.info("LangChain modules imported successfully.")
except Exception as e:
    logger.error(f"Failed to import LangChain modules: {e}")
    raise

app = FastAPI(title="CUDA-Q RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
embedding_model = "qwen3-embedding:8b"
llm_model = "qwen3:14b-q4_K_M"
chroma_db_dir = "cuda_quantum_chroma_db"

# Lazy initialization to avoid hanging on startup
vectorstore = None
retriever = None
rag_chain = None

def get_rag_chain():
    global vectorstore, retriever, rag_chain
    if rag_chain is None:
        logger.info("Initializing RAG chain (Lazy)...")
        try:
            embedding = OllamaEmbeddings(model=embedding_model)
            vectorstore = Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=embedding
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            llm = ChatOllama(model=llm_model)
            
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
            logger.info("RAG chain initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    return rag_chain

class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    content: str
    source: str
    chunk_file: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Received query: {request.question}")
    try:
        chain = get_rag_chain()
        
        # Get sources separately for output
        retrieved_docs = retriever.invoke(request.question)
        sources = [
            SourceDoc(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                chunk_file=doc.metadata.get("chunk_file", "Unknown")
            )
            for doc in retrieved_docs[:3]
        ]
        
        # Get answer
        logger.info("Invoking RAG chain...")
        answer = chain.invoke(request.question)
        logger.info("Answer generated.")
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": llm_model}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
