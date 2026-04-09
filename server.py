from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import os
import logging
import sys
import json
import asyncio

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
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    logger.info("LangChain modules imported successfully.")
except Exception as e:
    logger.error(f"Failed to import LangChain modules: {e}")
    raise

class RAGAssistant:
    """Singleton-style class to manage RAG components."""
    def __init__(self):
        self.embedding_model = "qwen3-embedding:8b"
        self.llm_model = "su_robin/gemma-4-E4B-it-Q4_K_M"
        self.chroma_db_dir = "cuda_quantum_chroma_db"
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.chat_history: List[tuple] = [] # List of (role, content)
        self.max_history = 10
        self.history_summary = ""

    def initialize(self):
        if self.vectorstore is None:
            logger.info("Initializing RAG components...")
            embedding = OllamaEmbeddings(model=self.embedding_model)
            self.vectorstore = Chroma(
                persist_directory=self.chroma_db_dir,
                embedding_function=embedding
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            self.llm = ChatOllama(model=self.llm_model)
            template = """Act as an expert NVIDIA CUDA-Q and Quantum Computing assistant. 
You are running on a state-of-the-art GPU-accelerated RAG system.

[System Memory Summary]
{history_summary}

[Context]
{context}

---
STRICT OUTPUT RULES:
1. ALWAYS respond in **Traditional Chinese (繁體中文)**. Do NOT switch to English.
2. START with a concise, high-level summary (2-3 sentences).
3. For keywords, function names, or single-line commands, use INLINE code: `like_this`.
4. For full examples, provide ONE integrated Python block and/or ONE integrated C++ block. 
5. Provide technical depth while remaining accessible.
6. If the provided context doesn't cover the answer, prefix your additional insights with [Supplemental Knowledge].

Answer in a premium, clean Markdown format:
"""
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            logger.info("RAG components initialized.")

    async def compress_history(self):
        """Compress old chat history into a summary to save context window."""
        if len(self.chat_history) > self.max_history:
            logger.info("Compressing chat history...")
            history_str = "\n".join([f"{r}: {c}" for r, c in self.chat_history[:-2]])
            summary_prompt = f"Summarize the following conversation key points in Traditional Chinese, specifically focusing on technical decisions and the user's requirements: \n\n{history_str}"
            summary = await self.llm.ainvoke(summary_prompt)
            self.history_summary += f"\nPrevious context: {summary.content}"
            # Keep only the last 2 messages for immediate context
            self.chat_history = self.chat_history[-2:]
            logger.info("History compressed.")

assistant = RAGAssistant()

app = FastAPI(title="CUDA-Q RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

async def generate_rag_stream(question: str) -> AsyncGenerator[str, None]:
    try:
        assistant.initialize()
        
        # 1. Retrieve documents (fast)
        # We run this in a thread because Chroma's invoke is synchronous
        docs = await asyncio.to_thread(assistant.retriever.invoke, question)
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_file": doc.metadata.get("chunk_file", "Unknown")
            }
            for doc in docs[:3]
        ]
        
        # Send sources as the first JSON-line message
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        
        # 2. Prepare context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        context = format_docs(docs)
        
        # 3. Stream language model response
        logger.info("Starting LLM stream...")
        
        # Convert history tuples to LangChain message objects
        langchain_history = []
        for role, content in assistant.chat_history:
            if role == "user":
                langchain_history.append(HumanMessage(content=content))
            else:
                langchain_history.append(AIMessage(content=content))

        chain = assistant.prompt | assistant.llm | StrOutputParser()
        
        assistant_full_response = ""
        input_data = {
            "context": context, 
            "question": question, 
            "chat_history": langchain_history,
            "history_summary": assistant.history_summary
        }

        async for chunk in chain.astream(input_data):
            assistant_full_response += chunk
            yield json.dumps({"type": "token", "data": chunk}) + "\n"
            
        # 4. Update memory
        assistant.chat_history.append(("user", question))
        assistant.chat_history.append(("assistant", assistant_full_response))
        await assistant.compress_history()
            
        logger.info("Stream completed.")
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"

@app.post("/query")
async def query_rag(request: QueryRequest):
    logger.info(f"Received streaming query: {request.question}")
    return StreamingResponse(
        generate_rag_stream(request.question),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": assistant.llm_model}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
