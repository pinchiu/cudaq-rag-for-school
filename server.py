import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cudaq-rag")

try:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError as e:
    logger.error(f"Failed to import LangChain modules: {e}")
    sys.exit(1)

class RAGAssistant:
    """Class to manage RAG components with session-isolated memory."""
    def __init__(self):
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        self.llm_model = os.getenv("LLM_MODEL", "su_robin/gemma-4-E4B-it-Q4_K_M")
        self.chroma_db_dir = os.getenv("CHROMA_DB_DIR", "cuda_quantum_chroma_db")
        self.max_history = int(os.getenv("MAX_HISTORY", "10"))
        
        # State & Components
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        
        # Session isolated memory
        self.chat_history: dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.history_summary: dict[str, str] = defaultdict(str)

    def initialize(self):
        """Called once during FastAPI startup lifecycle."""
        logger.info("Initializing RAG components and loading models into memory...")
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
1. LANGUAGE DYNAMICS: Strictly mirror the language of the user's prompt.
2. NO EMOJIS: Do NOT use emojis anywhere in your response. Keep the output professional.
3. STRUCTURE: Adhere strictly to the following Markdown hierarchy:
   - `### Executive Summary / 執行摘要`: A concise 2-3 sentence high-level overview.
   - `### Core Mechanics / 核心原理解析`: Deep-dive technical explanation based on the context.
   - `### Implementation / 實作範例`: Code examples and commands.
4. FORMATTING & COMMANDS: Use INLINE code for keywords, functions, and variables.
5. CODE BLOCKS: Provide ONE integrated Python block (`@cudaq.kernel`) and/or ONE integrated C++ block (`__qpu__` or `cudaq::builder`).
6. KNOWLEDGE BOUNDARIES: Anchor explanations in the [Context]. Use `> **[Supplemental Knowledge / 補充知識]**` if pulling from base training.
7. TONE: Precise, developer-centric, and authoritative. Avoid fluff.
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        logger.info("RAG components fully initialized.")

    async def compress_history(self, session_id: str):
        """Compress old chat history into a summary in the background."""
        history = self.chat_history[session_id]
        
        if len(history) > self.max_history:
            logger.info(f"Compressing chat history for session: {session_id}")
            history_str = "\n".join([f"{r}: {c}" for r, c in history[:-2]])
            
            summary_prompt = (
                "Summarize the following conversation key points in Traditional Chinese, "
                "specifically focusing on technical decisions and user requirements:\n\n"
                f"{history_str}"
            )
            
            try:
                summary = await self.llm.ainvoke(summary_prompt)
                self.history_summary[session_id] += f"\nPrevious context: {summary.content}"
                # Keep only the last 2 messages
                self.chat_history[session_id] = history[-2:]
                logger.info(f"History compressed for session: {session_id}.")
            except Exception as e:
                logger.error(f"Failed to compress history: {e}")

# Global instance
assistant = RAGAssistant()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    assistant.initialize()
    yield
    logger.info("Shutting down CUDA-Q RAG API...")

app = FastAPI(title="CUDA-Q RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str = Field(default="default", description="Unique identifier for the user session")

async def generate_rag_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    try:
        # 1. Retrieve documents in a thread to prevent event loop blocking
        docs = await asyncio.to_thread(assistant.retriever.invoke, question)
        
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_file": doc.metadata.get("chunk_file", "Unknown")
            }
            for doc in docs[:3]
        ]
        
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        
        # 2. Prepare context
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # 3. Stream language model response
        logger.info(f"Starting LLM stream for session {session_id}...")
        
        langchain_history = [
            HumanMessage(content=c) if r == "user" else AIMessage(content=c)
            for r, c in assistant.chat_history[session_id]
        ]

        chain = assistant.prompt | assistant.llm | StrOutputParser()
        
        input_data = {
            "context": context, 
            "question": question, 
            "chat_history": langchain_history,
            "history_summary": assistant.history_summary[session_id]
        }

        assistant_full_response = ""
        async for chunk in chain.astream(input_data):
            assistant_full_response += chunk
            yield json.dumps({"type": "token", "data": chunk}) + "\n"
            
        # 4. Update memory & trigger background compression
        assistant.chat_history[session_id].append(("user", question))
        assistant.chat_history[session_id].append(("assistant", assistant_full_response))
        
        # Create a background task so it doesn't delay closing the generator/stream
        asyncio.create_task(assistant.compress_history(session_id))
            
        logger.info("Stream completed successfully.")
        
    except asyncio.CancelledError:
        logger.warning(f"Client disconnected during stream (Session: {session_id})")
        raise
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"

@app.post("/query")
async def query_rag(request: QueryRequest):
    logger.info(f"Received query from session '{request.session_id}': {request.question}")
    return StreamingResponse(
        generate_rag_stream(request.question, request.session_id),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": assistant.llm_model}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
