from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import uuid
import time
import pandas as pd
import re
import lancedb
import psutil
import json
from dotenv import load_dotenv
import numpy as np
from db.ingestor import Ingestor
from retrieval.retriever import Retriever
from db.session_logger import SessionLogger
from llm.conversational import get_conversational_answer
from preprocess.document_loader import preprocess_text
from llm.summarizer import summarizer
from router import (
    rule_gate, call_llm_router, retrieval_strength, route_and_answer,
    TAU_RETRIEVE_STRONG, TAU_RETRIEVE_WEAK, TAU_ROUTER_HIGH,
    reply_greeting, reply_handoff, reply_safety, reply_oos, reply_not_found, reply_chitchat
)

LANCEDB_PATH = "rag_chatbot/data/lancedb"
DATA_DIR = "rag_chatbot/data"
CHATDB_PATH = "rag_chatbot/data/chatdb"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHATDB_PATH, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_unified_knowledge_base_docs(db_path):
    """Get all documents in the unified knowledge base"""
    try:
        db = lancedb.connect(db_path)
        if "unified_knowledge_base" not in db.table_names():
            return []
        
        table = db.open_table("unified_knowledge_base")
        df = table.to_pandas()
        
        if df.empty:
            return []
        
        # Get unique documents with stats
        docs = df.groupby('doc_id').agg({
            'file_hash': 'first',
            'doc_version': 'first',
            'chunk_id': 'count',
            'page': 'max'
        }).reset_index()
        
        docs.columns = ['doc_id', 'file_hash', 'doc_version', 'chunk_count', 'max_page']
        return docs.to_dict('records')
    except Exception as e:
        print(f"Error getting KB docs: {e}")
        return []

def sanitize_table_name(name):
    name_no_ext = os.path.splitext(name)[0]
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name_no_ext)

def get_existing_tables(db_path):
    db = lancedb.connect(db_path)
    return set(db.table_names())

app = FastAPI(
    title="Hybrid RAG Chatbot API",
    description="API endpoints for document ingestion, session management, chat, and comprehensive analytics.",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",       
    "http://13.233.255.83"         
]

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models - response-models
class DocumentIngestRequest(BaseModel):
    force_reindex: Optional[bool] = False

class SessionCreateRequest(BaseModel):
    llm_model: Optional[str] = "Groq"
    retrieval_mode: Optional[str] = "hybrid"
    top_k_dense: Optional[int] = 10
    top_k_sparse: Optional[int] = 10
    rrf_k: Optional[int] = 60
    top_k_final: Optional[int] = 10
    doc_filter: Optional[List[str]] = None  # List of doc_ids to filter

class ChatRequest(BaseModel):
    session_id: str
    query: str
    llm_model: Optional[str] = None  # Use session default if not provided
    retrieval_mode: Optional[str] = None  # Use session default if not provided
    top_k_dense: Optional[int] = None
    top_k_sparse: Optional[int] = None
    rrf_k: Optional[int] = None
    top_k_final: Optional[int] = None

class SessionUpdateRequest(BaseModel):
    llm_model: Optional[str] = None
    retrieval_mode: Optional[str] = None
    top_k_dense: Optional[int] = None
    top_k_sparse: Optional[int] = None
    rrf_k: Optional[int] = None
    top_k_final: Optional[int] = None
    doc_filter: Optional[List[str]] = None

# Enhanced session storage with configuration
sessions = {}
retrievers = {}
session_logger = SessionLogger(CHATDB_PATH)

# --- HTTP Exception global handler ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Enhanced Root Endpoint
@app.get("/")
async def root():
    kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
    endpoints = [
        {"method": "GET", "path": "/health", "description": "Check system and database health"},
        {"method": "GET", "path": "/knowledge-base", "description": "Get unified knowledge base information"},
        {"method": "GET", "path": "/lancedb/vector-tables", "description": "List all vector DB tables"},
        {"method": "GET", "path": "/lancedb/session-tables", "description": "List all session DB tables"},
        {"method": "GET", "path": "/lancedb/table/{db_type}/{table_name}", "description": "Get table info for a specific table"},
        {"method": "GET", "path": "/documents", "description": "List all documents available in the data folder"},
        {"method": "POST", "path": "/documents/ingest", "description": "Ingest documents into the unified knowledge base"},
        {"method": "POST", "path": "/documents/upload", "description": "Upload PDF documents"},
        {"method": "GET", "path": "/documents/chunks", "description": "Get chunks from unified knowledge base"},
        {"method": "POST", "path": "/sessions", "description": "Create a new session for chat"},
        {"method": "PUT", "path": "/sessions/{session_id}", "description": "Update session configuration"},
        {"method": "DELETE", "path": "/sessions/{session_id}", "description": "Delete a session"},
        {"method": "GET", "path": "/sessions/{session_id}", "description": "Get current session details"},
        {"method": "GET", "path": "/sessions/history/{session_id}", "description": "Get session chat history"},
        {"method": "GET", "path": "/sessions/{session_id}/goal-set", "description": "Export session as goal set for RAGAS evaluation"},
        {"method": "POST", "path": "/chat", "description": "Send a chat query and get response"},
        {"method": "GET", "path": "/analytics/sessions/{session_id}/metrics", "description": "Get analytics metrics for a session"},
        {"method": "GET", "path": "/analytics/system", "description": "Get current system and application performance metrics"},
        {"method": "POST", "path": "/sessions/{session_id}/end", "description": "End a session and log summary"}
    ]

    return {
        "msg": "RAG Chatbot API connected",
        "version": "1.0.0",
        "features": ["unified_knowledge_base", "document_filtering", "enhanced_routing_metrics"],
        "endpoints": endpoints,
        "active_sessions": len(sessions),
        "knowledge_base": {
            "total_documents": len(kb_docs),
            "available": len(kb_docs) > 0
        }
    }

# --- Health Check ---
@app.get("/health")
async def health_check():
    try:
        vector_db = lancedb.connect(LANCEDB_PATH)
        chat_db = lancedb.connect(CHATDB_PATH)
        kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "databases": {
                "vector_db": "connected",
                "chat_db": "connected",
                "vector_tables": len(vector_db.table_names()),
                "chat_tables": len(chat_db.table_names())
            },
            "knowledge_base": {
                "documents": len(kb_docs),
                "available": len(kb_docs) > 0
            },
            "active_sessions": len(sessions),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# --- Knowledge Base Endpoint ---
@app.get("/knowledge-base")
def get_knowledge_base_info():
    try:
        kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
        
        if not kb_docs:
            return {
                "status": "empty",
                "documents": [],
                "total_documents": 0,
                "message": "Knowledge base is empty. Upload documents to get started."
            }
        
        return {
            "status": "available",
            "documents": kb_docs,
            "total_documents": len(kb_docs),
            "total_chunks": sum(doc["chunk_count"] for doc in kb_docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving knowledge base info: {str(e)}")

# LanceDB Table Endpoints (unchanged)
@app.get("/lancedb/vector-tables")
def list_vector_tables():
    try:
        db = lancedb.connect(LANCEDB_PATH)
        tables = db.table_names()
        
        table_info = []
        for table_name in tables:
            try:
                table = db.open_table(table_name)
                df = table.to_pandas()
                table_info.append({
                    "name": table_name,
                    "row_count": len(df),
                    "columns": list(df.columns) if not df.empty else []
                })
            except Exception as e:
                table_info.append({
                    "name": table_name,
                    "error": str(e)
                })
        
        return {
            "tables": table_info,
            "total_tables": len(tables)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing vector tables: {str(e)}")

@app.get("/lancedb/session-tables")
def list_session_tables():
    try:
        db = lancedb.connect(CHATDB_PATH)
        tables = db.table_names()
        
        table_info = []
        for table_name in tables:
            try:
                table = db.open_table(table_name)
                df = table.to_pandas()
                table_info.append({
                    "name": table_name,
                    "row_count": len(df),
                    "columns": list(df.columns) if not df.empty else []
                })
            except Exception as e:
                table_info.append({
                    "name": table_name,
                    "error": str(e)
                })
        
        return {
            "tables": table_info,
            "total_tables": len(tables)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing session tables: {str(e)}")

@app.get("/lancedb/table/{db_type}/{table_name}")
def get_table_info(db_type: str, table_name: str, limit: Optional[int] = 10):
    if db_type not in ["vector", "session"]:
        raise HTTPException(status_code=400, detail="db_type must be 'vector' or 'session'")
    
    try:
        db_path = LANCEDB_PATH if db_type == "vector" else CHATDB_PATH
        db = lancedb.connect(db_path)
        
        if table_name not in db.table_names():
            raise HTTPException(status_code=404, detail="Table not found")
        
        table = db.open_table(table_name)
        df = table.to_pandas()
        
        return {
            "table_name": table_name,
            "db_type": db_type,
            "total_rows": len(df),
            "columns": list(df.columns) if not df.empty else [],
            "sample_rows": df.head(limit).to_dict("records") if not df.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving table info: {str(e)}")

# Document Endpoints
@app.get("/documents")
def list_documents():
    try:
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
        kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
        
        # Create a mapping of processed files
        indexed_files = set()
        for doc in kb_docs:
            # Try to match doc_id to filename
            indexed_files.add(doc['doc_id'])
        
        documents = []
        for f in pdf_files:
            file_path = os.path.join(DATA_DIR, f)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            is_indexed = f in indexed_files or os.path.splitext(f)[0] in indexed_files
            
            documents.append({
                "name": f,
                "is_indexed": is_indexed,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            })
        
        return {
            "documents": documents,
            "total_documents": len(pdf_files),
            "indexed_documents": len([d for d in documents if d["is_indexed"]]),
            "knowledge_base_documents": len(kb_docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/documents/ingest")
def ingest_documents(request: DocumentIngestRequest, background_tasks: BackgroundTasks):
    try:
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            raise HTTPException(status_code=404, detail="No PDF files found in data directory.")
        
        # Always use unified knowledge base
        table_name = "unified_knowledge_base"
        existing_tables = get_existing_tables(LANCEDB_PATH)
        
        if table_name in existing_tables and not request.force_reindex:
            kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
            return {
                "status": "already_indexed",
                "table_name": table_name,
                "existing_documents": len(kb_docs),
                "message": "Knowledge base already exists. Use force_reindex=true to reindex all documents."
            }
        
        # Ingest all PDF files into unified knowledge base
        ingestor = Ingestor(db_path=LANCEDB_PATH, table_name=table_name)
        
        def ingest_all_files():
            for pdf_file in pdf_files:
                file_path = os.path.join(DATA_DIR, pdf_file)
                try:
                    ingestor.run(file_path)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
        
        background_tasks.add_task(ingest_all_files)
        
        return {
            "status": "ingestion_started",
            "table_name": table_name,
            "files_to_process": len(pdf_files),
            "force_reindex": request.force_reindex,
            "files": pdf_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting ingestion: {str(e)}")

@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                continue
            
            file_path = os.path.join(DATA_DIR, file.filename)
            
            # Check if file already exists
            if os.path.exists(file_path):
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "exists",
                    "message": "File already exists"
                })
                continue
            
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            uploaded_files.append({
                "filename": file.filename,
                "status": "uploaded",
                "file_size_bytes": len(content),
                "file_size_mb": round(len(content) / (1024 * 1024), 2)
            })
        
        return {
            "status": "completed",
            "files": uploaded_files,
            "total_uploaded": len([f for f in uploaded_files if f["status"] == "uploaded"]),
            "message": "Use /documents/ingest to add uploaded files to knowledge base"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

def clean_chunk(chunk):
    """Clean chunk data for JSON serialization"""
    cleaned = {}
    for k, v in chunk.items():
        if isinstance(v, (np.int32, np.int64)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.float32, np.float64)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned

@app.get("/documents/chunks")
def get_document_chunks(limit: Optional[int] = 20, doc_filter: Optional[str] = None):
    try:
        table_name = "unified_knowledge_base"
        existing_tables = get_existing_tables(LANCEDB_PATH)

        if table_name not in existing_tables:
            raise HTTPException(status_code=404, detail="Knowledge base is not indexed.")
        
        db = lancedb.connect(LANCEDB_PATH)
        table = db.open_table(table_name)
        df = table.to_pandas()
        
        if df.empty:
            return {
                "table_name": table_name,
                "summary": {"total_chunks": 0},
                "chunks": [],
                "total_returned": 0
            }
        
        # Apply document filter if provided
        if doc_filter:
            doc_ids = [d.strip() for d in doc_filter.split(",")]
            df = df[df["doc_id"].isin(doc_ids)]
        
        if "chunk_id" not in df.columns:
            df["chunk_id"] = range(len(df))
        
        df["chunk_id"] = df["chunk_id"].astype(int)
        df["length"] = df["text"].apply(len)
        
        # Get chunk params safely
        chunk_size = int(df.get("chunk_size", pd.Series([512])).iloc[0]) if "chunk_size" in df.columns else 512
        chunk_overlap = int(df.get("chunk_overlap", pd.Series([50])).iloc[0]) if "chunk_overlap" in df.columns else 50
        
        summary = {
            "total_chunks": int(len(df)),
            "avg_length": float(round(df['length'].mean(), 2)),
            "max_length": int(df['length'].max()),
            "min_length": int(df['length'].min()),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_overlap_ratio": float(round(chunk_overlap / chunk_size, 2)) if chunk_size > 0 else 0
        }
        
        if "doc_id" in df.columns:
            summary["unique_documents"] = int(df["doc_id"].nunique())
        
        # Return cleaned chunks
        raw_chunks = df[["chunk_id", "text", "length", "doc_id"]].head(limit).to_dict("records")
        chunks_data = [clean_chunk(chunk) for chunk in raw_chunks]
        
        return {
            "table_name": table_name,
            "summary": summary,
            "chunks": chunks_data,
            "total_returned": len(chunks_data),
            "filter_applied": doc_filter is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

# Session Endpoints 
@app.post("/sessions")
def create_session(request: SessionCreateRequest):
    try:
        session_id = str(uuid.uuid4())
        table_name = "unified_knowledge_base"
        
        # Check if knowledge base exists
        existing_tables = get_existing_tables(LANCEDB_PATH)
        if table_name not in existing_tables:
            kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
            if not kb_docs:
                raise HTTPException(status_code=400, detail="Knowledge base is empty. Please upload and ingest documents first.")
        
        retriever = Retriever(db_path=LANCEDB_PATH, table_name=table_name)
        retrievers[session_id] = retriever
        
        # Enhanced session storage with configuration
        sessions[session_id] = {
            "table_name": table_name,
            "turn_id": 0,
            "summary": "",
            "created_at": time.time(),
            "last_activity": time.time(),
            "llm_model": request.llm_model,
            "retrieval_mode": request.retrieval_mode,
            "top_k_dense": request.top_k_dense,
            "top_k_sparse": request.top_k_sparse,
            "rrf_k": request.rrf_k,
            "top_k_final": request.top_k_final,
            "doc_filter": request.doc_filter,
            "latency_logs": []
        }
        
        return {
            "session_id": session_id,
            "table_name": table_name,
            "configuration": {
                "llm_model": request.llm_model,
                "retrieval_mode": request.retrieval_mode,
                "top_k_dense": request.top_k_dense,
                "top_k_sparse": request.top_k_sparse,
                "rrf_k": request.rrf_k,
                "top_k_final": request.top_k_final,
                "doc_filter": request.doc_filter
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.put("/sessions/{session_id}")
def update_session_config(session_id: str, request: SessionUpdateRequest):
    """Update session configuration"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session_info = sessions[session_id]
        
        # Update only provided fields
        updates = {}
        if request.llm_model is not None:
            session_info["llm_model"] = request.llm_model
            updates["llm_model"] = request.llm_model
        if request.retrieval_mode is not None:
            session_info["retrieval_mode"] = request.retrieval_mode
            updates["retrieval_mode"] = request.retrieval_mode
        if request.top_k_dense is not None:
            session_info["top_k_dense"] = request.top_k_dense
            updates["top_k_dense"] = request.top_k_dense
        if request.top_k_sparse is not None:
            session_info["top_k_sparse"] = request.top_k_sparse
            updates["top_k_sparse"] = request.top_k_sparse
        if request.rrf_k is not None:
            session_info["rrf_k"] = request.rrf_k
            updates["rrf_k"] = request.rrf_k
        if request.top_k_final is not None:
            session_info["top_k_final"] = request.top_k_final
            updates["top_k_final"] = request.top_k_final
        if request.doc_filter is not None:
            session_info["doc_filter"] = request.doc_filter
            updates["doc_filter"] = request.doc_filter
        
        session_info["last_activity"] = time.time()
        
        return {
            "session_id": session_id,
            "updates_applied": updates,
            "current_configuration": {
                "llm_model": session_info["llm_model"],
                "retrieval_mode": session_info["retrieval_mode"],
                "top_k_dense": session_info["top_k_dense"],
                "top_k_sparse": session_info["top_k_sparse"],
                "rrf_k": session_info["rrf_k"],
                "top_k_final": session_info["top_k_final"],
                "doc_filter": session_info["doc_filter"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        try:
            session_logger.log_session_summary(session_id)
            session_info = sessions.pop(session_id)
            retrievers.pop(session_id, None)
            
            return {
                "status": "deleted",
                "session_id": session_id,
                "total_turns": session_info["turn_id"],
                "session_duration": round(time.time() - session_info["created_at"], 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="Session not found.")

@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session_info = sessions[session_id]
        buffer_df = session_logger.get_buffer(session_id)
        items_df = session_logger.get_items(session_id)
        
        return {
            "session_id": session_id,
            "session": session_info,
            "buffer": buffer_df.to_dict("records") if not buffer_df.empty else [],
            "items": items_df.to_dict("records") if not items_df.empty else [],
            "conversation_count": len(buffer_df) if not buffer_df.empty else 0,
            "configuration": {
                "llm_model": session_info.get("llm_model", "Groq"),
                "retrieval_mode": session_info.get("retrieval_mode", "hybrid"),
                "top_k_dense": session_info.get("top_k_dense", 10),
                "top_k_sparse": session_info.get("top_k_sparse", 10),
                "rrf_k": session_info.get("rrf_k", 60),
                "top_k_final": session_info.get("top_k_final", 10),
                "doc_filter": session_info.get("doc_filter", None)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session info: {str(e)}")

@app.get("/sessions/history/{session_id}")
def get_session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        # Get buffer and items safely
        buffer_df = session_logger.get_buffer(session_id)
        items_df = session_logger.get_items(session_id)

        # Latest buffer turn (if exists)
        latest_turn = buffer_df.iloc[-1].to_dict() if not buffer_df.empty else None

        # All items (queries + answers)
        items_records = items_df.to_dict("records") if not items_df.empty else []

        # Metrics
        metrics_list = [m for m in items_df["metrics"] if isinstance(m, dict)] if "metrics" in items_df.columns else []

        total_items = len(items_records)
        total_rag_queries = sum(1 for m in metrics_list if m.get("used_rag", False)) if metrics_list else 0
        avg_response_time = float(np.mean([m.get("total_time", 0) for m in metrics_list])) if metrics_list else 0

        # Intent distribution
        intents = []
        for m in metrics_list:
            intents.extend(m.get("intent_labels", [])) if "intent_labels" in m else None
        intent_distribution = pd.Series(intents).value_counts().to_dict() if intents else {}

        return {
            "session_id": session_id,
            "buffer": latest_turn,        # Latest query & answer
            "items": items_records,       # All queries & answers
            "metrics": {
                "total_items": total_items,
                "total_rag_queries": total_rag_queries,
                "avg_response_time_sec": round(avg_response_time, 3),
                "intent_distribution": intent_distribution
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {repr(e)}")

@app.get("/sessions/{session_id}/goal-set")
def export_session_goal_set(session_id: str):
    """Export session as goal set for RAGAS evaluation"""
    try:
        items_df = session_logger.get_items(session_id)
        
        if items_df.empty:
            raise HTTPException(status_code=404, detail="No chat history found for this session.")
        
        # Load knowledge base for chunk mapping
        try:
            db = lancedb.connect(LANCEDB_PATH)
            table = db.open_table("unified_knowledge_base")
            kb_df = table.to_pandas()
            kb_df["chunk_id"] = kb_df["chunk_id"].astype(str)
        except Exception as e:
            kb_df = pd.DataFrame()
        
        # Prepare goal set records
        goal_set = []
        for idx, row in items_df.iterrows():
            # Parse chunk ids as list of strings
            chunk_ids = []
            if pd.notna(row.get("retrieved_chunk_ids", "")):
                chunk_ids = [cid.strip() for cid in str(row["retrieved_chunk_ids"]).split(",") if cid.strip().isdigit()]
            
            # Map to chunk texts
            retrieved_contexts = []
            if not kb_df.empty and chunk_ids:
                match = kb_df[kb_df["chunk_id"].isin(chunk_ids)]
                retrieved_contexts = match["text"].tolist()
            
            goal_set.append({
                "question": row.get("user_query", ""),
                "answer": row.get("bot_response", ""),
                "retrieved_contexts": retrieved_contexts,
                "reference": "",  # Add reference answer if available
                "turn_id": row.get("turn_id", idx),
                "session_id": session_id
            })
        
        return {
            "session_id": session_id,
            "goal_set": goal_set,
            "total_records": len(goal_set),
            "format": "RAGAS compatible"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting goal set: {str(e)}")

# Chat Endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    if request.session_id not in sessions or request.session_id not in retrievers:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        retriever = retrievers[request.session_id]
        session_info = sessions[request.session_id]
        
        # Use session defaults if parameters not provided in request
        llm_model = request.llm_model or session_info.get("llm_model", "Groq")
        retrieval_mode = request.retrieval_mode or session_info.get("retrieval_mode", "hybrid")
        top_k_dense = request.top_k_dense or session_info.get("top_k_dense", 10)
        top_k_sparse = request.top_k_sparse or session_info.get("top_k_sparse", 10)
        rrf_k = request.rrf_k or session_info.get("rrf_k", 60)
        top_k_final = request.top_k_final or session_info.get("top_k_final", 10)
        doc_filter = session_info.get("doc_filter", None)
        
        cleaned_input = preprocess_text(request.query)
        past_turns_summary = session_info["summary"]
        
        # Track timing
        query_times = {}
        total_start = time.perf_counter()

        def answer_with_llm(top_texts, query, model_type, past_summary, chunk_ids=None):
            return get_conversational_answer(
                top_texts, query, model_type, past_summary, chunk_ids=chunk_ids
            )
        
        model_type = "openai" if llm_model == "OpenAI" else "groq"
        route_result = route_and_answer(
            cleaned_input,
            past_turns_summary,
            retriever,
            model_type,
            answer_with_llm,
            retrieval_mode=retrieval_mode,
            k_dense=top_k_dense,
            k_sparse=top_k_sparse,
            rrf_k=rrf_k,
            top_k_final=top_k_final,
            doc_filter=doc_filter
        )
        
        # Extract results
        answer = route_result["answer"]
        intent_info = route_result["intent"]
        retrieval_results = route_result["retrieval_results"]
        retrieval_strength_val = route_result["retrieval_strength"]
        used_rag = route_result["used_rag"]
        answer_decision = route_result.get("answer_decision", "")
        top_scores_router = route_result.get("top_scores", [])
        
        # Compile timing information 
        if "timing_info" in route_result:
            query_times.update(route_result["timing_info"])
        if "retrieval_timings" in route_result:
            query_times.update(route_result["retrieval_timings"])
        
        query_times["intent_routing_time"] = query_times.get("rule_routing_time", 0) + query_times.get("llm_routing_time", 0)
        query_times["total_time"] = time.perf_counter() - total_start
        
        # Prepare chunks information
        retrieved_chunks = []
        chunk_ids = []
        if used_rag and retrieval_results is not None:
            retrieved_chunks = retrieval_results.to_dict("records")
            chunk_ids = [int(cid) for cid in retrieval_results["chunk_id"] if cid is not None]
        
        # Add comprehensive metrics 
        query_times.update({
            "query": cleaned_input,
            "response": answer,
            "intent_labels": intent_info.get("labels", []),
            "intent_confidence": intent_info.get("confidence", 0),
            "slots": intent_info.get("slots", {}),
            "retrieval_strength": retrieval_strength_val,
            "retrieved_chunk_ids": chunk_ids,
            "used_rag": used_rag,
            "answer_decision": answer_decision,
            "top_scores": top_scores_router
        })
        
        # Update session
        session_info["turn_id"] += 1
        session_info["last_activity"] = time.time()
        session_info["latency_logs"].append(query_times)
        
        # Update conversation summary
        last_turn = f"User: {cleaned_input}\nAssistant: {answer}"
        session_info["summary"] = summarizer(
            last_turn=last_turn,
            past_summary=session_info["summary"]
        )
        
        # Log to session databases
        session_logger.log_to_buffer(
            session_id=request.session_id,
            turn_id=session_info["turn_id"],
            user_query=cleaned_input,
            bot_response=answer
        )
        
        session_logger.log_to_items(
            session_id=request.session_id,
            turn_id=session_info["turn_id"],
            user_query=cleaned_input,
            bot_response=answer,
            summary=session_info["summary"],
            metrics=query_times,
            retrieval_type=retrieval_mode
        )
        
        return {
            "session_id": request.session_id,
            "turn_id": session_info["turn_id"],
            "query": cleaned_input,
            "answer": answer,
            "intent_info": intent_info,
            "retrieval_strength": retrieval_strength_val,
            "used_rag": used_rag,
            "answer_decision": answer_decision,
            "top_scores": top_scores_router,
            "timing_metrics": query_times,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_chunk_ids": chunk_ids,
            "configuration_used": {
                "llm_model": llm_model,
                "retrieval_mode": retrieval_mode,
                "top_k_dense": top_k_dense,
                "top_k_sparse": top_k_sparse,
                "rrf_k": rrf_k,
                "top_k_final": top_k_final,
                "doc_filter": doc_filter
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/analytics/sessions/{session_id}/metrics")
def get_session_metrics(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        session_info = sessions[session_id]
        latency_logs = session_info.get("latency_logs", [])
        
        if not latency_logs:
            return {
                "session_id": session_id,
                "metrics": {},
                "message": "No metrics available yet. Start chatting to generate metrics."
            }
        
        # Calculate metrics
        df = pd.DataFrame(latency_logs)
        
        # Get retrieval mode specific columns
        mode = session_info.get("retrieval_mode", "hybrid")
        base_columns = ["intent_routing_time"]
        
        if mode == "hybrid":
            columns = base_columns + ["embedding_time", "dense_search_time", "sparse_search_time",
                       "fusion_time", "retrieval_time", "generation_time", "total_time"]
        elif mode == "dense":
            columns = base_columns + ["embedding_time", "dense_search_time", "retrieval_time", "generation_time", "total_time"]
        else:
            columns = base_columns + ["embedding_time", "sparse_search_time", "retrieval_time", "generation_time", "total_time"]
        
        # Calculate statistics
        latest = latency_logs[-1]
        avg_metrics = {}
        for col in columns:
            if col in df.columns:
                avg_metrics[col] = round(df[col].mean(), 3)
        
        # Calculate additional metrics
        total_queries = len(latency_logs)
        total_runtime_sec = sum(log.get("total_time", 0) for log in latency_logs)
        throughput_qps = round(total_queries / total_runtime_sec, 3) if total_runtime_sec > 0 else 0
        
        # Intent distribution
        all_intents = []
        for log in latency_logs:
            all_intents.extend(log.get("intent_labels", []))
        intent_distribution = pd.Series(all_intents).value_counts().to_dict() if all_intents else {}
        
        # Get top scores and format them
        top_scores = latest.get("top_scores", [])
        formatted_scores = [round(x, 3) for x in top_scores]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        return {
            "session_id": session_id,
            "retrieval_mode": mode,
            "total_queries": total_queries,
            "throughput_qps": throughput_qps,
            "latest_query": {
                "intent_labels": latest.get("intent_labels", []),
                "intent_confidence": round(latest.get("intent_confidence", 0), 2),
                "retrieval_strength": round(latest.get("retrieval_strength", 0), 2),
                "used_rag": latest.get("used_rag", False),
                "top_scores": formatted_scores,
                "avg_retrieval_strength": round(avg_score, 4),
                "timing": {k: round(latest.get(k, 0), 3) for k in columns}
            },
            "average_metrics": avg_metrics,
            "intent_distribution": intent_distribution,
            "rag_usage": {
                "total_rag_queries": sum(1 for log in latency_logs if log.get("used_rag", False)),
                "rag_percentage": round((sum(1 for log in latency_logs if log.get("used_rag", False)) / total_queries * 100), 2) if total_queries > 0 else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session metrics: {str(e)}")

@app.get("/analytics/system")
def get_system_metrics():
    """Get current system performance metrics"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024 ** 3)
        ram_used = memory.used / (1024 ** 3)
        ram_available = memory.available / (1024 ** 3)
        ram_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024 ** 3)
        disk_used = disk.used / (1024 ** 3)
        disk_free = disk.free / (1024 ** 3)
        disk_percent = disk.percent
        
        # Application metrics
        total_sessions = len(sessions)
        total_queries = sum(len(s.get("latency_logs", [])) for s in sessions.values())
        total_runtime_sec = sum(
            sum(log.get("total_time", 0) for log in s.get("latency_logs", []))
            for s in sessions.values()
        )
        overall_throughput = round(total_queries / total_runtime_sec, 3) if total_runtime_sec > 0 else 0
        
        return {
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_usage_percent": round(cpu_percent, 2),
                "cpu_frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else 0,
                "ram_total_gb": round(ram_total, 2),
                "ram_used_gb": round(ram_used, 2),
                "ram_available_gb": round(ram_available, 2),
                "ram_usage_percent": round(ram_percent, 2),
                "disk_total_gb": round(disk_total, 2),
                "disk_used_gb": round(disk_used, 2),
                "disk_free_gb": round(disk_free, 2),
                "disk_usage_percent": round(disk_percent, 2)
            },
            "application_metrics": {
                "active_sessions": total_sessions,
                "total_queries_processed": total_queries,
                "overall_throughput_qps": overall_throughput
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system metrics: {str(e)}")

# --- Session Management Endpoints ---
@app.post("/sessions/{session_id}/end")
def end_session(session_id: str):
    try:
        session_logger.log_session_summary(session_id)
        if session_id in sessions:
            session_info = sessions[session_id]
            return {
                "message": "Session ended and summary logged.",
                "session_id": session_id,
                "total_turns": session_info.get("turn_id", 0),
                "duration_seconds": round(time.time() - session_info.get("created_at", time.time()), 2)
            }
        else:
            return {
                "message": "Session summary logged (session was not active).",
                "session_id": session_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")
