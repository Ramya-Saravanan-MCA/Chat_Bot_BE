import lancedb
import pandas as pd
import pyarrow as pa
from datetime import datetime
import json

class SessionLogger:
    def __init__(self, db_path: str):
        self.db = lancedb.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        buffer_schema = pa.schema([
            ("session_id", pa.string()),
            ("turn_id", pa.int64()),
            ("user_query", pa.string()),
            ("bot_response", pa.string()),
            ("timestamp", pa.string())
        ])
        items_schema = pa.schema([
            ("session_id", pa.string()),
            ("turn_id", pa.int64()),
            ("user_query", pa.string()),
            ("bot_response", pa.string()),
            ("summary", pa.string()),
            ("intent_labels", pa.string()),
            ("intent_confidence", pa.float64()),
            ("slots", pa.string()),
            ("retrieval_type", pa.string()),
            ("retrieval_strength", pa.float64()),
            ("retrieved_chunk_ids", pa.string()),
            ("used_rag", pa.bool_()),
            ("answer_decision", pa.string()),
            ("embedding_latency_ms", pa.int64()),
            ("dense_retrieval_latency_ms", pa.int64()),
            ("sparse_retrieval_latency_ms", pa.int64()),
            ("llm_latency_ms", pa.int64()),
            ("total_latency_ms", pa.int64()),
            ("timestamp", pa.string())
        ])
        sessions_schema = pa.schema([
            ("session_id", pa.string()),
            ("total_turns", pa.int64()),
            ("total_session_time_ms", pa.int64()),
            ("avg_embedding_latency_ms", pa.float64()),
            ("avg_dense_retrieval_latency_ms", pa.float64()),
            ("avg_sparse_retrieval_latency_ms", pa.float64()),
            ("avg_llm_latency_ms", pa.float64()),
            ("avg_retrieval_strength", pa.float64()),
            ("created_at", pa.string()),
            ("ended_at", pa.string())
        ])
    
        if "buffer" not in self.db.table_names():
            self.db.create_table("buffer", schema=buffer_schema)
        if "items" not in self.db.table_names():
            self.db.create_table("items", schema=items_schema)
        if "sessions" not in self.db.table_names():
            self.db.create_table("sessions", schema=sessions_schema)

    def log_to_buffer(self, session_id, turn_id, user_query, bot_response):
        buffer_table = self.db.open_table("buffer")
        df = buffer_table.to_pandas()
        if not df.empty and df["session_id"].iloc[0] != session_id:
            buffer_table.delete(f"session_id != '{session_id}'")
        row = {
            "session_id": str(session_id) if session_id is not None else "",
            "turn_id": int(turn_id) if turn_id is not None else 0,
            "user_query": str(user_query) if user_query is not None else "",
            "bot_response": str(bot_response) if bot_response is not None else "",
            "timestamp": datetime.now().isoformat()
        }
        buffer_table.add([row])

    

    def log_to_items(self, session_id, turn_id, user_query,summary, bot_response, metrics, retrieval_type):
        items_table = self.db.open_table("items")
        def safe_int(val, default=0):
            try:
                if val is None:
                    return default
                return int(float(val))
            except Exception:
                return default
        row = {
            "session_id": str(session_id) if session_id is not None else "",
            "turn_id": int(turn_id) if turn_id is not None else 0,
            "user_query": str(user_query) if user_query is not None else "",
            "bot_response": str(bot_response) if bot_response is not None else "",
            "summary": str(summary) if summary is not None else "",
            "intent_labels": ",".join(metrics.get("intent_labels", [])),
            "intent_confidence": float(metrics.get("intent_confidence", 0)),
            "slots": json.dumps(metrics.get("slots", {})),
            "retrieval_type": str(retrieval_type) if retrieval_type else "",
            "retrieval_strength": float(metrics.get("retrieval_strength", 0)),
            "retrieved_chunk_ids": ",".join(str(x) for x in metrics.get("retrieved_chunk_ids", [])),
            "used_rag": bool(metrics.get("used_rag", False)),
            "answer_decision": str(metrics.get("answer_decision", "")),
            "embedding_latency_ms": safe_int(metrics.get("embedding_time", 0) * 1000),
            "dense_retrieval_latency_ms": safe_int(metrics.get("dense_search_time", 0) * 1000),
            "sparse_retrieval_latency_ms": safe_int(metrics.get("sparse_search_time", 0) * 1000),
            "llm_latency_ms": safe_int(metrics.get("generation_time", 0) * 1000),
            "total_latency_ms": safe_int(metrics.get("total_time", 0) * 1000),
            "timestamp": datetime.now().isoformat()
        }
        items_table.add([row])

    def log_session_summary(self, session_id):
        items_df = self.db.open_table("items").to_pandas()
        session_df = items_df[items_df["session_id"] == session_id]
        if session_df.empty:
            return
        created_at = session_df["timestamp"].min()
        ended_at = session_df["timestamp"].max()
        total_turns = len(session_df)
        total_time = (pd.to_datetime(ended_at) - pd.to_datetime(created_at)).total_seconds() * 1000
        summary = {
            "session_id": str(session_id),
            "total_turns": int(total_turns),
            "total_session_time_ms": int(total_time),
            "avg_embedding_latency_ms": float(session_df["embedding_latency_ms"].mean()) if not session_df["embedding_latency_ms"].isnull().all() else 0.0,
            "avg_dense_retrieval_latency_ms": float(session_df["dense_retrieval_latency_ms"].mean()) if not session_df["dense_retrieval_latency_ms"].isnull().all() else 0.0,
            "avg_sparse_retrieval_latency_ms": float(session_df["sparse_retrieval_latency_ms"].mean()) if not session_df["sparse_retrieval_latency_ms"].isnull().all() else 0.0,
            "avg_llm_latency_ms": float(session_df["llm_latency_ms"].mean()) if not session_df["llm_latency_ms"].isnull().all() else 0.0,
            "avg_retrieval_strength": float(session_df["retrieval_strength"].mean()) if "retrieval_strength" in session_df.columns and not session_df["retrieval_strength"].isnull().all() else 0.0,
            "created_at": str(created_at),
            "ended_at": str(ended_at)
        }
        sessions_table = self.db.open_table("sessions")
        sessions_table.delete(f"session_id == '{session_id}'")
        sessions_table.add([summary])

    def get_buffer(self, session_id):
        df = self.db.open_table("buffer").to_pandas()
        return df[df["session_id"] == session_id] if not df.empty else pd.DataFrame()

    def get_items(self, session_id):
        df = self.db.open_table("items").to_pandas()
        return df[df["session_id"] == session_id] if not df.empty else pd.DataFrame()
    def get_items_full(self):
        df = self.db.open_table("items").to_pandas()
        return df  if not df.empty else pd.DataFrame()

    def get_all_sessions(self):
        df = self.db.open_table("sessions").to_pandas()
        return df if not df.empty else pd.DataFrame()
    
    def get_curr_session(self,session_id):
        df = self.db.open_table("sessions").to_pandas()
        return df[df["session_id"] == session_id] if not df.empty else pd.DataFrame()