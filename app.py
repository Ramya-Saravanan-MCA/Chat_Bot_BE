import streamlit as st
import os
import uuid
import time
import pandas as pd
import re
from dotenv import load_dotenv
import lancedb
import psutil
from config import GROQ_API_KEY, EMBEDDING_MODEL_NAME, LANCEDB_PATH
from db.ingestor import Ingestor
from retrieval.retriever import Retriever
from llm.conversational import get_conversational_answer
from preprocess.document_loader import preprocess_text

from llm.summarizer import summarizer
from db.session_logger import SessionLogger
from router import (
    rule_gate, call_llm_router, retrieval_strength, route_and_answer,
    TAU_RETRIEVE_STRONG, TAU_RETRIEVE_WEAK, TAU_ROUTER_HIGH,
    reply_greeting, reply_handoff, reply_safety, reply_oos, reply_not_found, reply_chitchat
)

DATA_DIR = os.getenv("DATA_DIR")
CHATDB_PATH = os.getenv("CHATDB_PATH")
LANCEDB_PATH= os.getenv("LANCEDB_PATH")

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

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHATDB_PATH, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

session_logger = SessionLogger(CHATDB_PATH)

# session state initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "turn_id" not in st.session_state:
    st.session_state["turn_id"] = 0
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "proceeded" not in st.session_state:
    st.session_state["proceeded"] = False
for key, default in [
    ("db_initialized", False), ("messages", []), ("latency_logs", []),
    ("retrieval_mode", "hybrid"), ("selected_doc", None), ("top_texts", []),
    ("top_chunk_ids", []), ("uploaded_doc_name", None), ("llm_model", "Groq"),
    ("top_k_dense", 10), ("top_k_sparse", 10), ("rrf_k", 60), ("top_k_final", 10),
    ("doc_filter", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# MAIN CONFIGURATION INTERFACE 
if not st.session_state["proceeded"]:

    st.set_page_config(page_title="Hybrid RAG Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Hybrid RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("### Session Info")
    st.markdown(f"**Session ID:** `{st.session_state['session_id']}`")

    st.header("📄 Select or Upload Document")
    
    # Get unified knowledge base documents
    kb_docs = get_unified_knowledge_base_docs(LANCEDB_PATH)
    
    # Show knowledge base status
    if kb_docs:
        st.success(f" Knowledge Base contains {len(kb_docs)} documents")
        
        # Display KB documents in expandable section
        with st.expander("View Knowledge Base Documents", expanded=False):
            kb_df = pd.DataFrame(kb_docs)
            if not kb_df.empty:
                kb_df['chunks'] = kb_df['chunk_count']
                kb_df['pages'] = kb_df['max_page']
                display_df = kb_df[['doc_id', 'chunks', 'pages', 'doc_version']].copy()
                st.dataframe(display_df, use_container_width=True)
    else:
        st.info(" Knowledge Base is empty - upload documents below")
    
    # Simplified options - only unified knowledge base approach
    options = []
    if kb_docs:
        options.append("Use entire Knowledge Base")
        options.append("Select specific documents from Knowledge Base")
    
    options.append("Upload new document...")

    selected_doc = st.selectbox("Choose option:", options, key="selected_doc_main")
    uploaded_file = None
    doc_filter = None
    
    # Handle different selection modes
    if selected_doc == "Use entire Knowledge Base":
        st.success(" Using entire Knowledge Base for retrieval")
            
    elif selected_doc == "Select specific documents from Knowledge Base":
        if kb_docs:
            doc_options = [doc['doc_id'] for doc in kb_docs]
            selected_kb_docs = st.multiselect(
                "Select documents to search:",
                doc_options,
                default=doc_options,
                key="kb_doc_select"
            )
            if selected_kb_docs:
                st.info(f"Selected {len(selected_kb_docs)} documents for retrieval")
                doc_filter = selected_kb_docs
            else:
                st.warning("Please select at least one document")
        else:
            st.warning("No documents in Knowledge Base")
            
    elif selected_doc == "Upload new document...":
        uploaded_file = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True, key="file_uploader_main")
        if uploaded_file:
            st.info(f"Ready to upload {len(uploaded_file)} documents")

    llm_model = st.selectbox("Select LLM Model", ["Groq", "OpenAI"], key="llm_model_main")
    retrieval_mode = st.selectbox("Choose retrieval mode:", ["hybrid", "dense", "sparse"], key="retrieval_mode_main")

    if retrieval_mode == "dense":
        top_k_dense = st.number_input("Top-K Dense", 1, 50, 10, step=1, key="top_k_dense_main")
        top_k_sparse = 1
        rrf_k = top_k_final = None
    elif retrieval_mode == "sparse":
        top_k_sparse = st.number_input("Top-K Sparse", 1, 50, 10, step=1, key="top_k_sparse_main")
        top_k_dense = 1
        rrf_k = top_k_final = None
    else:
        top_k_dense = st.number_input("Top-K Dense", 1, 50, 10, step=1, key="top_k_dense_main")
        top_k_sparse = st.number_input("Top-K Sparse", 1, 50, 10, step=1, key="top_k_sparse_main")
        rrf_k = st.number_input("RRF Fusion Parameter (k)", 10, 60, 60, step=1, key="rrf_k_main")
        top_k_final = st.number_input("Final Top-K Results (after fusion)", 1, 20, 10, step=1, key="top_k_final_main")

    if st.button("Proceed"):
        # Handle different scenarios
        if selected_doc == "Use entire Knowledge Base":
            if not kb_docs:
                st.error("Knowledge Base is empty. Please upload documents first.")
                st.session_state["db_initialized"] = False
            else:
                st.session_state["selected_doc"] = "unified_knowledge_base"
                st.session_state["db_initialized"] = True
                st.session_state["doc_filter"] = None
                for k in ["messages", "latency_logs", "top_texts", "top_chunk_ids"]:
                    st.session_state[k] = []
                st.session_state["summary"] = ""
                
        elif selected_doc == "Select specific documents from Knowledge Base":
            if 'selected_kb_docs' in locals() and selected_kb_docs:
                st.session_state["selected_doc"] = "unified_knowledge_base"
                st.session_state["db_initialized"] = True
                st.session_state["doc_filter"] = doc_filter
                for k in ["messages", "latency_logs", "top_texts", "top_chunk_ids"]:
                    st.session_state[k] = []
                st.session_state["summary"] = ""
            else:
                st.error("Please select at least one document")
                st.session_state["db_initialized"] = False
                
        elif selected_doc == "Upload new document..." and uploaded_file:
            ingestor = Ingestor(db_path=LANCEDB_PATH, table_name="unified_knowledge_base")
            with st.spinner(f"🔄 Uploading and indexing {len(uploaded_file)} documents..."):
                success_count = 0
                for file in uploaded_file:
                    file_path = os.path.join(DATA_DIR, file.name)
                    
                    # Save uploaded file
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    
                    try:
                        ingestor.run(file_path)
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                if success_count > 0:
                    st.success(f" Successfully uploaded and indexed {success_count} documents!")
                    st.session_state["selected_doc"] = "unified_knowledge_base"
                    st.session_state["db_initialized"] = True
                    st.session_state["doc_filter"] = None
                    for k in ["messages", "latency_logs", "top_texts", "top_chunk_ids"]:
                        st.session_state[k] = []
                    st.session_state["summary"] = ""
                else:
                    st.error("Failed to upload and index any documents")
                    st.session_state["db_initialized"] = False
                    
        elif selected_doc == "Upload new document..." and not uploaded_file:
            st.error("Please upload at least one document")
            st.session_state["db_initialized"] = False
        else:
            st.session_state["db_initialized"] = False

        st.session_state["llm_model"] = llm_model
        st.session_state["retrieval_mode"] = retrieval_mode
        st.session_state["top_k_dense"] = top_k_dense
        st.session_state["top_k_sparse"] = top_k_sparse
        st.session_state["rrf_k"] = rrf_k
        st.session_state["top_k_final"] = top_k_final
        st.session_state["proceeded"] = True
        st.rerun()

#  MAIN CHAT INTERFACE 
if st.session_state["proceeded"] and st.session_state["db_initialized"]:
    # Always use unified knowledge base
    table_name = "unified_knowledge_base"
        
    retriever_key = f"retriever_{table_name}"
    if retriever_key not in st.session_state:
        with st.spinner("Loading & warming up model..."):
            st.session_state[retriever_key] = Retriever(
                db_path=LANCEDB_PATH,
                table_name=table_name
            )
    retriever = st.session_state[retriever_key]

    llm_model = st.session_state["llm_model"]
    retrieval_mode = st.session_state["retrieval_mode"]
    top_k_dense = st.session_state["top_k_dense"]
    top_k_sparse = st.session_state["top_k_sparse"]
    rrf_k = st.session_state["rrf_k"]
    top_k_final = st.session_state["top_k_final"]
    doc_filter = st.session_state.get("doc_filter", None)

    st.warning(f":orange[Session ID: {st.session_state['session_id']}]")
    st.error(f":red[Turn ID: {st.session_state['turn_id']}]")
    
    # Show active document filter if any
    if doc_filter:
        st.info(f" Filtering search to documents: {', '.join(doc_filter)}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Chat Interface",
        "Retrieval metrics",
        "Data Evaluation Metrics",
        "Conversational Logs",
        "System Metrics",
        "Buffer Tab",
        "Items Tab",
        "Sessions Tab",
    ])

    with tab1:
        st.subheader("💬 Ask Anything")
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Type your question here...", key="main_chat_input")
        
        if user_input:
            turn = st.session_state["turn_id"]
            cleaned_input = preprocess_text(user_input)
            st.session_state["messages"].append({
                "role": "user",
                "content": cleaned_input
            })
            
            query_times = {}
            total_start = time.perf_counter()
            
            # Get past conversation summary for context
            past_turns_summary = st.session_state["summary"]

            def answer_with_llm(top_texts, query, model_type, past_summary, chunk_ids=None):
                return get_conversational_answer(
                    top_texts,
                    query,
                    model_type,
                    past_summary,
                    chunk_ids=chunk_ids
                )
            
            # routing with document filtering
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

            # Extract the results
            answer = route_result["answer"]
            intent_info = route_result["intent"]
            retrieval_results = route_result["retrieval_results"]
            retrieval_strength_val = route_result["retrieval_strength"]
            used_rag = route_result["used_rag"]
            answer_decision = route_result.get("answer_decision", "")
            top_scores_router= route_result.get("top_scores", [])
            st.session_state["top_scores_router"] = top_scores_router
            # Get timings
            if "timing_info" in route_result:
                query_times.update(route_result["timing_info"])
            if "retrieval_timings" in route_result:
                query_times.update(route_result["retrieval_timings"])
            query_times["intent_routing_time"] = query_times.get("rule_routing_time", 0) + query_times.get("llm_routing_time", 0)

            # Chunks
            if used_rag and retrieval_results is not None:
                st.session_state["top_texts"] = list(retrieval_results["text"])
                st.session_state["top_chunk_ids"] = [int(cid) for cid in retrieval_results["chunk_id"] if cid is not None]
            else:
                st.session_state["top_texts"] = []
                st.session_state["top_chunk_ids"] = []
                # Set timing to 0 when no RAG is used
                for k in ["retrieval_time", "embedding_time", "dense_search_time", "sparse_search_time", "fusion_time", "generation_time"]:
                    if k not in query_times:
                        query_times[k] = 0

            query_times["total_time"] = time.perf_counter() - total_start

            # Add all required metrics
            query_times["query"] = cleaned_input
            query_times["response"] = answer
            query_times["intent_labels"] = intent_info.get("labels", [])
            query_times["intent_confidence"] = intent_info.get("confidence", 0)
            query_times["slots"] = intent_info.get("slots", {})
            query_times["retrieval_strength"] = retrieval_strength_val
            query_times["retrieved_chunk_ids"] = st.session_state["top_chunk_ids"]
            query_times["used_rag"] = used_rag
            query_times["answer_decision"] = answer_decision

            st.session_state["latency_logs"].append(query_times)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer
            })
            
            # Update conversation summary
            last_turn = f"User: {cleaned_input}\nAssistant: {answer}"
            st.session_state["summary"] = summarizer(
                last_turn=last_turn,
                past_summary=st.session_state["summary"]
            )
        
            # Log to session buffer
            session_logger.log_to_buffer(
                session_id=st.session_state["session_id"],
                turn_id=turn,
                user_query=cleaned_input,
                bot_response=answer
            )

            # LOG TO ITEMS TABLE + summary
            session_logger.log_to_items(
                session_id=st.session_state["session_id"],
                turn_id=turn,
                user_query=cleaned_input,
                bot_response=answer,
                summary=st.session_state["summary"],
                metrics=query_times,
                retrieval_type=retrieval_mode
            )

            # Update session summary
            session_logger.log_session_summary(st.session_state["session_id"])
            
            st.session_state["turn_id"] += 1
            st.rerun()

    with tab2:
        st.header("Latency Metrics")
        # thresholds -- default values

        st.markdown("--------------------------------------------------------------")
        data = {
            "TAU_RETRIEVE_STRONG": [0.030],
            "TAU_RETRIEVE_WEAK": [0.015],
            "TAU_ROUTER_HIGH": [0.80]
        }

        thresholds_df = pd.DataFrame(data)

        st.subheader("Default Thresholds")
        st.table(thresholds_df)
        st.markdown("--------------------------------------------------------------")
        mode = st.session_state["retrieval_mode"]
        
        # Add intent routing metrics to the columns
        base_columns = ["intent_routing_time"]
        
        if mode == "hybrid":
            columns = base_columns + ["embedding_time", "dense_search_time", "sparse_search_time",
                       "fusion_time", "retrieval_time", "generation_time", "total_time"]
            labels = ["Intent Routing", "Embedding", "Dense", "Sparse", "Fusion", "Retrieval", "Generation", "Total"]
        elif mode == "dense":
            columns = base_columns + ["embedding_time", "dense_search_time", "retrieval_time", "generation_time", "total_time"]
            labels = ["Intent Routing", "Embedding", "Dense", "Retrieval", "Generation", "Total"]
        else:
            columns = base_columns + ["embedding_time", "sparse_search_time", "retrieval_time", "generation_time", "total_time"]
            labels = ["Intent Routing", "Embedding", "Sparse", "Retrieval", "Generation", "Total"]

        if st.session_state["latency_logs"]:
            latest = st.session_state["latency_logs"][-1]
            
            # Display intent information
            st.markdown("**Latest Query Intent**")


            intent_labels = latest.get("intent_labels", [])
            intent_confidence = latest.get("intent_confidence", 0)
            retrieval_strength = latest.get("retrieval_strength", 0)
            used_rag = latest.get("used_rag", False)

            # Get top scores and format them
            top_scores = st.session_state.get("top_scores_router", [])
            formatted_scores = [f"{x:.3f}" for x in top_scores]  
            avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

            # Display all info
            st.write(f"**Intent Labels:** {', '.join(intent_labels)}")
            st.write(f"**Intent Confidence:** {intent_confidence:.2f}")
            st.write(f"**Retrieval Strength:** {retrieval_strength:.2f}")
            st.write(f"**RAG Used:** {'Yes' if used_rag else 'No'}")
            st.write(f"**Top Scores (Retrieval Confidence):** {formatted_scores}")
            st.write(f"**Average Retrieval Strength:** {avg_score:.4f}")

            

            
            st.markdown("**Latest Query Timing**")
            st.table(pd.DataFrame({"Components": labels, "Time (s)": [round(latest.get(c, 0), 3) for c in columns]}))

            df = pd.DataFrame(st.session_state["latency_logs"])
            avg = df.mean(numeric_only=True)
            st.markdown("**Average (All Queries)**")
            st.table(pd.DataFrame({"Components": labels, "Time (s)": [round(avg.get(c, 0), 3) for c in columns]}))

            df["Session ID"] = st.session_state["session_id"]
            log_csv = df[["Session ID", "query", "response"] + columns].round(3).to_csv(index=False).encode("utf-8")
            st.download_button(" Download Logs with Responses", data=log_csv,
                               file_name="latency_logs.csv", mime="text/csv")
        else:
            st.info("No latency metrics yet. Ask something first!")

    with tab3:
        st.header("Data Evaluation Metrics")


        db = lancedb.connect("rag_chatbot/data/lancedb")
        table = db.open_table(table_name)
        df = table.to_pandas()
        
        # Handle metadata columns
        if "chunk_id" not in df.columns:
            df["chunk_id"] = range(len(df))
        df["chunk_id"] = df["chunk_id"].astype(int)
        df["length"] = df["text"].apply(len)

        # Get chunk size from actual data or default
        chunk_size = int(df.get("chunk_size", pd.Series([512])).iloc[0]) if "chunk_size" in df.columns else 512
        chunk_overlap = int(df.get("chunk_overlap", pd.Series([50])).iloc[0]) if "chunk_overlap" in df.columns else 50

        st.subheader("Knowledge Base Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", len(df))
        with col2:
            st.metric("Avg Length", f"{df['length'].mean():.0f} chars")
        with col3:
            if "doc_id" in df.columns:
                unique_docs = df["doc_id"].nunique()
                st.metric("Total Documents", unique_docs)

        # Document breakdown for unified KB
        if "doc_id" in df.columns:
            st.subheader("Documents in Knowledge Base")
            doc_stats = df.groupby('doc_id').agg({
                'chunk_id': 'count',
                'page': 'max',
                'length': 'mean',
                'doc_version': 'first'
            }).reset_index()
            doc_stats.columns = ['Document', 'Chunks', 'Pages', 'Avg Length', 'Version']
            st.dataframe(doc_stats, use_container_width=True)

        # Metadata summary
        with st.expander("Metadata Overview"):
            if "embedder" in df.columns:
                st.write(f"**Embedder:** {df['embedder'].iloc[0]}")
            if "embedder_ver" in df.columns:
                st.write(f"**Embedder Version:** {df['embedder_ver'].iloc[0]}")
            if "lang" in df.columns:
                st.write(f"**Language:** {df['lang'].iloc[0]}")

        with st.expander("Detailed Statistics"):
            st.markdown(f"- **Chunk Size**: {chunk_size}")
            st.markdown(f"- **Chunk Overlap**: {chunk_overlap}")
            st.markdown(f"- **Chunk Overlap Ratio**: {chunk_overlap/chunk_size:.2f}")
            st.markdown(f"- **Max Chunk Length**: {df['length'].max()} characters")
            st.markdown(f"- **Min Chunk Length**: {df['length'].min()} characters")

        st.subheader("Sample Chunks with Metadata")
        # Show sample chunks with metadata
        display_cols = ["chunk_id", "text", "length", "doc_id"]
        metadata_cols = ["page", "chunk_ordinal", "logical_id", "content_hash"]
        for col in metadata_cols:
            if col in df.columns:
                display_cols.append(col)
        
        sample_df = df[display_cols].head(5).copy()
        # Truncate text for display
        if "text" in sample_df.columns:
            sample_df["text"] = sample_df["text"].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
        st.dataframe(sample_df, use_container_width=True)

        st.subheader("Retrieved Chunks")
        
        if st.session_state.get("top_chunk_ids") and len(st.session_state["top_chunk_ids"]) > 0:
            top_chunk_ids = [int(cid) for cid in st.session_state["top_chunk_ids"] if cid is not None and str(cid).isdigit()]
            
            if top_chunk_ids:
                top_df = df[df["chunk_id"].isin(top_chunk_ids)]
                
                if not top_df.empty:
                    st.success(f"Retrieved {len(top_df)} relevant chunks:")
                    
                    # Group by document
                    for doc_id in top_df["doc_id"].unique():
                        doc_chunks = top_df[top_df["doc_id"] == doc_id]
                        st.markdown(f"**📄 Document: {doc_id}** ({len(doc_chunks)} chunks)")
                        
                        for rank, (_, row) in enumerate(doc_chunks.iterrows(), start=1):
                            with st.container():
                                col1, col2 = st.columns([1, 10])
                                with col1:
                                    st.markdown(f"**#{rank}**")
                                    st.caption(f"ID: {row['chunk_id']}")
                                    if "page" in row:
                                        st.caption(f"Page: {row['page']}")
                                    if "chunk_ordinal" in row:
                                        st.caption(f"Ord: {row['chunk_ordinal']}")
                                    st.caption(f"Len: {row['length']}")
                                with col2:
                                    clean_txt = " ".join(row["text"].split())
                                    display_text = clean_txt[:500] + "..." if len(clean_txt) > 500 else clean_txt
                                    st.markdown(f"**Chunk {row['chunk_id']}:** {display_text}")
                                    
                                    # Show metadata
                                    if "logical_id" in row:
                                        st.caption(f"Logical ID: {row['logical_id']}")
                                    if "content_hash" in row:
                                        st.caption(f"Content Hash: {row['content_hash']}")
                    
                    # Download retrieved chunks with metadata
                    download_cols = ["chunk_id", "text", "length", "doc_id"]
                    for col in ["page", "chunk_ordinal", "logical_id", "uid", "content_hash"]:
                        if col in top_df.columns:
                            download_cols.append(col)
                    
                    retrieved_csv = top_df[download_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Retrieved Chunks with Metadata",
                        data=retrieved_csv,
                        file_name=f"retrieved_chunks_turn_{st.session_state['turn_id']}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Chunk IDs found but no matching chunks in database.")
            else:
                st.warning("Invalid chunk IDs retrieved.")
        elif st.session_state.get("top_texts") and len(st.session_state["top_texts"]) > 0:
            st.info("Retrieved text contexts (chunk IDs unavailable):")
            for i, text in enumerate(st.session_state["top_texts"], 1):
                clean_text = " ".join(text.split())
                display_text = clean_text[:400] + "..." if len(clean_text) > 400 else clean_text
                st.markdown(f"**Context {i}:** {display_text}")
        else:
            if st.session_state["turn_id"] == 0:
                st.info("Ask a question to see retrieved document chunks here.")
            else:
                last_logs = st.session_state.get("latency_logs", [])
                if last_logs:
                    last_intent = last_logs[-1].get("intent_labels", ["unknown"])
                    if "greeting" in last_intent or "out_of_scope" in last_intent or "chitchat_smalltalk" in last_intent:
                        st.info(f"Last query was classified as '{', '.join(last_intent)}' - no document retrieval needed.")
                    else:
                        st.warning("No chunks were retrieved for the last query. Try asking about document content.")
                else:
                    st.info("Ask a question about your documents to see retrieved chunks here.")

        st.subheader("Chunk Length Distribution")
        st.bar_chart(df["length"], use_container_width=True)

        st.subheader("Download Options")
        col1, col2 = st.columns(2)
        with col1:
            # Download all chunks with metadata
            all_download_cols = ["chunk_id", "text", "length", "doc_id"]
            for col in ["page", "chunk_ordinal", "logical_id", "uid", "doc_version", "file_hash", "content_hash", "embedder", "embedder_ver", "is_active", "lang"]:
                if col in df.columns:
                    all_download_cols.append(col)
            
            download_df = df[all_download_cols].copy()
            download_df["text"] = download_df["text"].str.replace("\n", " ", regex=False)
            
            st.download_button(
                "Download All Chunks with Metadata",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{table_name}_all_chunks_metadata.csv",
                mime="text/csv"
            )
        with col2:
            if st.session_state.get("top_chunk_ids"):
                st.download_button(
                    "Download Retrieved Chunks",
                    data=retrieved_csv if 'retrieved_csv' in locals() else b"No chunks retrieved",
                    file_name=f"retrieved_chunks_latest.csv",
                    mime="text/csv"
                )
            else:
                st.button("Download Retrieved Chunks", disabled=True, help="No chunks retrieved yet")

    with tab4:
        st.header(" Conversation Logs ")
        st.write(":blue[Latest conversations will be logged here.]")
        buf_df = session_logger.get_buffer(st.session_state["session_id"])
        if not buf_df.empty and "turn_id" in buf_df.columns:
            buf_df = buf_df.sort_values("turn_id")
        
        logs = []
        for _, row in buf_df.iterrows():
            if row["user_query"]:
                logs.append({
                    "Turn ID": row["turn_id"],
                    "Role": "User",
                    "Content": row["user_query"],
                    "Timestamp": row["timestamp"]
                })
            if row["bot_response"]:
                logs.append({
                    "Turn ID": row["turn_id"],
                    "Role": "Assistant",
                    "Content": row["bot_response"],
                    "Timestamp": row["timestamp"]
                })
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs.tail(10), use_container_width=True)
            csv = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Conversation Logs as CSV",
                data=csv,
                file_name="conversation_logs.csv",
                mime="text/csv"
            )
        else:
            st.info("No conversation logs yet. Start chatting to generate logs.")

    with tab5:
        st.header(" System Metrics")

        total_queries = len(st.session_state["latency_logs"])
        total_runtime_sec = sum(log["total_time"] for log in st.session_state["latency_logs"]) or 1
        throughput_qps = total_queries / total_runtime_sec

        if st.session_state["latency_logs"]:
            latest = st.session_state["latency_logs"][-1]
            total_time = latest.get("total_time", 1)
            retrieval_pct = (latest.get("retrieval_time", 0) / total_time) * 100
            generation_pct = (latest.get("generation_time", 0) / total_time) * 100
            intent_pct = (latest.get("intent_routing_time", 0) / total_time) * 100
        else:
            retrieval_pct = generation_pct = intent_pct = 0

        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()

        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024 ** 3)
        ram_used = memory.used / (1024 ** 3)
        ram_available = memory.available / (1024 ** 3)
        ram_percent = memory.percent

        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024 ** 3)
        disk_used = disk.used / (1024 ** 3)
        disk_free = disk.free / (1024 ** 3)
        disk_percent = disk.percent

        st.subheader(" Live Metrics Summary")

        metrics_data = {
            "Metric": [
                "Total Queries",
                "Throughput (QPS)",
                "Intent Routing Time (%)",
                "Retrieval Time (%)",
                "Generation Time (%)",
                "CPU Usage (%)",
                "CPU Frequency (MHz)",
                "RAM Total (GB)",
                "RAM Used (GB)",
                "RAM Available (GB)",
                "RAM Usage (%)",
                "Disk Total (GB)",
                "Disk Used (GB)",
                "Disk Free (GB)",
                "Disk Usage (%)"
            ],
            "Value": [
                total_queries,
                throughput_qps,
                intent_pct,
                retrieval_pct,
                generation_pct,
                cpu_percent,
                cpu_freq.current if cpu_freq else 0,
                ram_total,
                ram_used,
                ram_available,
                ram_percent,
                disk_total,
                disk_used,
                disk_free,
                disk_percent
            ]
        }
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics["Value"] = pd.to_numeric(df_metrics["Value"], errors="coerce")
        st.dataframe(df_metrics.style.format({"Value": "{:.2f}"}), use_container_width=True)

    with tab6:
        st.header("Buffer Table")
        st.success("**Buffer Table**: Contains the latest user queries and bot responses for the current session.")
        buf_df = session_logger.get_buffer(st.session_state["session_id"])
        st.dataframe(buf_df, use_container_width=True)
        if not buf_df.empty:
            st.download_button(
                " Download Buffer CSV",
                data=buf_df.to_csv(index=False).encode("utf-8"),
                file_name=f"buffer_{st.session_state['session_id']}.csv",
                mime="text/csv"
            )
        else:
            st.info("Buffer is empty for this session.")

    with tab7:
        st.header("Items Table")
        st.success("**Items Table**: Contains detailed logs of each chat interaction.")
        items_df = session_logger.get_items(st.session_state["session_id"])
        st.dataframe(items_df, use_container_width=True)
        full_items_df= session_logger.get_items_full()
        if not full_items_df.empty:
            st.download_button(
                " Download Entire Item Table",
                data=full_items_df.to_csv(index=False).encode("utf-8"),
                file_name=f"items_{st.session_state['session_id']}.csv",
                mime="text/csv"
            )
        else:
            st.info("No items logged yet for this session.")

    with tab8:
        st.header("Session Summaries")
        st.success("**Session Summaries**: Contains overall session metrics and summaries.")
        sessions_df = session_logger.get_all_sessions()
        session_curr = session_logger.get_curr_session(st.session_state["session_id"])
        st.dataframe(session_curr, use_container_width=True)
        if not sessions_df.empty:
            st.download_button(
                " Download Sessions Table",
                data=sessions_df.to_csv(index=False).encode("utf-8"),
                file_name="sessions_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("No session summaries yet.")

else:
    st.info("Proceed to the chatbot interface by selecting documents and retrieval mode.")