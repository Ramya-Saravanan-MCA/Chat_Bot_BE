import lancedb
import pandas as pd
import time
from embeddings.embedder import embed_chunks
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, db_path, table_name, device='cpu'):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self.df = self.table.to_pandas()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        _ = self.embedding_model.encode(["warmup"], normalize_embeddings=True)

    def embed_query(self, query):
        """Embed the query using the provided embedding model."""
        return embed_chunks([query], model=self.embedding_model)[0]

    def search_dense(self, table, query_vector, k=10, doc_filter=None):
        """Dense vector search using LanceDB with STRICT document filtering."""
        search_obj = table.search(query_vector)
        
        # Apply document filter BEFORE search if specified
        if doc_filter:
            if isinstance(doc_filter, str):
                search_obj = search_obj.where(f"doc_id == '{doc_filter}'")
            elif isinstance(doc_filter, list) and len(doc_filter) > 0:
                doc_list = "', '".join(doc_filter)
                search_obj = search_obj.where(f"doc_id IN ('{doc_list}')")
        
        # Select required columns and apply limit
        search_obj = search_obj.select(["chunk_id", "text", "doc_id"]).limit(k)
        
        try:
            result = search_obj.to_pandas()
            # Double-check filtering on the result (safety check)
            if doc_filter and not result.empty:
                if isinstance(doc_filter, str):
                    result = result[result["doc_id"] == doc_filter]
                elif isinstance(doc_filter, list):
                    result = result[result["doc_id"].isin(doc_filter)]
            return result
        except Exception as e:
            print(f"Dense search error: {e}")
            return pd.DataFrame(columns=["chunk_id", "text", "doc_id"])

    def search_sparse(self, table, query, k=10, doc_filter=None):
        """Full-text search using LanceDB (fts) with STRICT document filtering."""
        search_obj = table.search(query, query_type="fts")
        
        # Apply document filter BEFORE search if specified
        if doc_filter:
            if isinstance(doc_filter, str):
                search_obj = search_obj.where(f"doc_id == '{doc_filter}'")
            elif isinstance(doc_filter, list) and len(doc_filter) > 0:
                doc_list = "', '".join(doc_filter)
                search_obj = search_obj.where(f"doc_id IN ('{doc_list}')")
        
        # Select required columns and apply limit
        search_obj = search_obj.select(["chunk_id", "text", "doc_id"]).limit(k)
        
        try:
            result = search_obj.to_pandas()
            # Double-check filtering on the result (safety check)
            if doc_filter and not result.empty:
                if isinstance(doc_filter, str):
                    result = result[result["doc_id"] == doc_filter]
                elif isinstance(doc_filter, list):
                    result = result[result["doc_id"].isin(doc_filter)]
            return result
        except Exception as e:
            print(f"Sparse search error: {e}")
            return pd.DataFrame(columns=["chunk_id", "text", "doc_id"])

    def reciprocal_rank_fusion(self, dense_df, sparse_df, k=60, limit=10):
        k = int(k) if k is not None else 60
        limit = int(limit) if limit is not None else 10
        vector_ranks = {row['text']: i+1 for i, (_, row) in enumerate(dense_df.iterrows())}
        fts_ranks = {row['text']: i+1 for i, (_, row) in enumerate(sparse_df.iterrows())}
        all_texts = set(vector_ranks) | set(fts_ranks)
        scores = {}
        for text in all_texts:
            score = 0
            if text in vector_ranks: score += 1 / (k + vector_ranks[text])
            if text in fts_ranks: score += 1 / (k + fts_ranks[text])
            scores[text] = score
        sorted_texts = sorted(scores, key=scores.get, reverse=True)[:limit]
        text_to_chunk_id = {row['text']: row['chunk_id'] for _, row in pd.concat([dense_df, sparse_df]).iterrows()}
        text_to_doc_id = {row['text']: row['doc_id'] for _, row in pd.concat([dense_df, sparse_df]).iterrows()}
        final = []
        for text in sorted_texts:
            src = 'both' if text in vector_ranks and text in fts_ranks else ('vector' if text in vector_ranks else 'fts')
            chunk_id = text_to_chunk_id.get(text, -1)
            doc_id = text_to_doc_id.get(text, 'unknown')
            final.append({'chunk_id': chunk_id, 'text': text, 'doc_id': doc_id, 'final_score': scores[text], 'search_source': src})
        return pd.DataFrame(final)

    def retrieve(self, query, mode, k_dense, k_sparse, rrf_k, top_k_final, filter_dict=None, doc_filter=None):
        timings = {}
        total_start = time.perf_counter()

        k_dense = max(int(k_dense or 1), 1) if k_dense is not None else None
        k_sparse = max(int(k_sparse or 1), 1) if k_sparse is not None else None

        # Debug: Log document filter
        if doc_filter:
            print(f" APPLYING DOCUMENT FILTER: {doc_filter}")
        else:
            print(" NO DOCUMENT FILTER - Searching entire knowledge base")

        # Apply metadata filters to table first
        table = self.table
        if filter_dict:
            filter_exprs = []
            for k, v in filter_dict.items():
                if isinstance(v, str):
                    filter_exprs.append(f"{k} == '{v}'")
                else:
                    filter_exprs.append(f"{k} == {v}")
            if filter_exprs:
                filter_expr = " and ".join(filter_exprs)
                table = table.filter(filter_expr)

        if mode == "dense":
            embed_start = time.perf_counter()
            qvec = self.embed_query(query)
            timings["embedding_time"] = float(time.perf_counter() - embed_start)

            dense_start = time.perf_counter()
            dense = self.search_dense(table, qvec, k=k_dense, doc_filter=doc_filter)
            timings["dense_search_time"] = float(time.perf_counter() - dense_start)

            timings["sparse_search_time"] = 0.0

            fusion_start = time.perf_counter()
            results = dense.head(k_dense).copy() if not dense.empty else dense
            if "score" in results.columns:
                results = results.rename(columns={"score": "final_score"})
            elif "distance" in results.columns:
                results["final_score"] = -results["distance"]
            else:
                results["final_score"] = None
            results["search_source"] = "vector"
            timings["fusion_time"] = float(time.perf_counter() - fusion_start)

        elif mode == "sparse":
            timings["embedding_time"] = 0.0
            timings["dense_search_time"] = 0.0

            sparse_start = time.perf_counter()
            sparse = self.search_sparse(table, query, k=k_sparse, doc_filter=doc_filter)
            timings["sparse_search_time"] = float(time.perf_counter() - sparse_start)

            fusion_start = time.perf_counter()
            results = sparse.head(k_sparse).copy() if not sparse.empty else sparse
            if "score" in results.columns:
                results = results.rename(columns={"score": "final_score"})
            elif "distance" in results.columns:
                results["final_score"] = -results["distance"]
            else:
                results["final_score"] = None
            results["search_source"] = "fts"
            timings["fusion_time"] = float(time.perf_counter() - fusion_start)

        else:  # hybrid
            embed_start = time.perf_counter()
            qvec = self.embed_query(query)
            timings["embedding_time"] = float(time.perf_counter() - embed_start)

            dense_start = time.perf_counter()
            dense = self.search_dense(table, qvec, k=k_dense, doc_filter=doc_filter)
            timings["dense_search_time"] = float(time.perf_counter() - dense_start)

            sparse_start = time.perf_counter()
            sparse = self.search_sparse(table, query, k=k_sparse, doc_filter=doc_filter)
            timings["sparse_search_time"] = float(time.perf_counter() - sparse_start)

            fusion_start = time.perf_counter()
            rrf_k = int(rrf_k) if rrf_k is not None else 60
            top_k_final = int(top_k_final) if top_k_final is not None else 10
            
            # Only proceed with fusion if we have results
            if not dense.empty or not sparse.empty:
                results = self.reciprocal_rank_fusion(dense, sparse, k=rrf_k, limit=top_k_final)
            else:
                results = pd.DataFrame(columns=["chunk_id", "text", "doc_id", "final_score", "search_source"])
            
            timings["fusion_time"] = float(time.perf_counter() - fusion_start)

        # Debug: Log results
        if not results.empty and "doc_id" in results.columns:
            unique_docs = results["doc_id"].unique().tolist()
            print(f" SEARCH RESULTS FROM DOCUMENTS: {unique_docs}")
        else:
            print(" NO RESULTS FOUND")

        timings["retrieval_time"] = float(time.perf_counter() - total_start)

        # Ensure required columns
        for col in ["chunk_id", "text", "final_score", "search_source", "doc_id"]:
            if col not in results.columns:
                results[col] = None

        output_cols = ["chunk_id", "text", "doc_id", "final_score", "search_source"]
        for meta_col in ["uid", "logical_id", "doc_version", "file_hash", "page", "chunk_ordinal", "content_hash", "embedder", "embedder_ver", "is_active", "lang"]:
            if meta_col in results.columns and meta_col not in output_cols:
                output_cols.append(meta_col)

        results = results[[col for col in output_cols if col in results.columns]]
        
        # Final safety check - ensure document filter is respected
        if doc_filter and not results.empty and "doc_id" in results.columns:
            if isinstance(doc_filter, str):
                results = results[results["doc_id"] == doc_filter]
            elif isinstance(doc_filter, list):
                results = results[results["doc_id"].isin(doc_filter)]
            print(f" FINAL FILTERED RESULTS: {len(results)} chunks")
        
        return results, timings