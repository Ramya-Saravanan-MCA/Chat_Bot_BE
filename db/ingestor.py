import os
import hashlib
import lancedb
import pandas as pd
from preprocess.document_loader import load_text, load_pdf_with_pages
from preprocess.chunker import chunk_text
from embeddings.embedder import embed_chunks
from sentence_transformers import SentenceTransformer

def compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

class Ingestor:
    def __init__(self, db_path, table_name):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
        self.embedder_ver = "v1"
        self.embedder_name = "hf/all-MiniLM-L6-v2"

    def document_exists(self, file_path):
        """Check if document already exists in unified knowledge base"""
        if self.table_name not in self.db.table_names():
            return False
        
        # Compute file hash to check for existence
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
        
        table = self.db.open_table(self.table_name)
        df = table.to_pandas()
        
        # Check if this file hash already exists
        existing = df[df["file_hash"] == f"f_{file_hash}"]
        return not existing.empty

    def run(self, file_path, force_reindex=False):
        """
        Ingests the file as a LanceDB table with full metadata.
        If force_reindex=True, will re-create the table.
        """
        # Check if document already exists
        if self.document_exists(file_path) and not force_reindex:
            print(f"Document '{os.path.basename(file_path)}' already exists in knowledge base. Skipping ingestion.")
            return

        ext = os.path.splitext(file_path)[1].lower()
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        doc_version = "v1"

        # Compute file hash
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

        # If force_reindex, remove existing document
        if force_reindex and self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.delete(f"file_hash == 'f_{file_hash}'")
            print(f"Removed existing document '{doc_id}' for re-indexing")

        all_rows = []

        if ext == ".pdf":
            # Page-wise extraction for PDF
            pages = load_pdf_with_pages(file_path)
            for page_info in pages:
                if not page_info["text"].strip():
                    continue
                chunks = chunk_text(page_info["text"])
                embeddings = embed_chunks(chunks, model=self.embedding_model)
                for ord, (chunk_text_s, vector) in enumerate(zip(chunks, embeddings)):
                    content_hash = compute_hash(chunk_text_s)[:8]
                    logical_id = f"{doc_id}:p{page_info['page']}:c{ord+1:02d}"
                    uid = f"{logical_id}:h={content_hash}:m={file_hash[:8]}"
                    row = {
                        "uid": uid,
                        "logical_id": logical_id,
                        "doc_id": doc_id,
                        "doc_version": doc_version,
                        "file_hash": f"f_{file_hash[:12]}",
                        "page": page_info["page"],
                        "chunk_ordinal": ord + 1,
                        "text": chunk_text_s,
                        "vector": vector,
                        "content_hash": content_hash[:8],
                        "embedder": self.embedder_name,
                        "embedder_ver": self.embedder_ver,
                        "is_active": True,
                        "lang": "en",
                    }
                    all_rows.append(row)
        else:
            # Non-PDF files
            text = load_text(file_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks, model=self.embedding_model)
            for ord, (chunk_text_s, vector) in enumerate(zip(chunks, embeddings)):
                content_hash = compute_hash(chunk_text_s)[:8]
                logical_id = f"{doc_id}:p1:c{ord+1:02d}"
                uid = f"{logical_id}:h={content_hash}:m={file_hash[:8]}"
                row = {
                    "uid": uid,
                    "logical_id": logical_id,
                    "doc_id": doc_id,
                    "doc_version": doc_version,
                    "file_hash": f"f_{file_hash[:12]}",
                    "page": 1,
                    "chunk_ordinal": ord + 1,
                    "text": chunk_text_s,
                    "vector": vector,
                    "content_hash": content_hash[:8],
                    "embedder": self.embedder_name,
                    "embedder_ver": self.embedder_ver,
                    "is_active": True,
                    "lang": "en",
                }
                all_rows.append(row)

        if not all_rows:
            print("No content to ingest")
            return

        df = pd.DataFrame(all_rows)
        
        # Create or append to unified table
        if self.table_name not in self.db.table_names():
            # Add sequential chunk_id for new table
            df["chunk_id"] = range(len(df))
            table = self.db.create_table(self.table_name, data=df)
            table.create_fts_index("text")
            table.wait_for_index(["text_idx"])
            print(f"Created unified knowledge base with document '{doc_id}'")
        else:
            # Append to existing table
            table = self.db.open_table(self.table_name)
            existing_df = table.to_pandas()
            max_chunk_id = existing_df["chunk_id"].max() if not existing_df.empty else -1
            df["chunk_id"] = range(max_chunk_id + 1, max_chunk_id + 1 + len(df))
            table.add(df)
            print(f"Added document '{doc_id}' to unified knowledge base")

        print("Ingestion and indexing complete with full metadata.")