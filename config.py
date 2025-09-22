import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LANCEDB_PATH = "rag_chatbot/data/lancedb"
LANCEDB_TABLE = "documents"
