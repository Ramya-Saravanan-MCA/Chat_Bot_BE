import lancedb




#------- lancedb doesnt support schema change, if we want to change our schema we have to drop the existing one and create a new one

db = lancedb.connect("rag_chatbot/data/chatdb") 

# Drop the tables you want to recreate
for table in ["items_v2", "buffer_v2", "sessions_v2","summaries_v2"]:
    try:
        db.drop_table(table)
        print(f"Dropped table: {table}")
    except Exception as e:
        print(f"Could not drop table {table}: {e}")