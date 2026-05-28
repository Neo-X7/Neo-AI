import lancedb
import os
EMBEDDING_DIM=768
DB_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"lance_store")
def get_lance_table():
    db=lancedb.connect(DB_PATH)
    if "memory" not in db.table_names():
        import pyarrow as pa
        schema=pa.schema([pa.field("embedding_id",pa.utf8()),pa.field("vector",pa.list_(pa.float32(),EMBEDDING_DIM))])
        db.create_table("memory",schema=schema)
    return db.open_table("memory")
def insert_vector(embedding_id:str,vector:list)->None:
    table=get_lance_table()
    table.add([{"embedding_id":embedding_id,"vector":vector}])
def search_similarity(query_vector:list,top_k:int=3)->list:
    table=get_lance_table()
    results=table.search(query_vector).limit(top_k).to_list()
    return[r["embedding_id"]for r in results]
def clear_all_vectors()->None:
    db=lancedb.connect(DB_PATH)
    if "memory" in db.table_names():
        db.drop_table("memory")