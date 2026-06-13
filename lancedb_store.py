import lancedb
import os
EMBEDDING_DIM=768
DB_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"lance_store")
_db = None
_table = None
def get_lance_table():
    global _db, _table
    if _table is None:
        _db = lancedb.connect(DB_PATH)
        if "memory" not in _db.table_names():
            import pyarrow as pa
            schema = pa.schema([pa.field("embedding_id", pa.utf8()), pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM))])
            _db.create_table("memory", schema=schema)
        _table = _db.open_table("memory")
    return _table
def insert_vector(embedding_id:str,vector:list)->None:
    table=get_lance_table()
    table.add([{"embedding_id":embedding_id,"vector":vector}])
def search_similarity(query_vector:list,top_k:int=3)->list:
    table=get_lance_table()
    if table.count_rows()==0:
        return []
    results=table.search(query_vector).limit(top_k).to_list()
    return[r["embedding_id"]for r in results]
def clear_all_vectors()->None:
    global _db,_table
    _db=lancedb.connect(DB_PATH)
    if "memory" in _db.table_names():
        _db.drop_table("memory")
    import pyarrow as pa
    schema = pa.schema([pa.field("embedding_id", pa.utf8()), pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM))])
    _db.create_table("memory", schema=schema)
    _table = _db.open_table("memory")