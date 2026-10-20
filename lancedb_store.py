import lancedb
import os
EMBEDDING_DIM=768
DB_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"lance_store")
_db = None
_table = None
def get_lance_table():
    """Return the shared 'memory' table, creating the database and table on first use."""
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
    """Add one embedding, keyed by embedding_id, to the table."""
    table=get_lance_table()
    table.add([{"embedding_id":embedding_id,"vector":vector}])

def search_similarity(query_vector:list,top_k:int=5,max_distance:float=450)->list:
    """Return embedding ids of the top_k nearest vectors, keeping only those within max_distance.

    Distance is L2, so lower is closer. The default cutoff of 450 was tuned by hand on a
    small test set.
    """
    table=get_lance_table()
    if table.count_rows()==0:
        return []
    results=table.search(query_vector).limit(top_k).to_list()
    return [r["embedding_id"] for r in results if r["_distance"] <= max_distance]

def clear_all_vectors()->None:
    """Drop and recreate the memory table, removing every stored vector."""
    global _db,_table
    _db=lancedb.connect(DB_PATH)
    if "memory" in _db.table_names():
        _db.drop_table("memory")
    import pyarrow as pa
    schema = pa.schema([pa.field("embedding_id", pa.utf8()), pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM))])
    _db.create_table("memory", schema=schema)
    _table = _db.open_table("memory")

def search_with_scores(query_vector : list, top_k : int =10)->list[dict]:
    """Return the top_k nearest vectors as {'embedding_id', 'similarity'} using cosine similarity (higher is closer). Used for fact candidate lookup."""

    table=get_lance_table()
    if table.count_rows()==0:
        return []
    results=table.search(query_vector).metric("cosine").limit(top_k).to_list()
    return [{"embedding_id" : r["embedding_id"], "similarity": 1-r["_distance"]} for r in results]