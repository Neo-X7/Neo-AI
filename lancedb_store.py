import lancedb
import os
import pyarrow as pa
from logger import ai_log_info

EMBEDDING_DIM = 768  # matches all-mpnet-base-v2
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lance_store")

_db = None
_table = None
_entity_table = None
_index_partitions={"memory" : None, "entities" : None}


# ---------------------------------------------------------------------------
# Entity vector table — used for entity-based similarity search
# ---------------------------------------------------------------------------

def get_entity_table():
    """Connects to LanceDB and opens the 'entities' table, creating it if
    needed. Only does this work once (first call) — reuses the cached
    table object on every call after that."""
    global _db, _entity_table
    if _entity_table is None:
        _db = lancedb.connect(DB_PATH)
        if "entities" not in _db.table_names():
            schema = pa.schema([
                pa.field("row_id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            ])
            _db.create_table("entities", schema=schema)
        _entity_table = _db.open_table("entities")
    return _entity_table


def insert_entity_vector(row_id: int, vector: list) -> None:
    table = get_entity_table()
    table.add([{"row_id": row_id, "vector": vector}])
    _maybe_build_index(table,"entities")

def search_entity_vectors(query_vector: list, top_k: int = 10) -> list:
    """Returns the ai_history row_ids of the top_k closest entity vectors.
    Empty list if the table has nothing in it yet."""
    table = get_entity_table()
    if table.count_rows() == 0:
        return []
    search=table.search(query_vector).metric('cosine')
    if _index_partitions["entities"] is not None:
        search=search.nprobes(_index_partitions["entities"])
    results=search.limit(top_k).to_list()
    return [r["row_id"] for r in results]


# ---------------------------------------------------------------------------
# Memory vector table — used for general keyword/content similarity search
# ---------------------------------------------------------------------------

def get_lance_table():
    """Same lazy-connect-and-cache pattern as get_entity_table, but for
    the 'memory' table (general content search, not entity-specific)."""
    global _db, _table
    if _table is None:
        _db = lancedb.connect(DB_PATH)
        if "memory" not in _db.table_names():
            schema = pa.schema([
                pa.field("embedding_id", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            ])
            _db.create_table("memory", schema=schema)
            ai_log_info("Vector Database 'memory' created", level="INFO", module="LANCE_DATABASE")
        _table = _db.open_table("memory")
    return _table


def insert_vector(embedding_id: str, vector: list) -> None:
    table = get_lance_table()
    try:
        table.add([{"embedding_id": embedding_id, "vector": vector}])
        ai_log_info(f"Insert inserted: embedding_id={embedding_id}, dim={len(vector)}", level="INFO", module="LANCE_DATABASE")
        _maybe_build_index(table,"memory")
    except Exception as e:
        ai_log_info(f"Insert failed: embedding_id={embedding_id}, dim={len(vector)}, error={e}", level="ERROR", module="LANCE_DATABASE")
        raise

_insert_counters = {"memory": 0, "entities": 0}
INDEX_REBUILD_EVERY = 50
MIN_ROWS_FOR_INDEX = 10000  # below this, brute-force search is fast enough anyway


def _maybe_build_index(table, table_name: str) -> None:
    """Builds an ANN index once the table crosses MIN_ROWS_FOR_INDEX.
    Below that, brute-force search is fast and an index just adds
    overhead with degenerate clustering (empty partitions). Rechecks
    only every INDEX_REBUILD_EVERY inserts to avoid rebuilding on
    every single call."""
    _insert_counters[table_name] += 1
    if _insert_counters[table_name] % INDEX_REBUILD_EVERY != 0:
        return

    row_count = table.count_rows()
    if row_count < MIN_ROWS_FOR_INDEX:
        return

    try:
        num_partitions = max(1, int(row_count ** 0.5))  # sqrt(N) is the standard IVF heuristic
        table.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            index_type="IVF_HNSW_SQ",
        )
        _index_partitions[table_name]=num_partitions
        ai_log_info(f"Index built: {table_name}, {row_count} rows, {num_partitions} partitions", level="INFO", module="LANCE_DATABASE")
    except Exception as e:
        ai_log_info(f"Index build failed: {table_name}, error={e}", level="WARNING", module="LANCE_DATABASE")


def search_similarity(query_vector: list, top_k: int = 15, max_distance: float = 2.0) -> list:
    """Returns embedding_ids of the top_k closest matches, already ordered
    best-to-worst by LanceDB, filtered to only those within max_distance.
    Empty list if the table has nothing in it yet."""
    table = get_lance_table()
    if table.count_rows() == 0:
        return []
    search=table.search(query_vector).metric('cosine')
    if _index_partitions["memory"] is not None:
        search=search.nprobes(_index_partitions["memory"])
    results = search.limit(top_k).to_list()
    filtered = [r for r in results if r.get("_distance", 0) <= max_distance]
    ai_log_info(f"Search: {len(results)} candidates, {len(filtered)} passed max_distance={max_distance}", level="INFO", module="LANCE_DATABASE")
    return [r["embedding_id"] for r in filtered]


def delete_vector(embedding_id: str) -> None:
    table = get_lance_table()
    table.delete(f"embedding_id = '{embedding_id}'")
    ai_log_info(f"Vector deleted : embedding_id={embedding_id}", level="INFO", module="LANCE_DATABASE")


def clear_all_vectors() -> None:
    """Drops and recreates BOTH the 'memory' and 'entities' tables.
    Used by neo.py's /delete command. Both must be cleared together —
    clearing only 'memory' would leave entity vectors pointing at
    ai_history rows that no longer exist."""
    global _db, _table, _entity_table
    _db = lancedb.connect(DB_PATH)
    _index_partitions["memory"]=None
    _index_partitions["entities"]=None

    if "memory" in _db.table_names():
        _db.drop_table("memory")
    memory_schema = pa.schema([
        pa.field("embedding_id", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])
    _db.create_table("memory", schema=memory_schema)
    _table = _db.open_table("memory")

    if "entities" in _db.table_names():
        _db.drop_table("entities")
    entity_schema = pa.schema([
        pa.field("row_id", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])
    _db.create_table("entities", schema=entity_schema)
    _entity_table = _db.open_table("entities")

    ai_log_info("Vector Databases 'memory' and 'entities' have been dropped and recreated", level="INFO", module="LANCE_DATABASE")