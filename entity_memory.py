import json, re
from datetime import datetime
import numpy as np

from neo_ollama import get_embedding
from extraction import _NLP

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a, b):
    """How similar two vectors are, from -1 (opposite) to 1 (identical)."""
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)


_embed_cache = {}
_EMBED_CACHE_MAX = 2000  # ceiling so a long-running session doesn't leak memory forever


def _cached_embed(text):
    """Same as get_embedding, but avoids re-embedding a string we've
    already embedded before in this run. Capped at _EMBED_CACHE_MAX —
    simple FIFO eviction (oldest entries dropped first) once full, not a
    full LRU, since exact recency ordering isn't worth the complexity
    here."""
    if text not in _embed_cache:
        if len(_embed_cache) >= _EMBED_CACHE_MAX:
            # dict preserves insertion order in Python 3.7+, so this pops
            # the oldest entry
            _embed_cache.pop(next(iter(_embed_cache)))
        _embed_cache[text] = get_embedding(text)
    return _embed_cache[text]


def _similar(a: str, b: str, threshold=0.8) -> bool:
    """True if two values are basically the same fact said differently
    (e.g. 'Bangalore' vs 'Bengaluru'). Exact text match short-circuits
    the embedding check."""
    if a.lower() == b.lower():
        return True
    return _cosine_sim(_cached_embed(a), _cached_embed(b)) >= threshold


# ---------------------------------------------------------------------------
# Schema setup — safe to call every startup
# ---------------------------------------------------------------------------

def create_entity_tables(conn) -> None:
    """Create entity_attrs and entity_mentions if they don't exist yet.
    Call this on every startup, same as db.py's table creation.

    entity_attrs stores one row per fact-version — e.g. location=Bengaluru
    at version 1, location=Delhi at version 2 — so history is never lost.
    entity_mentions just stores alternate names/nicknames, since those
    don't version (see upsert_entity_profile for why mentions are separate).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_attrs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT COLLATE NOCASE,
            slot TEXT,
            value TEXT,
            ts TEXT,
            version INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_mentions(
            entity_name TEXT COLLATE NOCASE,
            mention TEXT,
            PRIMARY KEY (entity_name, mention)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_slot ON entity_attrs(entity_name, slot)")
    # Speeds up the MAX(version)-per-entity-per-slot lookups used throughout
    # this file (get_active_attrs, resolve_entity_query, etc.) without
    # changing any query logic — SQLite can use this index to find the top
    # version per group directly instead of scanning per row.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_slot_version ON entity_attrs(entity_name, slot, version DESC)")
    _migrate_version_column(conn)


def _migrate_version_column(conn) -> None:
    """One-time patch for entity_attrs tables created before the version
    column existed. If version is already there, does nothing."""
    cols = [row["name"] for row in conn.execute("PRAGMA table_info(entity_attrs)")]
    if "version" in cols:
        return

    conn.execute("ALTER TABLE entity_attrs ADD COLUMN version INTEGER DEFAULT 1")
    rows = conn.execute(
        "SELECT id, entity_name, slot FROM entity_attrs ORDER BY entity_name, slot, ts, id"
    ).fetchall()
    counters = {}
    for r in rows:
        key = (r["entity_name"], r["slot"])
        counters[key] = counters.get(key, 0) + 1
        conn.execute("UPDATE entity_attrs SET version=? WHERE id=?", (counters[key], r["id"]))


def migrate_entity_profile_json(conn) -> None:
    """NOT called automatically. Run this manually, once, only if you have
    an old database still using the OLD entity_profile JSON-blob format
    (single row per entity, attributes as a JSON blob). Converts that old
    data into entity_attrs / entity_mentions.

    NOTE: entity_profile itself is NOT retired — upsert_entity_profile()
    in this file still writes to it on every message, and storage.py
    reads/writes it live. This function only migrates the OLD JSON-blob
    shape of that table, not the table's existence.
    """
    
    try:
        rows = conn.execute("SELECT entity_name, attributes FROM entity_profile").fetchall()
    except Exception:
        rows = []  # entity_profile table doesn't exist — nothing to migrate

    for r in rows:
        attrs = json.loads(r["attributes"])
        for slot, v in attrs.items():
            if slot == "mentions":
                for m in v:
                    conn.execute(
                        "INSERT OR IGNORE INTO entity_mentions(entity_name, mention) VALUES (?,?)",
                        (r["entity_name"], m))
                continue
            for i, e in enumerate(v, start=1):
                conn.execute(
                    "INSERT INTO entity_attrs(entity_name, slot, value, ts, version) VALUES (?,?,?,?,?)",
                    (r["entity_name"], slot, e["value"], e["ts"], i))


# ---------------------------------------------------------------------------
# Reading facts
# ---------------------------------------------------------------------------

def get_primary_entity(conn) -> str | None:
    """Returns the logged-in user's entity name as stored in entity_attrs
    (correcting case if needed), or falls back to the raw username if
    nothing's stored yet."""
    from config import get_user
    username, _ = get_user()
    row = conn.execute(
        "SELECT DISTINCT entity_name FROM entity_attrs WHERE entity_name = ? COLLATE NOCASE",
        (username,)
    ).fetchone()
    return row["entity_name"] if row else username


def get_active_attrs(conn, entity_name: str) -> dict:
    """The current (highest-version) value for every slot known about
    this entity. Ignores history — just what's true right now.

    Slots in the same alias group (see _SLOT_ALIAS_GROUPS, e.g.
    location/moved/transferred) are collapsed to a single entry, keyed
    under whichever alias was most recently written — since
    upsert_entity_profile can leave two alias slots both holding
    "current" (max-version-for-their-own-slot) rows, and both would
    otherwise show up as separate facts even though they mean the same
    thing.
    """
    rows = conn.execute("""
        SELECT slot, value, ts FROM entity_attrs a
        WHERE entity_name=? AND version = (
            SELECT MAX(version) FROM entity_attrs b
            WHERE b.entity_name = a.entity_name AND b.slot = a.slot
        )""", (entity_name,)).fetchall()

    rows = [r for r in rows if r["value"] != "NONE"]

    # Group rows by alias group, keep only the most recently-timestamped
    # row per group.
    best_by_group = {}
    for r in rows:
        group_key = frozenset(_alias_group_for(r["slot"]))
        existing = best_by_group.get(group_key)
        if existing is None or r["ts"] > existing["ts"]:
            best_by_group[group_key] = r

    return {r["slot"]: r["value"] for r in best_by_group.values()}


def get_full_history(conn, entity_name):
    """Every version of every fact ever recorded for this entity, in order."""
    rows = conn.execute(
        "SELECT ts, slot, value, version FROM entity_attrs WHERE entity_name=? ORDER BY slot, version",
        (entity_name,)).fetchall()
    return [(r["ts"], r["slot"], r["value"], r["version"]) for r in rows]


# ---------------------------------------------------------------------------
# Writing facts
# ---------------------------------------------------------------------------

# Slots that mean roughly the same fact, so updating one updates them all
# instead of creating unrelated duplicate slots.
_SLOT_ALIAS_GROUPS = [
    {"location", "moved", "transferred"},
]


def _alias_group_for(slot: str) -> set:
    for group in _SLOT_ALIAS_GROUPS:
        if slot in group:
            return group
    return {slot}


def upsert_entity_profile(conn, profiles: dict, source: str = "chat",primary_entity : str=None) -> None:
    from write_gate import write_gate
    """Add or update facts extracted from a message.

    'mentions' (nicknames) go to entity_mentions directly — they're a
    growing list, not a single versioned value, so none of the
    versioning/write-gate logic below applies to them.

    Every other slot: if the new value matches what's already stored
    (same or _similar), just refresh the timestamp — same slot name if
    unchanged, or a new version under the new alias slot name if the
    fact was restated under a different but related slot (see #8 fix
    in get_active_attrs). Otherwise, run it through write_gate before
    storing it as a new version — write_gate can block implausible or
    contradictory claims.
    """
    now = datetime.now().isoformat()
    for name, attrs in profiles.items():
        for k, v in attrs.items():
            if k == "mentions":
                for m in v:
                    conn.execute(
                        "INSERT OR IGNORE INTO entity_mentions(entity_name, mention) VALUES (?,?)",
                        (name, m))
                continue

            alias_group = _alias_group_for(k)
            placeholders = ",".join("?" * len(alias_group))
            last = conn.execute(
                f"SELECT id, slot, value, version FROM entity_attrs WHERE entity_name=? AND slot IN ({placeholders}) "
                f"ORDER BY version DESC LIMIT 1",
                (name, *alias_group)).fetchone()

            if v=="NONE":
                # Explicit correction/denial - write a cleared version directly,
                # bypassing write_gate entirely. This isn't a new competing
                # claim to plausibility; it's the resolution of one.
                next_version=(last["version"] if last else 0)+1
                conn.execute(
                    "INSERT INTO entity_attrs(entity_name, slot, value, ts, version) VALUES (?,?,?,?,?)",(name,k,"NONE",now,next_version)
                        )
                if last:
                    _flag_retracted_history(conn, name, last["value"])
                continue

            if last and last["slot"] in alias_group and _similar(last["value"], v):
                if last["slot"] == k:
                    # Same slot name, same fact restated — just refresh timestamp.
                    conn.execute("UPDATE entity_attrs SET ts=? WHERE id=?", (now, last["id"]))
                else:
                    # Same fact, but restated under a different alias slot
                    # name (e.g. "moved" -> "location"). Insert as a new
                    # version under the NEW slot name so history preserves
                    # both labels, and the newest phrasing wins as active.
                    next_version = last["version"] + 1
                    conn.execute(
                        "INSERT INTO entity_attrs(entity_name, slot, value, ts, version) VALUES (?,?,?,?,?)",
                        (name, k, v, now, next_version))
                continue

            allowed, reason = write_gate(conn, name, k, v, source=source, primary_entity=primary_entity)
            if not allowed:
                from logger import log_info
                log_info(f"WRITE_GATE_BLOCKED: {name}.{k}={v} — {reason}", level="WARNING", module="WRITE_GATE")
                continue

            next_version = (last["version"] if last else 0) + 1
            is_new_slot = last is None
            conn.execute(
                "INSERT INTO entity_attrs(entity_name, slot, value, ts, version) VALUES (?,?,?,?,?)",
                (name, k, v, now, next_version))
            if is_new_slot:
                invalidate_slot_embed_cache()


# ---------------------------------------------------------------------------
# Answering questions about entities
# ---------------------------------------------------------------------------

_SLOT_EMBED_CACHE = None


def _get_attr_embeddings(conn):
    """Pre-embed a 'what is my <slot>' question for every known slot,
    so incoming queries can be matched against them by similarity."""
    keys = {r["slot"] for r in conn.execute("SELECT DISTINCT slot FROM entity_attrs")}
    return {k: get_embedding(f"what is my {k}") for k in keys}


def invalidate_slot_embed_cache() -> None:
    """Call after any write that could introduce a new slot name
    (upsert_entity_profile, backfill) so infer_attr_slot picks it up
    on next use instead of staying stale for the process lifetime."""
    global _SLOT_EMBED_CACHE
    _SLOT_EMBED_CACHE = None


def infer_attr_slot(query, conn, threshold=0.32):
    """Guess which slot a question is asking about, e.g. 'where do I live'
    -> 'location'. Returns None if nothing matches well enough."""
    global _SLOT_EMBED_CACHE
    if _SLOT_EMBED_CACHE is None:
        _SLOT_EMBED_CACHE = _get_attr_embeddings(conn)

    qv = get_embedding(query)
    best, best_score = None, threshold
    for slot, v in _SLOT_EMBED_CACHE.items():
        score = _cosine_sim(qv, v)
        if score > best_score:
            best, best_score = slot, score
    return best


def is_self_query(doc) -> bool:
    """True if a parsed sentence refers to the user (I/my/me) and is
    either phrased as a question, OR is an imperative request to talk
    about them ("tell me about myself", "say what you know about me") —
    these don't end in '?' and have no WP/WRB/WDT word, so they were
    previously missed entirely."""
    has_self_ref = any(t.text.lower() in ("i", "my", "me") for t in doc)
    is_question = doc.text.strip().endswith("?") or any(t.tag_ in ("WP", "WRB", "WDT") for t in doc)

    # Imperative self-request: a verb like tell/say/remind whose object is
    # "me"/"myself", or a lemma like "know"/"remember" near "me"/"myself".
    text_lower = doc.text.lower()
    is_imperative_self_request = any(
        f"{verb} {target}" in text_lower or f"{verb} about {target}" in text_lower
        for verb in ("tell", "say", "remind")
        for target in ("me", "myself")
    )

    return has_self_ref and (is_question or is_imperative_self_request)


def resolve_entity_query(query: str, conn) -> str | None:
    """Given a raw question like 'where do I live' or 'what's Rahul's job',
    work out who it's about, whether it wants full history or just the
    current value, and which slot — then return a formatted answer string,
    or None if nothing matches."""
    doc = _NLP(query) if _NLP else None
    is_self = is_self_query(doc) if doc else any(t in query.lower() for t in ("my ", " i "))

    # Self-reference is checked and resolved FIRST, before the loose
    # entity-name scan below runs at all. Previously the name scan ran
    # unconditionally against every entity ever stored, so a
    # self-referential query that happened to also contain another
    # entity's name as a substring word (e.g. "is Ravi's Coffee open" when
    # "Ravi" is a stored entity) could get hijacked into answering about
    # the wrong person, or a genuinely self-referential query with a
    # coincidental name match would never reach primary_entity at all.
    # This isn't a complete fix — a message like "Ravi's birthday, when's
    # mine?" that's ambiguously about two people is still a coin flip —
    # but it removes the common case where "my"/"I" queries get
    # mismatched against unrelated stored names.
    if is_self:
        match = get_primary_entity(conn)
    else:
        names = list({r["entity_name"].strip().lower(): r["entity_name"]
                      for r in conn.execute("SELECT DISTINCT entity_name FROM entity_attrs").fetchall()}.values())
        match = next((n for n in names if re.search(rf'\b{re.escape(n.lower())}\b', query.lower())), None)

    slot = infer_attr_slot(query, conn)

    if not match:
        # Fallback for queries that are neither self-referential nor an
        # exact entity-name match, but still ask broadly ("history",
        # "all") — resolve to primary_entity as a last resort.
        if "history" not in query.lower() and "all" not in query.lower():
            return None
        match = get_primary_entity(conn)
    if not match:
        return None

    history_terms = ("history", "all", "have i", "ever", "before", "used to")
    if any(t in query.lower() for t in history_terms):
        events = get_full_history(conn, match)
        return f"{match} history: " + "; ".join(f"{s}={val}@{ts[:10]}" for ts, s, val, _ in events)

    if slot:
        row = conn.execute("""
            SELECT value, ts FROM entity_attrs a
            WHERE entity_name=? AND slot=? AND version = (
                SELECT MAX(version) FROM entity_attrs b
                WHERE b.entity_name = a.entity_name AND b.slot = a.slot
            )""", (match, slot)).fetchone()
        return f"{match}: {slot}={row['value']} (as of {row['ts'][:10]})" if row else None

    rows = conn.execute("""
        SELECT slot, value FROM entity_attrs a
        WHERE entity_name=? AND version = (
            SELECT MAX(version) FROM entity_attrs b
            WHERE b.entity_name = a.entity_name AND b.slot = a.slot
        )""", (match,)).fetchall()
    mentions = [r["mention"] for r in conn.execute(
        "SELECT mention FROM entity_mentions WHERE entity_name=?", (match,)).fetchall()]

    parts = [f"{r['slot']}={r['value']}" for r in rows]
    if mentions:
        parts.append(f"mentions={mentions}")
    return f"{match}: " + ", ".join(parts) if parts else None


def _flag_retracted_history(conn, entity_name: str, old_value: str) -> None:
    """After a slot gets explicitly cleared, find ai_history rows whose
    response text asserts the non-retracted value, and flag them so
    retrieve_similar() stops surfacing them as live context.

    Uses a word-boundary match rather than raw substring containment.
    Raw substring matching (old_value.lower() in text.lower()) would
    match short values inside unrelated words — e.g. old_value="US"
    matching inside "BUS" or "USE" — and would also fire on unrelated
    mentions that happen to contain the retracted value as a fragment.
    Word boundaries don't make this semantically perfect (a row that
    mentions the old value in an unrelated context, like "I've never
    been to Delhi" when old_value="Delhi", still gets flagged even
    though it isn't actually asserting the retracted fact) — that would
    need an LLM call per retraction to fix properly, which isn't worth
    the cost for this. This only narrows the false-positive rate on
    short/substring-prone values, it doesn't eliminate false positives.
    """
    if not old_value or not old_value.strip():
        return
    pattern = re.compile(rf'\b{re.escape(old_value.strip())}\b', re.IGNORECASE)
    rows = conn.execute(
        "SELECT id, response FROM ai_history WHERE retracted=0"
    ).fetchall()
    for row in rows:
        if pattern.search(row["response"]):
            conn.execute("UPDATE ai_history SET retracted = 1 WHERE id = ?", (row["id"],))
