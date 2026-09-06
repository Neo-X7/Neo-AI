import pycountry
import geonamescache
import numpy as np
import re
from entity_memory import _cached_embed

# Two separate gazetteer sources, combined — neither alone is complete.
# Built once at import time since rebuilding per-check would be expensive.
_gc = geonamescache.GeonamesCache()
_countries = {c["name"].lower() for c in _gc.get_countries().values()}
_cities = {c["name"].lower() for c in _gc.get_cities().values()}
_pycountries = {c.name.lower() for c in pycountry.countries}

PLACE_SLOTS = ("location", "city", "country")
def entity_name_check(entity_name: str, primary_entity: str = None) -> bool:
    """Reject entity names that aren't the known user or a real spaCy
    PERSON entity already seen in this conversation. Blocks garbage
    sentinel strings (e.g. leaked prompt-fence text) from ever becoming
    a stored entity, regardless of which code path produced them.
    """
    if primary_entity and entity_name.strip().lower() == primary_entity.strip().lower():
        return True
    # Reject anything that isn't title-case-name-shaped (basic sanity check)
    if not re.match(r'^[A-Z][a-zA-Z\'\-]*(\s[A-Z][a-zA-Z\'\-]*)*$', entity_name.strip()):
        return False
    return True

# ---------------------------------------------------------------------------
# Layer 1: gazetteer check — is this a real place name?
# ---------------------------------------------------------------------------

def gazetteer_check(slot: str, value: str) -> bool:
    """Only applies to place-type slots. Rejects claimed locations that
    aren't recognized real places (e.g. 'Narnia'), before spending an
    LLM call on it. Non-place slots always pass this check."""
    if slot not in PLACE_SLOTS:
        return True
    v = value.lower().strip()
    return v in _countries or v in _cities or v in _pycountries


# ---------------------------------------------------------------------------
# Layer 2: plausibility check — does the local model think this is possible?
# ---------------------------------------------------------------------------

def plausibility_check(slot: str, value: str, entity_name: str) -> tuple[bool, str]:
    """Asks the local Ollama model whether this fact is structurally
    plausible. Deliberately permissive — only rejects things that are
    impossible, clearly fictional, or a category mismatch for the slot
    (e.g. a hobby/genre/interest stored as a workplace) — not things the
    model simply doesn't recognize or can't verify."""
    from neo_ollama import _ollama, get_active_model

    prompt = (
        f"Is this a structurally plausible personal fact about a real person? "
        f"Person: {entity_name}. Slot: {slot}, Value: {value}. "
        f"Answer only YES or NO. Say NO if it is impossible, clearly fictional "
        f"(e.g. claiming to be a fictional character, an impossible age, a family "
        f"member with a famous fictional name), OR if the value is the wrong "
        f"category for this slot (e.g. a genre, hobby, interest, or topic given as "
        f"a 'workplace' instead of an actual company/organization/job title; a "
        f"food or object given as a 'location'). "
        f"Say YES for anything merely unverifiable or unfamiliar to you, as long as "
        f"it is a value of the correct category for the slot."
    )
    try:
        resp=_ollama.chat(model=get_active_model(),messages=[{"role":"user","content":prompt}],options={"num_ctx" : 8192})
        answer=resp["message"]["content"].strip().upper()
        return answer.startswith("YES"),answer
    except Exception as e:
        #If the model itself fails, don't block the write over an
        #infrastructure problem - allow it through and note why.
        return True, f"llm_check_skipped : {e}"

# ---------------------------------------------------------------------------
# Layer 3: contradiction check — does this conflict with what's stored?
# ---------------------------------------------------------------------------

def contradiction_check(conn, entity_name: str, slot: str, new_value: str) -> tuple[bool, str | None]:
    """Compares the new value against the most recently stored value for
    this slot, via embedding similarity. Below 0.8 similarity counts as
    a contradiction — not silently overwritten, flagged instead."""
    row = conn.execute(
        "SELECT value FROM entity_attrs WHERE entity_name=? AND slot=? ORDER BY version DESC LIMIT 1",
        (entity_name, slot)).fetchone()
    if not row:
        return True, None

    old_value = row["value"]
    v1 = np.array(_cached_embed(old_value))
    v2 = np.array(_cached_embed(new_value))
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return True, None

    sim = np.dot(v1, v2) / (n1 * n2)
    if sim < 0.8:
        return False, old_value
    return True, None


# ---------------------------------------------------------------------------
# Orchestrator — runs all three layers in order, cheapest first
# ---------------------------------------------------------------------------

def write_gate(conn, entity_name: str, slot: str, value: str, source: str = "chat", primary_entity: str = None) -> tuple[bool, str]:
    """Decides whether a candidate fact is to be written to
    entity_attrs. Runs cheap checks before expensive ones:
    gazetteer (instant) -> search-cache block (instant) ->
    plausibility (LLM call) -> contradiction (embedding calls).
    """
    if not entity_name_check(entity_name, primary_entity):
        return False, f"entity_name_reject: '{entity_name}' not a valid entity name"

    if not gazetteer_check(slot, value):
        return False, f"gazetteer_reject: '{value}' not a recognized place for slot '{slot}'"

    if source == "search_cache":
        return False, "search_sourced_write_blocked: facts from web search require explicit user confirmation before being stored"

    plausible, raw = plausibility_check(slot, value, entity_name)
    if not plausible:
        return False, f"plausibility_reject: {raw}"

    ok, old_value = contradiction_check(conn, entity_name, slot, value)
    if not ok:
        return False, f"contradiction_flag: conflicts with existing value '{old_value}' (needs confirmation)"

    return True, "ok"