import re
from logger import ai_log_info
import yake
# ---------------------------------------------------------------------------
# Keyword extraction setup (yake library)
# ---------------------------------------------------------------------------

_kw_extractor_2 = yake.KeywordExtractor(lan="en", n=2, top=10)  # 2-word phrases
_kw_extractor_1 = yake.KeywordExtractor(lan="en", n=1, top=10)  # single words


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# gpt4, rtx4050, v2  OR  NLP, API, HTTP (2+ capital letters)
_TECH_TERM_RE = re.compile(r"\b[a-zA-Z]+\d+[a-zA-Z]*\b|\b[A-Z]{2,}\b")

# Capitalized word(s) in a row — crude name/place detector, e.g. "New York"
_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")

# 2026-01-05, 5/1/2026, Jan 5, 2026
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4}\b"
)

_PERSONAL_FACT_RE = re.compile(
    r"\b\d{1,2}\s?(?:yr|year)s?\s?old\b|\bfrom\s+[a-zA-Z]+\b|\blive[sd]?\s+in\s+[a-zA-Z]+\b|\bbased\s+in\s+[a-zA-Z]+\b",
    re.IGNORECASE)

_AGE_HINT_RE = re.compile(r'\b\d{1,2}\b')

# Pulls just the number out of "25 years old" -> "25"
_AGE_RE = re.compile(r'\b(\d{1,2})\s?(?:years?|yrs?)\s?old\b', re.IGNORECASE)

_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|im|my|me)\b", re.IGNORECASE)
# Which trigger words suggest which fact slot, used in _extract_attrs_from_doc
_ATTR_PATTERNS = {
    "transferred": ["transfer", "relocat"],
    "moved": ["move", "shift"],
    "location": ["live", "based", "from"],
    "workplace": ["work", "employ"],
    "age": ["year", "yr", "old", "age"],
}
_NEGATION_VALUES = {"none", "not there anymore", "no longer", "n/a", "null", ""}

_NEGATION_PATTERNS=(
    re.compile(r'^\s*not\s+(there\s)?anymore\s*$', re.IGNORECASE),
    re.compile(r'^\s*no\s+longer\s*(there|applicable)?\s*$',re.IGNORECASE),
    re.compile(r'^\s*not\s+(applicable|disclosed|specified|stated|given)?\s*$',re.IGNORECASE)
)

def _is_negation_value(v: str) -> bool:
    v_lower = v.strip().lower()
    if v_lower in _NEGATION_VALUES:
        return True
    return any(p.match(v_lower)for p in _NEGATION_PATTERNS)
# ---------------------------------------------------------------------------
# spaCy setup — real grammar-aware NLP, with a regex fallback if unavailable
# ---------------------------------------------------------------------------

try:
    import spacy
    _NLP = spacy.load("en_core_web_md")
except (ImportError, OSError):
    _NLP = None
    ai_log_info("spaCy unavailable, falling back to regex entity extraction", level="WARNING", module="MEMORY")


# ---------------------------------------------------------------------------
# Keyword extraction (used for embeddings / search, not entity facts)
# ---------------------------------------------------------------------------

def extract_keywords(text):
    """Pull out up to 10 representative keywords/phrases from text, using
    yake. Tries 2-word phrases first, falls back to single words if none
    found. Also appends any tech-looking tokens (gpt4, v2, etc.) yake
    might have missed."""
    if not text or not text.strip():
        return ""

    keywords = _kw_extractor_2.extract_keywords(text)
    if not keywords:
        keywords = _kw_extractor_1.extract_keywords(text)
    kw_list = [kw for kw, score in keywords]  # yake's relevance score is discarded here

    tech_tokens = re.findall(r'\b[a-z]+\d+\b', text.lower())
    for tok in tech_tokens:
        if tok not in kw_list:
            kw_list.append(tok)

    return ", ".join(kw_list)


# ---------------------------------------------------------------------------
# Fallback entity extraction — only used when spaCy isn't installed
# ---------------------------------------------------------------------------

def _regex_entities(text: str) -> set:
    """Crude entity detection using only regex, for when spaCy is
    unavailable. Lower quality than spaCy but has zero dependencies."""
    ents = set()
    ents.update(m.group() for m in _CAPWORD_RE.finditer(text))
    ents.update(m.group() for m in _DATE_RE.finditer(text))
    ents.update(m.group() for m in _TECH_TERM_RE.finditer(text))
    ents.update(m.group() for m in _PERSONAL_FACT_RE.finditer(text))
    ents.discard("")
    return ents


# ---------------------------------------------------------------------------
# Structured fact extraction (spaCy required)
# ---------------------------------------------------------------------------

def _extract_attrs_from_doc(doc, primary_entity=None, session_last_entity=None):
    """Walk a spaCy-parsed document sentence by sentence. For each
    sentence, work out who it's about (subject tracking across pronouns),
    then pull out any location/workplace/age/mentions facts it states
    about that subject."""
    profiles = {}
    last_entity = session_last_entity

    for sent in doc.sents:
        subjects = [t for t in sent if t.dep_ in ("nsubj", "nsubjpass") and t.pos_ == "PROPN"]
        pronoun_subj = [t for t in sent if t.dep_ in ("nsubj", "nsubjpass") and t.text.lower() in ("he", "she", "they")]
        first_person = [t for t in sent if t.dep_ in ("nsubj", "nsubjpass") and t.text.lower() == "i"]

        # Work out who this sentence is about:
        if subjects:
            # Named subject ("Manoj went to...") -> remember for next sentence
            last_entity = subjects[0].text
            subj_name = last_entity
        elif pronoun_subj and last_entity:
            # Pronoun subject ("He went to...") -> assume same as last named subject
            subj_name = last_entity
        elif first_person and (primary_entity or last_entity):
            # "I went to..." -> resolve to the known user
            subj_name = primary_entity or last_entity
        else:
            continue  # no identifiable subject, skip this sentence

        attrs_found = {}
        sent_ents = {ent.label_: ent.text for ent in sent.ents}

        # Location: place mentioned + a trigger verb (live/moved/from/etc.)
        _PLACE_SLOTS_FOR_GPE = ("location", "moved", "transferred")

        gpe_ents = [ent.text for ent in sent.ents if ent.label_ in ("GPE", "LOC")]
        if gpe_ents:
            loc = gpe_ents[0]
            matched_slot = None
            for slot in _PLACE_SLOTS_FOR_GPE:
                verbs = _ATTR_PATTERNS[slot]
                if any(any(v in t.lemma_ for v in verbs) for t in sent):
                    matched_slot = slot
                    break
            if matched_slot and loc:
                attrs_found[matched_slot] = loc

        # Workplace: organization mentioned + a work-related trigger verb
        if "workplace" not in attrs_found:
            org_ents = [ent.text for ent in sent.ents if ent.label_ in ("ORG", "PRODUCT", "FAC")]
            if org_ents and any(v in t.lemma_ for t in sent for v in _ATTR_PATTERNS["workplace"]):
                attrs_found["workplace"] = org_ents[0]

        # Age: pulled directly via regex
        age_match = _AGE_RE.search(sent.text)
        if age_match:
            attrs_found["age"] = age_match.group(1)

        # Mentions: any other named person in the sentence, besides the subject
        if "PERSON" in sent_ents:
            for ent in sent.ents:
                if ent.label_ == "PERSON" and ent.text != subj_name:
                    attrs_found.setdefault("mentions", []).append(ent.text)

        if attrs_found:
            existing = profiles.setdefault(subj_name, {})
            for k, v in attrs_found.items():
                if k == "mentions":
                    existing.setdefault("mentions", [])
                    existing["mentions"] = list(set(existing["mentions"] + v))
                else:
                    existing[k] = v

    return profiles, last_entity

def _llm_fact_fallback(text: str, primary_entity: str) -> dict:
    """When spaCy's structural extraction finds nothing, ask the LLM
    directly whether the message states personal facts. Only called
    for first-person messages that spaCy already missed."""
    from neo_ollama import _ollama, get_active_model
    prompt = (
    f"Given this message from {primary_entity}, extract any personal facts "
    f"explicitly stated about themselves (location, age, or workplace only). "
    f"Never infer or guess — only what is directly stated.\n"
    f"Evaluate each of the three slots (location, age, workplace) INDEPENDENTLY. "
    f"For each slot, ask: did this specific message explicitly state or explicitly "
    f"deny/correct THIS slot? If the message is silent on a slot, omit that slot "
    f"entirely from your JSON — do not include it as NONE just because a different "
    f"slot was corrected in the same message.\n"
    f"Only set a slot to the literal string \"NONE\" if the person explicitly denied "
    f"or corrected THAT SPECIFIC slot in this message.\n"
    f"Reply with ONLY valid JSON, nothing else: "
    f'{{"location": "...", "age": "...", "workplace": "..."}} '
    f"— include only the slots actually stated or explicitly corrected, omit the rest. "
    f"Reply {{}} if none.\n\n"
    f"Message: {text}"
)
    try:
        resp = _ollama.chat(
            model=get_active_model(),
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 4096},
        )
        raw = resp["message"]["content"].strip()
        print("DEBUG raw LLM output:",repr(raw))
        import json
         # Model sometimes keeps generating after the JSON object (hallucinated
        # continuation). Extract just the first {...} block instead of trying
        # to parse the whole response.
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not match:
            ai_log_info(f"LLM_FACT_FALLBACK: no JSON object found in output: {raw!r}", level="WARNING", module="EXTRACTION")
            return {}
        json_str = match.group()
        data = json.loads(json_str)
        if not isinstance(data, dict):
            # Valid JSON, but not a dict (e.g. model returned a list or string)
            ai_log_info(
                f"LLM_FACT_FALLBACK: parsed JSON but not a dict, got {type(data).__name__}: {raw!r}",
                level="WARNING", module="EXTRACTION")
            return {}

        result = {}
        for k, v in data.items():
            if k not in ("location", "age", "workplace") or not v:
                continue
            # Model doesn't reliably output the literal string "NONE" for
            # corrections — it paraphrases ("NOT THERE ANYMORE", "none"
            # lowercase, etc). Normalize anything that reads as a negation
            # into the canonical "NONE" before it goes any further.
            result[k] = "NONE" if _is_negation_value(str(v)) else v
        print("DEBUG parsed result:", result)
        return result
    except Exception as e:
        raw_snippet = raw if "raw" in locals() else "<no response received>"
        ai_log_info(
            f"LLM_FACT_FALLBACK: exception during fact extraction — {e}. Raw output: {raw_snippet!r}",
            level="WARNING", module="EXTRACTION")
        return {}
#-----------------------------------------------------------
# Entry point — called from storage.py's save_message
# ---------------------------------------------------------------------------

def process_message(text, primary_entity=None, session_last_entity=None):
    """Returns (entities_string, profiles_dict, updated_last_entity).
    ...
    NOTE: the LLM fact-fallback path (_llm_fact_fallback) is currently
    disabled. It was found to fabricate personal facts (wrong ages,
    invented locations, entire fictional conversations) with enough
    frequency to be unsafe to ship — see KNOWN_ISSUES.md. Structural
    (spaCy-based) extraction via _extract_attrs_from_doc still runs
    and catches facts stated in a directly parseable grammatical form;
    only the LLM-guess fallback for messages that structural extraction
    misses has been turned off.
    """
    if not text or not text.strip():
        return "", {}, session_last_entity

    if _NLP is None:
        return ", ".join(sorted(_regex_entities(text))), {}, session_last_entity

    doc = _NLP(text)
    ents = {
        ent.text.strip() for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "PRODUCT", "EVENT", "FAC"}
    }
    ents.update(m.group() for m in _TECH_TERM_RE.finditer(text))
    ents.update(m.group() for m in _PERSONAL_FACT_RE.finditer(text))
    ents.discard("")

    profiles, new_last_entity = _extract_attrs_from_doc(
        doc, primary_entity=primary_entity, session_last_entity=session_last_entity
    )
    return ", ".join(sorted(ents)), profiles, new_last_entity
def _keyword_supports_slot(text: str, slot: str) -> bool:
    """True if the message contains a trigger word for this slot.
    Used to guard against the LLM marking NONE on slots the message
    never actually touched."""
    text_lower = text.lower()
    triggers = _ATTR_PATTERNS.get(slot)
    if not triggers:
        # Unknown slot with no defined triggers — err on the side of
        # rejecting the NONE rather than allowing it through unchecked.
        return False
    return any(trig in text_lower for trig in triggers)