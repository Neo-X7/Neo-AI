# Known Issues

Honest list of what's broken, disabled, or limited in this build. Nothing
here is hidden — if you're reading this before relying on Neo, read this
whole file first.

## Fact extraction

- **`_llm_fact_fallback` is disabled.** When structural (spaCy-based)
  extraction doesn't catch a personal fact stated in an unusual grammatical
  form, Neo used to fall back to asking the LLM directly to infer the fact.
  This was found to fabricate facts with meaningful frequency — inventing
  ages, locations, workplaces, and in some cases entire fictional
  conversations that were never in the input. The fallback call itself is
  still in the codebase (`extraction.py`) but is never invoked. Only
  facts stated in a directly parseable grammatical form get captured now;
  facts phrased unusually may simply not be picked up. This is a
  deliberate tradeoff — silently missing a fact is safer than silently
  storing a wrong one.

- **Location extraction drops secondary place details.** "I live in
  Bengaluru, India" stores `location: Bengaluru`, not `Bengaluru, India` —
  only the first GPE entity spaCy tags per sentence is kept. Minor,
  measured impact: 1 miss out of 23 cases in the retrieval eval.

## Query understanding

- **`is_self_query` can misclassify imperative phrasing that names another
  person.** "tell me what Ravi said" contains "me" and matches the
  imperative self-request pattern, and can be misread as a query about the
  user rather than about Ravi. A regex-based disambiguation attempt was
  tried and reverted — it fixed this case but broke a different, already-
  correct case ("what's my location, did I mention Ravi's place?",
  which must resolve to the user). Properly separating these needs real
  dependency parsing, not string matching. Not fixed; edge case, not the
  common path.

- **`is_time_sensitive`'s example set doesn't cover in-progress standings
  language.** Queries like "who won the F1 constructors championship"
  (asked mid-season, before a final result exists) score below the
  time-sensitivity threshold and get a single search instead of the
  broader 3-variant fan-out. The single search still worked in testing,
  but coverage for this phrasing pattern is thin.

## Model / hardware

- **No model-tier fallback.** Neo targets `llama3.1:8b` only. Earlier
  versions had a fallback chain to smaller models (`phi3`, `qwen2.5:1.5b`)
  for lower-spec hardware. That fallback was removed after `phi3`
  consistently proved unreliable for grounded synthesis and fact
  extraction — a broken degrade path is worse than no degrade path. On
  hardware that can't run an 8B model, Neo will fail to start rather than
  silently degrading to a worse model.

- **`write_gate`'s plausibility check is soft on category-mismatch edge
  cases.** It correctly rejects clearly implausible facts, but can pass
  borderline cases (e.g. a genre name given as a workplace) that a
  stricter human reviewer would reject. Same underlying cause as the
  `_llm_fact_fallback` issue — local model instruction-following isn't
  perfectly reliable on this exact task shape.

## Search pipeline

- **`search_verify.py` (result fact-checking before caching) has been
  removed entirely.** Search results are now cached and used unfiltered.
  Grounding against outright fabrication is still enforced at the prompt
  level in `search_completion()` ("answer only from what's below, say so
  if it's not there"), but this does not protect against a genuinely
  wrong or unreliable source being treated as trustworthy just because it
  came back from search.
- **`search_completion()` grounding — fixed and verified.** An earlier
  version of the system prompt allowed the model to fabricate an answer
  disconnected from the actual search results in at least one confirmed
  case (asking about the current OpenAI model returned a fabricated
  "Llama 2, 2023" answer despite correct, specific results being present
  in context). The prompt was rewritten to explicitly forbid using prior
  knowledge and require the model to point to a specific supporting
  result before answering. Verified against three cases post-fix:
  relevant results present (correctly cited the right answer), zero
  results (clean refusal via an early guard), and irrelevant results
  present (correctly recognized the mismatch and refused rather than
  falling back on training data). Not proven bulletproof against every
  possible query, but the specific failure mode that was found is fixed.