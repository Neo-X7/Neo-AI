# Known Issues: v1.9.5

These limits were found by the 65-turn stress test and by real chat sessions. They are documented rather than fixed, because v1.9.5 is the final version.

## Memory

- **Comparative preferences are not extracted.** "I like X more than Y" produces no fact.
- **Some sentences split into fragments.** "I want to work in machine learning in Europe" can be stored as two facts. The second can then retract the first, which loses detail.
- **Extractor and gate depend on the model.** Both run on `llama3.1` 8B and can misjudge unusual wording. Results can differ between runs on different phrasings of the same fact.

## Retrieval

- **The distance cutoff is loose at larger fact counts.** `max_distance=450` was tuned on 8 facts. With about 50 facts, an unrelated question (for example about a country you once lived in) can pull in a loosely related fact.
- **Two distance measures are in use.** Retrieval for chat uses L2 distance, and candidate lookup for the write gate uses cosine similarity. Both were tuned separately and work, but the two scales are not comparable.
- **Short or very general queries match poorly.** A question like "what languages do I speak?" can rank learning-related facts above speaking-related ones. Retrieval returns the top 5 facts to compensate.

## Web search

- **Follow-up searches lose context.** The raw message is the search query, so a short follow-up has no memory of what an earlier message meant.
- **Search answers are only as deep as the snippets.** Neo reads titles and short snippets, not full pages.
- **Failures are silent to the user.** If SearXNG is down, Neo carries on without results. The error is written to `error.log`, not `ai_memory.log`.

## Logging

- **`ai_log_info` ignores its `module` argument.** Every line in `ai_memory.log` is tagged `[ai_memory]`.