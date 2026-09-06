"""
Seeds ai_history with fixture conversations for eval_retrieval.py's
TEST_SET (see eval_retrieval_v2.py). Run once before running the eval.

Uses save_message() directly — same real pipeline (entity extraction,
LanceDB embedding) that normal chat turns go through — rather than raw
SQL, so the eval actually tests the real retrieval path end to end.

IMPORTANT: run this, then inspect what actually got extracted
(entities / compressed_keywords) before trusting eval numbers. Fictional
names and unusual phrasing are not guaranteed to extract the way you'd
expect — verify, don't assume.
"""

from db import get_db
from storage import save_message

SEED_CONVERSATIONS = [
    # --- location ---
    ("I live in Bengaluru, India.", "Got it, noted you're based in Bengaluru, India."),
    ("I work out of Bengaluru most days.", "Understood, Bengaluru it is."),

    # --- age ---
    ("I'm 24 years old.", "Noted — you're 24."),

    # --- workplace (fictional, so it's testable without needing real facts) ---
    ("I work at a company called Stark Industries.", "Got it, Stark Industries — noted as your workplace."),

    # --- creator / identity (fictional framing, avoids real-identity claims) ---
    ("You were built by a developer named Rin.", "Understood — Rin built me."),

    # --- entity: Edith ---
    ("Edith is that AI system from Spider-Man Homecoming, the glasses Tony Stark left Peter.",
     "Right, Edith — Tony Stark's AI, given to Peter Parker in Homecoming."),

    # --- entity: Jarvis ---
    ("Jarvis is Tony Stark's AI assistant from the Iron Man movies, before Ultron replaced him.",
     "Right, Jarvis — Stark's original AI assistant in the Iron Man films."),

    # --- entity collision: both in one message ---
    ("Jarvis and Edith are both Stark AI systems, but Jarvis came first and Edith was the glasses AI.",
     "Correct — Jarvis predates Edith, and Edith is specifically the Homecoming glasses AI."),
]


def main():
    with get_db() as conn:
        for prompt, response in SEED_CONVERSATIONS:
            save_message(conn, prompt, response, source="chat")
    print(f"Seeded {len(SEED_CONVERSATIONS)} conversations.")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT prompt, entities, compressed_keywords FROM ai_history ORDER BY id DESC LIMIT ?",
            (len(SEED_CONVERSATIONS),)
        ).fetchall()
        print("\n--- Verify extraction before trusting the eval ---")
        for r in reversed(rows):
            print(f"prompt={r['prompt']!r}")
            print(f"  entities={r['entities']!r}")
            print(f"  keywords={r['compressed_keywords']!r}")


if __name__ == "__main__":
    main()
