from db import get_db
from storage import save_message
from entity_memory import get_active_attrs, get_primary_entity, create_entity_tables
from logger import ai_log_info
from neo_ollama import search_completion,chat
from extraction import extract_keywords
from retrieval import retrieve_similar
from config import get_user
from rich.console import Console
from rich.text import Text
from lancedb_store import clear_all_vectors
from query_classifier import wants_search
from neo_search_pipeline import search_and_learn
import traceback
console = Console()


def initialise():
    """Called once at program start. Makes sure every table Neo needs
    exists before anything else runs."""
    with get_db() as conn:
        create_entity_tables(conn)

_session_history = []

def ai_history_read() -> list:
    """Returns this session's conversation so far, most recent last.
    Capped at last 5 turns. Session-scoped (resets on restart) so old,
    unrelated conversations don't bleed into new ones."""
    return _session_history[-5:]


def ask_neo() -> None:
    """Main chat loop: read input, gather context (history + known facts
    + retrieved memories + web search if triggered), get a response,
    save the exchange, repeat until /exit."""
    username, _ = get_user()
    console.print(Text(f"You are now chatting with Neo. Type /exit to stop.", style="dim"))

    while True:
        history = ai_history_read()
        prompt = input(f" {username}>").strip()

        if not prompt:
            console.print(Text("Prompt cannot be empty.", style="bold red"))
            ai_log_info("Prompt was empty", level="WARNING", module="AI")
            continue

        if len(prompt) > 2000:
            console.print(Text("Prompt too long. Keep it under 2000 characters.", style="bold red"))
            ai_log_info("Prompt Length exceeded above 2000 characters", level="WARNING", module="AI")
            continue

        if prompt == "/exit":
            break

        try:
            compressed = extract_keywords(prompt)

            with get_db() as conn:
                primary_entity = get_primary_entity(conn)
                active_attrs = get_active_attrs(conn, primary_entity) if primary_entity else {}

            needs_search=wants_search(prompt)

            search_results = []
            if needs_search:
                console.print(Text(f"Searching the web...",style="dim"))
                with get_db() as conn:
                    search_results = search_and_learn(conn,prompt, primary_entity)
                    if not search_results:
                        console.print(Text("No usable search results found.", style="yellow"))

            similar = retrieve_similar(compressed, raw_query=prompt) if compressed else []
            if search_results:
                response=search_completion(prompt, search_results)
            else:
                response=chat(
                    username, prompt, history, similar, active_attrs,
                    has_search_results=False,
                    search_results=None,
                    search_attempted=needs_search)

            if search_results:
                source_tag = "[web]"
            elif similar and similar[0]["id"] != -1:
                source_tag = "[memory]"
            else:
                source_tag = "[model]"

            with get_db() as conn:
                save_message(conn, prompt, response, source="search_cache" if search_results else "chat")
                _session_history.append({"prompt": prompt, "response": response})

            console.print(Text(f"{source_tag} {response}", style="cyan"))
        except Exception as e:
            traceback.print_exc()
            ai_log_info(f" ERROR : {e}", level="ERROR", module="MEMORY")


def delete_memory():
    """The /delete command. Double-confirms before wiping ai_history and
    all vector data — this is irreversible."""
    try:
        confirm = input("Delete all AI memory? (yes/no): ").strip().lower()
        if confirm == "no":
            console.print(Text("Cancelled", style="yellow"))
            ai_log_info("User entered no ", level="INFO", module="AI")
            return
        elif confirm == "yes":
            console.print(Text("Command Accepted", style="green"))
            ai_log_info("User entered yes", level="INFO", module="AI")
        else:
            console.print(Text("Invalid Command", style="bold red"))
            ai_log_info("User entered Invalid Input", level="INFO", module="AI")
            return

        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ai_history").fetchone()[0]
        console.print(Text(f"This will permanently delete {count} memory record(s)", style="yellow"))

        input_1 = input("Are you sure? (yes/no): ").strip().lower()
        if input_1 == "yes":
            with get_db() as conn:
                conn.execute("DELETE from entity_mentions")
                conn.execute("DELETE from entity_attrs")
                conn.execute("DELETE from entity_profile")
                conn.execute("DELETE FROM ai_history")
            clear_all_vectors()
            console.print(Text("Memory cleared successfully", style="green"))
            ai_log_info("All AI memory deleted by user", level="INFO", module="AI")
        elif input_1 == "no":
            console.print(Text("Cancelled", style="yellow"))
            ai_log_info("User cancelled the deletion process", level="INFO", module="AI")
        else:
            console.print(Text("Invalid Input", style="bold red"))
    except Exception as e:
        console.print(Text("Failed to delete memory", style="bold red"))
        ai_log_info(f"Failed to delete memory: {e}", level="ERROR", module="AI")