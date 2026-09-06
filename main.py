import sys
import web_search
from neo import ask_neo, delete_memory, initialise
from file_data import verify
from logger import log_info, log_menu
from rich.console import Console
from rich.text import Text
import subprocess
import time
from web_search import load_saved_backend
import os, atexit
from web_db import init_web
from query_classifier import init_query_classifier
from config import get_context_override, set_context_override, get_response_length_override, set_response_length_override
from hardware_check import get_cached_budget_gb
def init_neo():
    """Called once, before the main loop starts. Restores saved search
    backend, sets up the web-cache db and query classifier, makes sure
    SearXNG is running, and claims the single-instance lock."""
    load_saved_backend()
    init_web()
    init_query_classifier()
    get_cached_budget_gb()
    start_searxng()
    acquire_lock()
LOCK_PATH = os.path.expanduser("~/.neo/neo.lock")
def acquire_lock():
    """Prevents two Neo instances running at once. Writes this process's
    PID to a lock file; if a lock file already exists and that PID is
    still alive, refuses to start."""
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    if os.path.exists(LOCK_PATH):
        with open(LOCK_PATH) as f:
            old_pid = f.read().strip()
        if old_pid and os.path.exists(f"/proc/{old_pid}"):
            print(f"Neo already running (PID {old_pid}). Exiting.")
            sys.exit(1)
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(LOCK_PATH) and os.remove(LOCK_PATH))
def is_ollama_running():
    import httpx
    try:
        resp = httpx.get("http://localhost:11434", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False
def start_ollama():
    if is_ollama_running():
        return
    console.print(Text("Starting Ollama server...", style="dim"))
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(10):
        time.sleep(1)
        if is_ollama_running():
            console.print(Text("Ollama server started.", style="dim"))
            return
    raise RuntimeError("Ollama failed to start after 10 seconds.")
def is_searxng_running() -> bool:
    import httpx
    try:
        resp = httpx.get("http://localhost:8888/search", params={"q": "test", "format": "json"}, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False

def start_searxng():
    if is_searxng_running():
        return
    console.print(Text("Starting SearXNG container...", style="dim"))
    subprocess.run(["docker", "start", "searxng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(15):
        time.sleep(1)
        if is_searxng_running():
            console.print(Text("SearXNG container started.", style="dim"))
            return
    console.print(Text("SearXNG failed to start — web search will be unavailable this session.", style="yellow"))
console = Console()
HELP_TEXT = """
[bold cyan]Neo AI[/bold cyan] - available commands:
  [cyan]/chat[/cyan]     start a conversation with Neo
  [cyan]/delete[/cyan]   delete all AI memory
  [cyan]/status[/cyan]   system status
  [cyan]/search-backend <name> [key][/cyan]  show or switch web search backend
  [cyan]/logs[/cyan]     open log viewer
  [cyan]/list[/cyan]     show recent memory rows with IDs
  [cyan]/forget <id>[/cyan]  delete a specific memory by row id
  [cyan]/backfill-entities[/cyan]  rebuild entity profiles from full history (use after fixing extraction logic)
  [cyan]/consolidate[/cyan]  cluster similar memories into semantic summaries, then run quality eval on them
  [cyan]/set-context <tokens>[/cyan]  show or set a manual context window size (overrides auto-detection)
  [cyan]/set-response-length <tokens>[/cyan]  show or set a max response length (overrides Ollama's default)
  [cyan]/help[/cyan]     show this message
  [cyan]/exit[/cyan]     exit

[dim]Inside /chat, web search triggers automatically for phrases like "search for", "look up", "latest news".[/dim]
"""
def _run_exit_consolidation():
    """Runs memory consolidation once, silently, as the program exits.
    Never blocks exit — if consolidation fails for any reason, it's
    logged and swallowed, not raised, since the user is trying to leave,
    not debug consolidation at that moment."""
    try:
        from consolidation import run_consolidation
        from db import get_db
        with get_db() as conn:
            num_clusters = run_consolidation(conn)
        if num_clusters:
            log_info(f"Exit consolidation: created {num_clusters} cluster(s)", level="INFO", module="CONSOLIDATION")
    except Exception as e:
        log_info(f"Exit consolidation failed (non-blocking): {e}", level="WARNING", module="CONSOLIDATION")


def main()->None:
    """The command loop. Reads a command, dispatches to the matching
    handler, repeats until /exit."""
    start_ollama()
    initialise()
    from config import get_user
    username,is_new=get_user()
    if is_new:
        console.print(f"\n Nice to meet you [bold cyan] {username}[/bold cyan]. I'm Neo, your personal offline AI")
    else:
        console.print(f"\nWelcome back [bold cyan]{username}[/bold cyan]")
    console.print(HELP_TEXT)
    while True:
        try:
            cmd_raw = input("Neo > ").strip()
            if not cmd_raw:
                continue
            parts = cmd_raw.split()
            cmd = parts[0].lower()
            args = parts[1:]
            log_info(f"User entered command: {cmd_raw}", level="INFO", module="MAIN")
            if cmd == "/chat":
                ask_neo()
            elif cmd == "/delete":
                delete_memory()
            elif cmd == "/status":
                verify()
            elif cmd == "/logs":
                log_menu()
            elif cmd == "/search-backend":
                name = args[0] if len(args) > 0 else None
                key = args[1] if len(args) > 1 else None
                if not name:
                    print(f"current backend: {web_search._active_backend_name}")
                else:
                    print(web_search.set_search_backend(name, key))
            elif cmd == "/list":
                from db import get_db
                with get_db() as conn:
                    rows = conn.execute("SELECT id, prompt, timestamp FROM ai_history ORDER BY id DESC LIMIT 10").fetchall()
                for r in rows:
                    console.print(f"[cyan]{r['id']}[/cyan] {r['timestamp']} — {r['prompt'][:60]}")
            elif cmd == "/forget":
                from storage import forget
                if not args:
                    console.print(Text("Usage: /forget <row_id>", style="yellow"))
                else:
                    try:
                        row_id = int(args[0])
                        success = forget(row_id)
                        msg = f"Forgot memory id={row_id}" if success else f"No memory found with id={row_id}"
                        console.print(Text(msg, style="green" if success else "yellow"))
                    except ValueError:
                        console.print(Text("row_id must be an integer", style="yellow"))
            elif cmd == "/consolidate":
                from consolidation import run_consolidation
                from consolidation_eval import eval_consolidation
                from db import get_db
                console.print(Text("Consolidating memory into semantic summaries...", style="dim"))
                with get_db() as conn:
                    num_clusters = run_consolidation(conn)
                if num_clusters == 0:
                    console.print(Text(
                        "0 clusters created. Either there's not enough unconsolidated history yet "
                        "(needs at least min_cluster_size=3 similar rows), or nothing was similar "
                        "enough to cluster together at the current eps setting.", style="yellow"))
                else:
                    console.print(Text(f"Created {num_clusters} semantic cluster(s). Running eval...", style="dim"))
                    with get_db() as conn:
                        results = eval_consolidation(conn)
                    for r in results:
                        console.print(Text(str(r), style="white"))
                    passed = sum(
                        1 for r in results
                        if r["coherence"] > 0.7 and r["alignment"] > 0.7 and r["faithfulness_avg"] > 0.7
                    )
                    console.print(Text(f"{passed}/{len(results)} clusters passed thresholds (>0.7 on all three)", style="green" if passed == len(results) else "yellow"))
            elif cmd == "/backfill-entities":
                confirm = input(
                    "This rewrites entity_profile and entity_attrs from your ENTIRE history. "
                    "Can take a while on large histories. Continue? (yes/no): "
                ).strip().lower()
                if confirm != "yes":
                    console.print(Text("Cancelled", style="yellow"))
                else:
                    from storage import backfill_entity_profiles
                    console.print(Text("Rebuilding entity profiles from history...", style="dim"))
                    total, errors = backfill_entity_profiles()
                    msg = f"Backfill complete: {total} rows processed, {errors} errors"
                    console.print(Text(msg, style="green" if errors == 0 else "yellow"))
                    log_info(msg, level="INFO", module="MAIN")
            elif cmd == "/set-context":
                if not args:
                    current = get_context_override()
                    if current:
                        console.print(f"Manual context override: {current} tokens")
                    else:
                        console.print("No manual override set. Neo is auto-picking context window per model.")
                    console.print(Text("Usage: /set-context <tokens>", style="dim"))
                else:
                    try:
                        tokens = int(args[0])
                        if tokens <= 0:
                            console.print(Text("Context size must be a positive number.", style="yellow"))
                        else:
                            set_context_override(tokens)
                            console.print(Text(f"Context window set to {tokens} tokens. Applies from your next message.", style="green"))
                            log_info(f"Context override set to {tokens}", level="INFO", module="MAIN")
                    except ValueError:
                        console.print(Text("tokens must be a whole number, e.g. /set-context 4096", style="yellow"))
            elif cmd == "/set-response-length":
                if not args:
                    current = get_response_length_override()
                    if current:
                        console.print(f"Manual response length override: {current} tokens")
                    else:
                        console.print("No manual override set. Neo is using Ollama's default response length.")
                    console.print(Text("Usage: /set-response-length <tokens>", style="dim"))
                else:
                    try:
                        tokens = int(args[0])
                        if tokens <= 0:
                            console.print(Text("Response length must be a positive number.", style="yellow"))
                        else:
                            set_response_length_override(tokens)
                            console.print(Text(f"Response length set to {tokens} tokens. Applies from your next message.", style="green"))
                            log_info(f"Response length override set to {tokens}", level="INFO", module="MAIN")
                    except ValueError:
                        console.print(Text("tokens must be a whole number, e.g. /set-response-length 1000", style="yellow"))
            elif cmd == "/help":
                console.print(HELP_TEXT)
            elif cmd == "/exit":
                console.print(Text("Consolidating memory before exit...", style="dim"))
                _run_exit_consolidation()
                log_info("Program exited", level="INFO", module="MAIN")
                sys.exit()
            else:
                console.print(Text(f"Unknown command '{cmd}'. Type /help to see available commands.", style="yellow"))
                log_info(f"Unknown command: {cmd}", level="WARNING", module="MAIN")
        except KeyboardInterrupt:
            console.print(Text("\nUse /exit to exit.", style="yellow"))
        except Exception as e:
            console.print(Text("An error occurred.", style="bold red"))
            log_info(f"ERROR: {e}", level="ERROR", module="MAIN")
if __name__ == "__main__":
    init_neo()
    main()