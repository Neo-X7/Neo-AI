import os
import logging
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

console = Console()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "error.log")
AI_LOG_PATH = os.path.join(BASE_DIR, "ai_memory.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

ai_logger = logging.getLogger("ai_memory")
ai_logger.setLevel(logging.DEBUG)
if not ai_logger.handlers:
    ai_handler = logging.FileHandler(AI_LOG_PATH)
    ai_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ai_logger.addHandler(ai_handler)


def log_info(message, level="INFO", module="system"):
    """General system/error logging. Written to error.log."""
    tagged = f"[{module}] {message}"
    getattr(logger, level.lower())(tagged)


def ai_log_info(message, level="INFO", module="system"):
    """Memory-system-specific logging (extraction, write_gate, retrieval,
    etc). Written to ai_memory.log, kept separate from general errors so
    memory behavior can be reviewed on its own."""
    tagged = f"[ai_memory] {message}"
    getattr(ai_logger, level.lower())(tagged)


def _print_log_file(path: str, empty_message: str, filter_type: str = "all") -> None:
    """Reads a log file and prints each line color-coded by severity:
    red for ERROR, yellow for WARNING, white otherwise. filter_type
    restricts output to only that severity ('error' or 'warning'),
    or 'all' to print everything."""
    if not os.path.exists(path):
        console.print(Text(empty_message, style="yellow"))
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        console.print(Text(empty_message, style="yellow"))
        return

    filter_type = filter_type.lower()
    printed_any = False
    for line in lines:
        line = line.strip()
        if filter_type == "error" and "ERROR" not in line:
            continue
        if filter_type == "warning" and "WARNING" not in line:
            continue

        printed_any = True
        if "ERROR" in line:
            console.print(Text(line, style="red"))
        elif "WARNING" in line:
            console.print(Text(line, style="yellow"))
        else:
            console.print(Text(line, style="white"))

    if not printed_any:
        console.print(Text(f"No '{filter_type}' entries found", style="yellow"))


def log_his(filter_type="all"):
    """Prints the general error log. filter_type: 'all', 'error', or 'warning'."""
    _print_log_file(LOG_PATH, "No log records found", filter_type)


def ai_log_his(filter_type="all"):
    """Prints the AI memory log. filter_type: 'all', 'error', or 'warning'."""
    _print_log_file(AI_LOG_PATH, "No AI memory logs found", filter_type)


def clear_log():
    """Wipes error.log."""
    try:
        with open(LOG_PATH, "w", encoding="utf-8"):
            pass
        console.print(Text("Logs cleared successfully", style="green"))
        log_info("Logs cleared by user", level="WARNING", module="LOG")
    except Exception as e:
        console.print(Text("Failed to clear logs", style="bold red"))
        log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")


def clear_ai_log():
    """Wipes ai_memory.log. Previously there was no way to do this
    without touching the file outside Neo."""
    try:
        with open(AI_LOG_PATH, "w", encoding="utf-8"):
            pass
        console.print(Text("AI memory logs cleared successfully", style="green"))
        ai_log_info("AI memory logs cleared by user", level="WARNING", module="LOG")
    except Exception as e:
        console.print(Text("Failed to clear AI memory logs", style="bold red"))
        log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")


def log_menu():
    """Interactive log viewer, reached via /logs in main.py."""
    console.print(Text("Log viewer — /all [error|warning] \n /clear\n /ai [error|warning]\n /clear-ai\n /exit\n", style="magenta"))
    while True:
        try:
            raw = input("logs > ").strip().lower()
            parts = raw.split()
            cmd = parts[0] if parts else ""
            arg = parts[1] if len(parts) > 1 else "all"

            if cmd == "/all":
                log_his(arg)
            elif cmd == "/clear":
                clear_log()
            elif cmd == "/ai":
                ai_log_his(arg)
            elif cmd == "/clear-ai":
                clear_ai_log()
            elif cmd == "/exit":
                return
            else:
                console.print(Text("Unknown command. Use /all, /clear, /ai, /clear-ai, /back", style="yellow"))
        except KeyboardInterrupt:
            return
        except Exception as e:
            console.print(Text("An error occurred", style="bold red"))
            log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")