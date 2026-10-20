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

LOG_HELP = """[cyan]/all[/cyan]              system logs (error.log)
[cyan]/all error[/cyan]        system logs, errors only
[cyan]/all warning[/cyan]      system logs, warnings only
[cyan]/ai[/cyan]               memory event logs (ai_memory.log)
[cyan]/ai error[/cyan]         memory logs, errors only
[cyan]/ai warning[/cyan]       memory logs, warnings only
[cyan]/clear[/cyan]            wipe system logs
[cyan]/clear-ai[/cyan]         wipe memory event logs
[cyan]/help[/cyan]             show this menu
[cyan]/exit[/cyan]             return to the main menu"""

def log_info(message, level="INFO", module="system"):
    """Write a message to error.log at the given level, tagged with [module]."""
    tagged = f"[{module}] {message}"
    getattr(logger, level.lower())(tagged)
def ai_log_info(message, level="INFO", module="system"):
    """Write a message to ai_memory.log at the given level. The module argument is currently ignored; every line is tagged [ai_memory]."""
    tagged = f"[ai_memory] {message}"
    getattr(ai_logger, level.lower())(tagged)
def _print_log(path, empty_message, filter_type="all"):
    """Print a log file coloured by level. filter_type 'error' or 'warning' keeps only that level."""
    if not os.path.exists(path):
        console.print(Text(empty_message, style="yellow"))
        return
    with open(path, "r", encoding="utf-8") as g:
        lines = g.readlines()
    shown = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if filter_type == "error" and " - ERROR - " not in line:
            continue
        if filter_type == "warning" and " - WARNING - " not in line:
            continue
        if " - ERROR - " in line:
            console.print(Text(line, style="red"))
        elif " - WARNING - " in line:
            console.print(Text(line, style="yellow"))
        else:
            console.print(Text(line, style="white"))
        shown += 1
    if shown == 0:
        console.print(Text(empty_message, style="yellow"))
def log_his(filter_type="all"):
    """Print error.log to the console, coloured by level. filter_type can be 'all', 'error' or 'warning'."""
    _print_log(LOG_PATH, "No log records found", filter_type)
def ai_log_his(filter_type="all"):
    """Print ai_memory.log to the console, coloured by level. filter_type can be 'all', 'error' or 'warning'."""
    _print_log(AI_LOG_PATH, "No AI memory logs found", filter_type)
def clear_log():
    """Empty error.log. ai_memory.log is not touched."""
    try:
        with open(LOG_PATH, "w", encoding="utf-8"):
            pass
        console.print(Text("System logs cleared successfully", style="green"))
        log_info("Logs cleared by user", level="WARNING", module="LOG")
    except Exception as e:
        console.print(Text("Failed to clear system logs", style="bold red"))
        log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")
def clear_ai_log():
    """Empty ai_memory.log. error.log is not touched."""
    try:
        with open(AI_LOG_PATH, "w", encoding="utf-8"):
            pass
        console.print(Text("Memory event logs cleared successfully", style="green"))
        log_info("AI memory logs cleared by user", level="WARNING", module="LOG")
    except Exception as e:
        console.print(Text("Failed to clear memory event logs", style="bold red"))
        log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")
def show_log_help():
    """Print the log viewer command list in a panel."""
    console.print(Panel(LOG_HELP, title="Neo Log Viewer", border_style="magenta", padding=(1, 2)))
def log_menu():
    """Interactive log viewer. Type /help to see the commands, /exit to leave."""
    show_log_help()
    while True:
        try:
            parts = input("logs > ").strip().lower().split()
            if not parts:
                continue
            name = parts[0]
            arg = parts[1] if len(parts) > 1 else "all"
            if name in ("/all", "/ai") and arg not in ("all", "error", "warning"):
                console.print(Text("Filter must be 'error' or 'warning'. Type /help to see the commands.", style="yellow"))
                continue
            if name == "/all":
                log_his(arg)
            elif name == "/ai":
                ai_log_his(arg)
            elif name == "/clear":
                clear_log()
            elif name == "/clear-ai":
                clear_ai_log()
            elif name == "/help":
                show_log_help()
            elif name in ("/exit", "/back"):
                return
            else:
                console.print(Text("Unknown command. Type /help to see the commands.", style="yellow"))
        except KeyboardInterrupt:
            return
        except Exception as e:
            console.print(Text("An error occurred", style="bold red"))
            log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")