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
    tagged = f"[{module}] {message}"
    getattr(logger, level.lower())(tagged)
def ai_log_info(message, level="INFO", module="system"):
    tagged = f"[ai_memory] {message}"
    getattr(ai_logger, level.lower())(tagged)
def log_his(filter_type="all"):
    if not os.path.exists(LOG_PATH):
        console.print(Text("No log records found", style="yellow"))
        return
    with open(LOG_PATH, "r", encoding="utf-8") as g:
        lines = g.readlines()
        if not lines:
            console.print(Text("No log records found", style="yellow"))
            return
        for line in lines:
            line = line.strip()
            if "ERROR" in line:
                console.print(Text(line, style="red"))
            elif "WARNING" in line:
                console.print(Text(line, style="yellow"))
            else:
                console.print(Text(line, style="white"))
def ai_log_his():
    if not os.path.exists(AI_LOG_PATH):
        console.print(Text("No AI memory logs found", style="yellow"))
        return
    with open(AI_LOG_PATH, "r", encoding="utf-8") as g:
        lines = g.readlines()
        if not lines:
            console.print(Text("No AI memory logs found", style="yellow"))
            return
        for line in lines:
            line = line.strip()
            if "ERROR" in line:
                console.print(Text(line, style="red"))
            elif "WARNING" in line:
                console.print(Text(line, style="yellow"))
            else:
                console.print(Text(line, style="white"))
def clear_log():
    try:
        with open(LOG_PATH, "w", encoding="utf-8"):
            pass
        console.print(Text("Logs cleared successfully", style="green"))
        log_info("Logs cleared by user", level="WARNING", module="LOG")
    except Exception as e:
        console.print(Text("Failed to clear logs", style="bold red"))
        log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")
def log_menu():
    console.print(Text("Log viewer — /all, /clear, /ai, /back", style="magenta"))
    while True:
        try:
            cmd = input("logs > ").strip().lower()
            if cmd == "/all":
                log_his("all")
            elif cmd == "/clear":
                clear_log()
            elif cmd == "/ai":
                ai_log_his()
            elif cmd == "/back":
                return
            else:
                console.print(Text("Unknown command. Use /all, /clear, /ai, /back", style="yellow"))
        except KeyboardInterrupt:
            return
        except Exception as e:
            console.print(Text("An error occurred", style="bold red"))
            log_info(f"ERROR : {e}", level="CRITICAL", module="LOG")