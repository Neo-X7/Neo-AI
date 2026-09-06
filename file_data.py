import os
from rich.console import Console
from rich.table import Table
from rich.text import Text
from logger import log_info
from db import get_db
from hardware_check import get_vram_gb, get_ram_gb, get_installed_models
from neo_ollama import get_active_model
console = Console()
def verify() -> None:
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(BASE_DIR, "ai_history.db")
        error_db_path = os.path.join(BASE_DIR, "error.log")
        log_path = os.path.join(BASE_DIR, "ai_memory.log")
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ai_history").fetchone()[0]
        table = Table(title="Neo System Status")
        table.add_column("Item", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        table.add_row("Database", "✓ Found" if os.path.exists(db_path) else "✗ Missing", db_path)
        table.add_row("AI Chats", str(count), "chats in database")
        table.add_row("AI Log File", "✓ Found" if os.path.exists(log_path) else "✗ Missing", log_path)
        table.add_row("Error Log", "✓ Found" if os.path.exists(error_db_path) else "✗ Missing", error_db_path)
        table.add_row("Active Model", "✓", get_active_model())
        table.add_row("VRAM", "✓", f"{get_vram_gb()}GB")
        table.add_row("RAM left", "✓", f"{get_ram_gb()}GB")
        table.add_row("Installed Models", "✓", ", ".join(get_installed_models()) or "none")
        console.print(table)
        log_info("System status checked", level="INFO", module="VERIFY")
    except Exception as e:
        console.print(Text("Status check failed", style="bold red"))
        log_info(f"ERROR : {e}", level="CRITICAL", module="VERIFY")