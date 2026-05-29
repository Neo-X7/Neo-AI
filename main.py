import sys
from neo import ask_neo, delete_memory, initialise
from file_data import verify
from logger import log_info, log_menu
from rich.console import Console
from rich.text import Text
import subprocess
import time
def start_ollama():
    try:
        import httpx
        httpx.get("http://localhost:11434")
        return
    except Exception:
        console.print(Text("Starting OLlama server...",style="dim"))
        subprocess.Popen(["ollama","serve"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        time.sleep(2)
        console.print(Text("Ollama server started.", style="dim"))
console = Console()
HELP_TEXT = """
[bold cyan]Neo AI[/bold cyan] — available commands:
  [cyan]/chat[/cyan]     start a conversation with Neo
  [cyan]/delete[/cyan]   delete all AI memory
  [cyan]/status[/cyan]   system status
  [cyan]/logs[/cyan]     open log viewer
  [cyan]/help[/cyan]     show this message
  [cyan]/exit[/cyan]     exit
"""
def main():
    start_ollama()
    initialise()
    from config import get_user
    username,is_new=get_user()
    if is_new:
        console.print(f"\n NIce to meet you [bold cyan] {username}[/bold cyan]. I'm Neo, your personal offline AI")
    else:
        console.print(f"\nWelcome back [bold cyan]{username}[/bold cyan]")
    console.print(HELP_TEXT)
    while True:
        try:
            cmd = input("Neo > ").strip().lower()
            if not cmd:
                continue
            log_info(f"User entered command: {cmd}", level="INFO", module="MAIN")
            if cmd == "/chat":
                ask_neo()
            elif cmd == "/delete":
                delete_memory()
            elif cmd == "/status":
                verify()
            elif cmd == "/logs":
                log_menu()
            elif cmd == "/help":
                console.print(HELP_TEXT)
            elif cmd == "/exit":
                log_info("Program exited", level="INFO", module="MAIN")
                sys.exit()
            else:
                console.print(Text(f"Unknown command '{cmd}'. Type /help to see available commands.", style="yellow"))
                log_info(f"Unknown command: {cmd}", level="WARNING", module="MAIN")
        except KeyboardInterrupt:
            console.print(Text("\nUse /quit to exit.", style="yellow"))
        except Exception as e:
            console.print(Text("An error occurred.", style="bold red"))
            log_info(f"ERROR: {e}", level="ERROR", module="MAIN")
if __name__ == "__main__":
    main()