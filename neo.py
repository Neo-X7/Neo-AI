from storage import connect_db,initialise_db,get_db
from logger import ai_log_info,log_info
from rich.console import Console
from rich.text import Text
console=Console()
def initialise():
    conn = connect_db()
    initialise_db(conn)
    conn.close()
def ai_history_read()->list:
    with get_db() as conn:
        cursor=conn.execute("""select prompt,response,timestamp from ai_history order by timestamp desc limit 5""")
        rows=cursor.fetchmany(5)
        return [dict(row) for row in rows] if rows else []
def ask_neo()->None:
    from neo_ollama import chat
    from storage import save_message,retrieve_similar,extract_keywords
    from config import get_user
    username,_=get_user()
    console.print(Text(f"You are now chatting with Neo. Type /exit to stop.",style="dim"))
    while True:
        history=ai_history_read()
        prompt = input(f" {username}>").strip()
        if not prompt:
            console.print(Text("Prompt cannot be empty.",style="bold red"))
            ai_log_info("Prompt was empty", level="WARNING",module="AI")
            continue
        if len(prompt)>2000:
            console.print(Text("Prompt too long. Keep it under 2000 characters.",style="bold red"))
            ai_log_info("Prompt Length exceeded above 2000 characters", level="WARNING",module="AI")
            continue
        if prompt=="/exit":
            break
        else:
            try:
                compressed = extract_keywords(prompt)
                similar=retrieve_similar(compressed) if compressed else []
                response = chat(username, prompt,history,similar)
                with get_db() as conn:
                    save_message(conn,prompt,response)
                console.print(Text(response, style="cyan"))
            except Exception as e:
                print("An Error occured while compression and memory storage")
                ai_log_info(f" ERROR : {e}", level="ERROR",module="MEMORY")
def delete_memory():
    try:
        confirm = input("Delete all AI memory? (yes/no): ").strip().lower()
        if confirm == "no":
            console.print(Text("Cancelled", style="yellow"))
            ai_log_info("User entered no ", level="INFO", module="AI")
            return
        elif confirm == "yes":
            console.print(Text("Command Accepted",style="green"))
            ai_log_info("User entered yes", level="INFO", module="AI")
        else:
            console.print(Text("Invalid Command",style="bold red"))
            ai_log_info("User entered Invalid Input", level="INFO", module="AI")
            return
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ai_history").fetchone()[0]
        console.print(Text(f"This will permanently delete {count} memory record(s)", style="yellow"))
        input_1 = input("Are you sure? (yes/no): ").strip().lower()
        if input_1 == "yes":
            with get_db() as conn:
                conn.execute("DELETE FROM ai_history")
            from lancedb_store import clear_all_vectors
            clear_all_vectors()
            console.print(Text("Memory cleared successfully", style="green"))
            ai_log_info("All AI memory deleted by user", level="INFO", module="AI")
        elif input_1 == "no":
            console.print(Text("Cancelled", style="yellow"))
            ai_log_info("User cancelled the deletion process",level="INFO",module="AI")
        else:
            console.print(Text("Invalid Input", style="bold red"))
    except Exception as e:
        console.print(Text("Failed to delete memory", style="bold red"))
        ai_log_info(f"Failed to delete memory: {e}", level="ERROR", module="AI")