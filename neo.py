from logger import ai_log_info,log_info
from rich.console import Console
from rich.text import Text
from neo_ollama import chat, extract_facts, needs_search
from storage import connect_db,initialise_db,get_db,save_message,retrieve_similar,extract_keywords,process_fact
from config import get_user
from web_search import web_search
console=Console()
def initialise():
    """Create the SQLite tables (ai_history, facts) if they do not exist yet."""
    conn = connect_db()
    initialise_db(conn)
    conn.close()
def ai_history_read()->list:
    """Return the last 5 chat turns (prompt, response, timestamp) as dicts, newest first."""
    with get_db() as conn:
        cursor=conn.execute("""select prompt,response,timestamp from ai_history order by timestamp desc limit 5""")
        rows=cursor.fetchmany(5)
        return [dict(row) for row in rows] if rows else []
def ask_neo()->None:
    """Run the interactive chat loop until the user types /exit.

    /search forces a web search; otherwise needs_search() decides. Each turn retrieves
    matching facts, calls chat(), and saves the exchange. Facts are extracted from the
    message only when no web search ran.
    """
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
        chat_prompt=prompt
        web=[]
        searched=False
        if prompt=="/search" or prompt.startswith("/search "):
            query=prompt[7:].strip()
            chat_prompt=query
            if query=="":
                console.print(Text("Query is empty",style="dim"))
                continue
            console.print(Text(f"Searching the web....",style="dim"))
            ai_log_info("Search has been initiated", level="INFO", module="WEB SEARCH")
            web=web_search(query)
            searched=True
        elif needs_search(prompt):
            console.print(Text("Searching the web....",style="dim"))
            ai_log_info("Search has been initiated", level="INFO", module="WEB SEARCH")
            web=web_search(prompt)
            searched=True
        try:
            similar=retrieve_similar(chat_prompt)
            response = chat(username, chat_prompt,history,similar,web)
            console.print(Text(response, style="cyan"))
            with get_db() as conn:
                save_message(conn,chat_prompt,response)
                if not searched:
                    remember_facts(conn, prompt)
        except Exception as e:
            print("An Error occured while replaying and memory storage")
            ai_log_info(f" ERROR : {e}", level="ERROR",module="MEMORY")
def delete_memory():
    """Delete all memory (ai_history, facts and every vector) after two yes/no confirmations."""
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
            fact_del=conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        console.print(Text(f"This will permanently delete {count} memory record(s) and {fact_del} fact(s)", style="yellow"))
        input_1 = input("Are you sure? (yes/no): ").strip().lower()
        if input_1 == "yes":
            with get_db() as conn:
                conn.execute("DELETE FROM ai_history")
                conn.execute("DELETE FROM facts")
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

def remember_facts(conn, prompt : str)->None:
    """Extract facts from the prompt and pass each through the write gate. A failure on one fact is logged and does not stop the rest."""

    facts=extract_facts(prompt)
    for fact in facts:
        try:
            process_fact(conn, fact)
        except Exception as e:
            ai_log_info(f"remember_facts failed on {fact} : {e}",level="ERROR",module="MEMORY")