import ollama as _ollama
import json
from logger import ai_log_info
def chat(username : str,prompt: str, history : list,similar:list=None, web_results=None)->str:
    """Send the conversation to llama3.1 and return the reply text.

    Known facts and web results go into the system message. With web results the offline
    wording is dropped; without them Neo presents itself as offline. Recent history and
    the prompt follow.
    """
    if similar is None:
        similar=[]
    if web_results is None:
        web_results=[]
    facts_block = ""
    if similar:
        facts_block = "Known facts about the user:\n" + "\n".join(s["fact_text"] for s in similar)
    web_block = ""
    if web_results:
        lines = [f"- {r['title']}: {r['snippet']}" for r in web_results]
        web_block = (
            "\n\nWeb results (fetched just now, use them to answer the question):\n"
            + "\n".join(lines)
        )
    if web_block:
        intro = f"You are Neo, a personal AI assistant for {username}. "
        web_rule = (
        "Web results fetched from the internet just now are listed below. "
        "Answer using only those results. "
        "Mention only titles and dates that appear in them, and never invent sources. "
        "If they do not answer the question, say so briefly. "
        "Never say you lack internet access. "
    )
    else:
        intro = f"You are Neo, a personal offline AI assistant for {username}. "
        web_rule = "You were built by an unknown developer as a fully offline, private AI. "

    messages = [{"role": "system", "content": (
    intro
    + web_rule
    + "Respond naturally and helpfully to all questions and tasks.\n"
    + facts_block
    + web_block
)}]
    for row in history:
        messages.append({"role": "user", "content": row["prompt"]})
        messages.append({"role": "assistant", "content": row["response"]})
    messages.append({"role": "user", "content": prompt})
    response = _ollama.chat(model="llama3.1",messages=messages)
    return response["message"]["content"]
def get_embedding(text : str) ->list:
    """Return the nomic-embed-text embedding (768 floats) for the text."""
    response=_ollama.embeddings(model="nomic-embed-text",prompt=text)
    return response["embedding"] 

def decide_action(new_fact_text : str, candidates : list[dict])->dict:
    """Ask llama3.1 whether a new fact should ADD, UPDATE, DELETE or NONE against the candidate facts.

    Returns {'action', 'target_id'}. Any error or malformed reply falls back to ADD.
    """

    lines=[f"{r['id']} : {r['fact_text']}" for r in candidates]
    candidates_lines="\n".join(lines)

    system_text = """You manage a list of stored facts about a user. You receive a NEW fact and a list of EXISTING facts, each written as "id: text". Decide what to do with the new fact.

Actions:
- ADD: the new fact is new information, unrelated to any existing fact.
- UPDATE: the new fact replaces an existing fact (same subject, changed value).
- DELETE: ONLY when the user explicitly asks to forget or remove a fact ("forget that...", "remove...", "that's wrong"). Statements that something changed or stopped ("I stopped...", "I quit...", "I no longer...", "I don't... anymore", "I am not...") are NOT deletes. Use UPDATE for them.
- NONE: the new fact repeats an existing fact.
- If an existing fact already says the same thing as the new fact, the action is NONE, and target_id is that fact's id. Never retract a fact that matches the new fact.

Reply with JSON only, in exactly this shape:
{"action": "ADD" | "UPDATE" | "DELETE" | "NONE", "target_id": <id of the existing fact, or null>}

Rules:
- target_id must be one of the ids listed. Use null for ADD.
- If unsure, choose ADD."""

    user_text = f"New fact: {new_fact_text}\n\nExisting facts:\n{candidates_lines}"  

    messages = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": user_text},
]
    try:
        response=_ollama.chat(
            model="llama3.1",
            messages=messages, 
            format="json", 
            options={"temperature":0})
        text= response["message"]["content"]
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return {"action": "ADD", "target_id": None}
    except Exception as e:
        return {"action":"ADD", "target_id":None}

def extract_facts(prompt: str) -> list[str]:
    """Return durable facts the user states about themselves, or []."""
    system_text = """You extract facts from ONE user message.

Rules:
- Use ONLY what the user states about themselves in the message below. Never use anything else.
- Each fact is one short standalone sentence, e.g. "I study robotics".
- If the message has several facts, return each one separately.
- Questions, greetings, thanks and filler contain no facts. Return an empty list.
- If the user says they stopped, left or no longer do something, return that as a fact.
- Do not guess or add facts that are not in the message.
- A message can contain a fact and a question together. Return the fact and ignore the question.
- Return each fact as one complete sentence. Never split one sentence into fragments or partial sentences.
- Preferences, wants and plans the user states about themselves are facts.
- If the user asks to forget or remove something, return that request as the fact, keeping the word "forget" or "remove" in it.

Reply with JSON only, in exactly this shape:
{"facts": ["...", "..."]}
If there are no facts: {"facts": []}"""

    user_text = f"User message: {prompt}"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        response = _ollama.chat(
            model="llama3.1",
            messages=messages,
            format="json",
            options={"temperature": 0},
        )
        result = json.loads(response["message"]["content"])
        if not isinstance(result, dict):
            return []
        facts = result.get("facts")
        if not isinstance(facts, list):
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except Exception as e:
        ai_log_info(f"ERROR : {e}", level="ERROR", module="MEMORY")
        return []

def needs_search(prompt: str) -> bool:
    """Ask llama3.1 whether the message needs a live web search. True only for an explicit true; any error returns False."""
    system_text = """You decide whether a user message needs a live web search.

Answer true ONLY if the message asks about something recent or that changes over time,
such as news, prices, scores, weather, or who currently holds a position.

Answer false for:
- questions about the user themselves (anything with "my" or "I")
- greetings and small talk
- coding, maths and writing tasks
- general knowledge that does not change

Reply with JSON only, in exactly this shape:
{"search": true}
or
{"search": false}"""

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"User message: {prompt}"},
    ]
    try:
        response = _ollama.chat(
            model="llama3.1",
            messages=messages,
            format="json",
            options={"temperature": 0},
        )
        result = json.loads(response["message"]["content"])
        if not isinstance(result, dict):
            return False
        return result.get("search") is True
    except Exception:
        return False