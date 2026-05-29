import ollama as _ollama
def chat(username : str,prompt: str, history : list,similar:list=None)->str:
    if similar is None:
        similar=[]
    messages = [{"role": "system","content": (f"You are Neo, a personal offline AI assistant for {username}. "
"You were built by an unknown developer as a fully offline, private AI. "
"Respond naturally and helpfully to all questions and tasks.")}]
    for row in similar:
        messages.append({"role":"user","content":row["prompt"]})
        messages.append({"role":"assistant","content":row["response"]})
    for row in history:
        messages.append({"role": "user", "content": row["prompt"]})
        messages.append({"role": "assistant", "content": row["response"]})
    messages.append({"role": "user", "content": prompt})
    response = _ollama.chat(model="llama3.1",messages=messages)
    return response["message"]["content"]
def get_embedding(text : str) ->list:
    response=_ollama.embeddings(model="nomic-embed-text",prompt=text)
    return response["embedding"]