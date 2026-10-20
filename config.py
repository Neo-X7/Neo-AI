import os
import json
from logger import ai_log_info
_BASE_DIR=os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH=os.path.join(_BASE_DIR,"config.json")
def get_user()->tuple[str,bool]:
    """Return (username, is_new).

    Reads the saved name from config.json, or asks for one on first run and saves it.
    """
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH,"r")as f:
            name=json.load(f)["username"]
        return name, False
    name=input("Welcome to Neo. What should I call you? : ").strip()
    ai_log_info("Name has been entered",level="INFO",module="JSON MEMORY")
    while not name:
        name=input("Name cannot be empty: ").strip()
        ai_log_info("Name field is empty",level="WARNING",module="JSON MEMORY")
    with open(_CONFIG_PATH,"w")as f:
        json.dump({"username":name},f)
    return name, True