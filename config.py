import os
import json
import base64
_BASE_DIR=os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH=os.path.join(_BASE_DIR,"config.json")
def _encode(name:str)->str:
    return base64.b64encode(name.encode()).decode()
def _decode(val:str)->str:
    return base64.b64decode(val.encode()).decode()
def get_user()->tuple[str,bool]:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH,"r")as f:
            raw=json.load(f)["username"]
        try:
            name=_decode(raw)
        except Exception:
            name=raw
            with open(_CONFIG_PATH,"w")as f:
                json.dump({"username":_encode(name)},f)
        return name, False
    name=input("Welcome to Neo. What should I call you? : ").strip()
    while not name:
        name=input("Name cannot be empty: ").strip()
    with open(_CONFIG_PATH,"w")as f:
        json.dump({"username":_encode(name)},f)
    return name, True