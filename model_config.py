import os
import json
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.path.expanduser("~/.neo/model_config.json")
_shared_embed_model = None

def _load_model_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def _save_model_config(path: str):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump({"embed_model_path": path}, f)

def get_embed_model_path() -> str:
    env_path = os.environ.get("NEO_EMBED_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    cfg = _load_model_config()
    saved_path = cfg.get("embed_model_path")
    if saved_path and os.path.exists(saved_path):
        return saved_path

    print("Embedding model not found via env var or saved config.")
    while True:
        user_path = input("Enter the full path to your all-mpnet-base-v2 model folder: ").strip()
        if os.path.exists(user_path):
            _save_model_config(user_path)
            return user_path
        print(f"Path '{user_path}' does not exist. Try again.")

def get_shared_embed_model() -> SentenceTransformer:
    global _shared_embed_model
    if _shared_embed_model is None:
        _shared_embed_model = SentenceTransformer(get_embed_model_path())
    return _shared_embed_model