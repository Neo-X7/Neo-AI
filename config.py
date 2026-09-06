import os
import json
from logger import ai_log_info
_BASE_DIR=os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH=os.path.join(_BASE_DIR,"config.json")


def _read_config() -> dict:
    """Read the whole config file. Returns {} if it doesn't exist yet."""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


def _write_config(data: dict) -> None:
    """Write the whole config file."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(data, f)


def get_user()->tuple[str,bool]:
    data = _read_config()
    name = data.get("username")
    if name:
        return name, False
    name = input("Welcome to Neo. What should I call you? : ").strip()
    ai_log_info("Name has been entered", level="INFO", module="JSON MEMORY")
    while not name:
        name = input("Name cannot be empty: ").strip()
        ai_log_info("Name field is empty", level="WARNING", module="JSON MEMORY")
    data["username"] = name
    _write_config(data)
    return name, True


def get_context_override() -> int | None:
    """Return the user's manually-set num_ctx value, or None if never set."""
    return _read_config().get("context_override")


def set_context_override(value: int) -> None:
    """Save a manual num_ctx override so it's used on every future run."""
    data = _read_config()
    data["context_override"] = value
    _write_config(data)


def get_response_length_override() -> int | None:
    """Return the user's manually-set num_predict value, or None if never set."""
    return _read_config().get("response_length_override")


def set_response_length_override(value: int) -> None:
    """Save a manual num_predict override so it's used on every future run."""
    data = _read_config()
    data["response_length_override"] = value
    _write_config(data)