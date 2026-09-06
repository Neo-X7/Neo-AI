import re
import shutil
import subprocess
import json
import sys
import psutil

# ---------------------------------------------------------------------------
# Fixed model — Neo always uses this family. No tier fallback to phi3 / 1.5b.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.1:8b"

# Accept common local tags that are still the same 8B Llama family so we
# don't "crash" or refuse if the user pulled a quant under a slightly
# different name. Never treat these as a reason to pick a different model.
_LLAMA_8B_ALIASES = (
    "llama3.1:8b",
    "llama3.1:8b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q5_0",
    "llama3.1:8b-instruct-q5_K_M",
    "llama3.1:8b-instruct-q8_0",
    "llama3.1:8b-q4_0",
    "llama3.1:8b-q4_K_M",
    "llama3.1:8b-q5_0",
    "llama3.1:8b-q8_0",
    "llama3.1:latest",
)

# Rough estimate: GB of VRAM/RAM per billion parameters (weights).
VRAM_PER_B_GB = 0.7

# Rough KV-cache: tokens of context per 1GB leftover after weights.
TOKENS_PER_GB_CONTEXT = 1500

MIN_CONTEXT = 1024
MAX_CONTEXT = 8192
USABLE_CONTEXT_WARNING_THRESHOLD = 2048

# Known sizes for tags that don't encode "Xb" in the name (used only for
# context math if someone passes an odd name into get_context_window).
_NAMED_MODEL_PARAM_SIZES = {
    "llama3.1:latest": 8.0,
    "llama3.1:8b": 8.0,
    "phi3:mini": 3.8,
    "phi3:medium": 14.0,
    "phi3.5:latest": 3.8,
    "phi3.5:3.8b": 3.8,
    "mistral:latest": 7.0,
}


def get_vram_gb() -> float:
    """GPU VRAM in GB, or 0.0 if no NVIDIA GPU / nvidia-smi unavailable."""
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return round(int(out.splitlines()[0]) / 1024, 1)
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return 0.0


def get_ram_gb() -> float:
    """Total system RAM in GB. Never raises."""
    try:
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return 8.0  # safe default so budget math still runs


def get_installed_models() -> list[str]:
    """Model names already in Ollama. Empty list if ollama missing/fails."""
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.check_output(["ollama", "list"], timeout=10).decode()
        lines = out.strip().splitlines()[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError, IndexError, OSError):
        return []


def parse_param_size(model_name: str) -> float:
    """Parameter count in billions from a model tag, or 0.0 if unknown."""
    if not model_name:
        return 0.0
    name_lower = model_name.lower()
    if name_lower in _NAMED_MODEL_PARAM_SIZES:
        return _NAMED_MODEL_PARAM_SIZES[name_lower]
    # Prefer explicit 8b for any llama3.1:*8b* tag
    if "llama3.1" in name_lower and "8b" in name_lower:
        return 8.0
    match = re.search(r"(\d+\.?\d*)b", name_lower)
    return float(match.group(1)) if match else 0.0


def get_budget_gb(vram_gb: float, ram_gb: float) -> float:
    """Memory budget for weights + context. Prefer VRAM; else 40% of RAM."""
    try:
        if vram_gb and vram_gb > 0:
            return float(vram_gb)
        return max(float(ram_gb) * 0.4, 1.0)
    except (TypeError, ValueError):
        return 4.0


def pull_model(model: str) -> bool:
    """Pull one model via Ollama. False on any failure — never raises."""
    if not model or not shutil.which("ollama"):
        return False
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            timeout=600,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _resolve_installed_8b(installed: list[str]) -> str | None:
    """Return DEFAULT_MODEL or a known 8B alias already installed, else None."""
    installed_set = set(installed)
    if DEFAULT_MODEL in installed_set:
        return DEFAULT_MODEL
    for alias in _LLAMA_8B_ALIASES:
        if alias in installed_set:
            return alias
    # Fuzzy: any installed tag that looks like llama3.1 + 8b
    for name in installed:
        low = name.lower()
        if "llama3.1" in low and "8b" in low:
            return name
    return None


def get_context_window(model: str, budget_gb: float) -> int:
    """
    Safe num_ctx for this model on this budget.
    Always returns a value in [MIN_CONTEXT, MAX_CONTEXT] — never raises.
    Warns (does not switch model) if the window is very small.
    """
    try:
        budget_gb = float(budget_gb) if budget_gb is not None else 4.0
    except (TypeError, ValueError):
        budget_gb = 4.0

    param_size = parse_param_size(model) or 8.0  # assume 8B if unknown
    weights_gb = param_size * VRAM_PER_B_GB
    leftover_gb = max(budget_gb - weights_gb, 0.0)
    raw_ctx = int(leftover_gb * TOKENS_PER_GB_CONTEXT)
    ctx = max(MIN_CONTEXT, min(raw_ctx if raw_ctx > 0 else MIN_CONTEXT, MAX_CONTEXT))

    if raw_ctx < USABLE_CONTEXT_WARNING_THRESHOLD:
        try:
            from logger import log_info
            msg = (
                f"Context window constrained: model '{model}' on {budget_gb}GB budget "
                f"→ ~{raw_ctx} raw tokens (using {ctx}). "
                f"Neo will keep using {model}; multi-turn may truncate. "
                f"Use /set-context to override, or free VRAM/RAM."
            )
            log_info(msg, level="WARNING", module="HARDWARE")
            print(f"[WARNING] {msg}")
        except Exception:
            pass

    return ctx


def get_recommended_model() -> tuple[str, bool]:
    """
    Always target llama3.1:8b (or an installed 8B alias).

    Returns (model_name, ready).
    ready=True  → tag is already in `ollama list` (or we just pulled DEFAULT_MODEL).
    ready=False → not installed and pull failed; caller should not crash the app
                  if it can show a clear error — neo_ollama retries once.

    Never selects phi3 / tiny fallbacks. Other installed models are ignored
    for selection (they remain on disk; Neo simply doesn't switch to them).
    """
    installed = get_installed_models()
    resolved = _resolve_installed_8b(installed)
    if resolved:
        return resolved, True

    # Not installed — try pull DEFAULT_MODEL only (no tier cascade)
    success = pull_model(DEFAULT_MODEL)
    if success:
        return DEFAULT_MODEL, True

    # Still return the name so logs/UI show what we wanted; ready=False
    return DEFAULT_MODEL, False


_cached_budget_gb = None


def get_cached_budget_gb() -> float:
    """Budget once per process. Never raises."""
    global _cached_budget_gb
    if _cached_budget_gb is None:
        try:
            _cached_budget_gb = get_budget_gb(get_vram_gb(), get_ram_gb())
        except Exception:
            _cached_budget_gb = 4.0
    return _cached_budget_gb


if __name__ == "__main__":
    vram, ram = get_vram_gb(), get_ram_gb()
    installed = get_installed_models()
    model, ready = get_recommended_model()
    budget = get_budget_gb(vram, ram)
    ctx = get_context_window(model, budget)

    if "--json" in sys.argv:
        print(json.dumps({
            "vram_gb": vram,
            "ram_gb": ram,
            "budget_gb": budget,
            "installed_models": installed,
            "selected_model": model,
            "ready": ready,
            "num_ctx": ctx,
        }))
    else:
        print(f"VRAM: {vram}GB | RAM: {ram}GB | Budget: {budget}GB")
        print(f"Installed: {installed}")
        print(f"Selected: {model} | ready={ready} | num_ctx={ctx}")
        if not ready:
            print("Model not available — run: ollama pull llama3.1:8b")