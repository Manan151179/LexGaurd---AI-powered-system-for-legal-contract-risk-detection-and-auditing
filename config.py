"""
LexGuard — Global Configuration & Determinism
===============================================
Central configuration for reproducibility.  Import this module at the
top of every script/notebook to guarantee deterministic behaviour.

    import config          # seeds are set on import
    uid = config.get_seeded_uuid()
    dev = config.get_device()
"""

import os
import random

import numpy as np

# ──────────────────────────────────────────────
# 1. Global Seed
# ──────────────────────────────────────────────
GLOBAL_SEED: int = int(os.environ.get("GLOBAL_SEED", "42"))

# Python hash seed (must be set before interpreter hashes anything,
# but setting it here is the best we can do short of a wrapper script).
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

# Stdlib random
random.seed(GLOBAL_SEED)

# NumPy
np.random.seed(GLOBAL_SEED)

# PyTorch (imported lazily so the project works without torch installed)
try:
    import torch

    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except ImportError:
    torch = None  # type: ignore[assignment]


# ──────────────────────────────────────────────
# 2. Device Selection  (MPS → CUDA → CPU)
# ──────────────────────────────────────────────
def get_device() -> str:
    """Return the best available compute device string.

    Priority:  Apple-Silicon MPS  →  NVIDIA CUDA  →  CPU
    Falls back to CPU if nothing else is available.
    """
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────
# 3. Deterministic UUID Generator
# ──────────────────────────────────────────────
_uuid_rng = random.Random(GLOBAL_SEED)


def get_seeded_uuid() -> str:
    """Return a UUID-4-style string generated from the seeded RNG.

    This ensures chunk IDs are identical across runs when the same
    data is processed in the same order.
    """
    # Build a 128-bit random int and format as UUID-4
    bits = _uuid_rng.getrandbits(128)
    # Set version (4) and variant bits to match UUID-4 spec
    bits &= ~(0xF000 << 64)          # clear version nibble
    bits |= (0x4000 << 64)           # set version 4
    bits &= ~(0xC000000000000000)     # clear variant bits
    bits |= 0x8000000000000000        # set variant 1
    # Format: 8-4-4-4-12
    h = f"{bits:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


# ──────────────────────────────────────────────
# 4. Pipeline Hyperparameters (single source of truth)
# ──────────────────────────────────────────────
HYPERPARAMS = {
    "global_seed": GLOBAL_SEED,
    "device": get_device(),
    "min_text_chars": int(os.environ.get("MIN_TEXT_CHARS", "50")),
    "agent_temperature": float(os.environ.get("AGENT_TEMPERATURE", "0.1")),
    "agent_max_steps": int(os.environ.get("AGENT_MAX_STEPS", "5")),
    "retrieval_top_k": int(os.environ.get("RETRIEVAL_TOP_K", "5")),
    # Local separated storage directory (inspired by HyperGraphRAG working_dir)
    "working_dir": os.environ.get("LEXGUARD_WORKING_DIR", "./project_data_store"),
}


if __name__ == "__main__":
    print("LexGuard Configuration")
    print("=" * 40)
    for k, v in HYPERPARAMS.items():
        print(f"  {k:25s} = {v}")
