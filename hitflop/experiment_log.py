"""Append-only experiment log (JSON lines). Every experiment records what was run, with which settings, and what came out."""
import json
import time
from pathlib import Path

from .config import LOG_PATH


def log_experiment(name, phase, settings, results, notes="", path=None):
    """Append one record. `results` must be JSON-serialisable; numpy scalars are converted."""
    path = Path(path or LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        try:
            import numpy as np
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return str(o)

    rec = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": name, "phase": phase,
           "settings": settings, "results": results, "notes": notes}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=conv, ensure_ascii=False) + "\n")
    return rec


def read_log(path=None):
    path = Path(path or LOG_PATH)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
