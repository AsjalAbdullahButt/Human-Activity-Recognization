"""One-shot driver for the real-data training run (used for this rework's
final report). Not part of the regular package API."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deephar.train import run_pipeline  # noqa: E402

result = run_pipeline(allow_synthetic=False, run_tuning=True)

serializable = {k: v for k, v in result.items() if k not in ("report", "model_path")}
serializable["model_path"] = str(result["model_path"])
serializable["report"] = result["report"]

out_path = PROJECT_ROOT / "outputs" / "metrics" / "real_run_summary.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(serializable, f, indent=2, default=str)

print("REAL_RUN_DONE", json.dumps(serializable, default=str))
