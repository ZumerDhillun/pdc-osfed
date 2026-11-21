"""Save the machine and library versions to notes/env_<name>.txt.

    python -m scripts.save_env --name ubuntu      (on Kaggle: --name kaggle)
"""
import argparse

from src.utils import env_report, project_root

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="ubuntu")
args = ap.parse_args()
path = project_root() / "notes" / f"env_{args.name}.txt"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(env_report())
print(env_report())
print("saved", path)
