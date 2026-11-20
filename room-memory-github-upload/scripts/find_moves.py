"""Compare an object's estimated placement across runs in user-supplied recording order."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visual_memory.movement import movement_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", help="Distinctive object prompt shared by all clips")
    parser.add_argument("runs", nargs="+", type=Path, help="Output folders, oldest first")
    parser.add_argument("--surface", action="append", help="Surface prompt (repeatable, default: desk and bed)")
    args = parser.parse_args()
    try:
        result = movement_history(args.runs, args.object, args.surface or ["desk", "bed"])
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
