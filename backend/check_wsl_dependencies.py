"""Check whether WSL training dependencies are installed in the active venv."""

from __future__ import annotations

import importlib.metadata as metadata
import re
import sys
from pathlib import Path


_REQ_NAME_RE = re.compile(r"[<>=!~\s\[]")


def _normalized_name(requirement: str) -> str:
    return _REQ_NAME_RE.split(requirement, maxsplit=1)[0].lower().replace("_", "-")


def _read_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", maxsplit=1)[0].strip()
        if requirement and not requirement.startswith("-"):
            requirements.append(requirement)
    return requirements


def main() -> int:
    requirements_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("requirements-wsl.txt")
    requirements = _read_requirements(requirements_path)
    installed = {
        dist.metadata["Name"].lower().replace("_", "-")
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }

    missing = [
        requirement
        for requirement in requirements
        if _normalized_name(requirement) not in installed
    ]

    if missing:
        print("Missing WSL dependencies: " + ", ".join(missing))
        return 1

    print("[OK] WSL training dependencies already installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
