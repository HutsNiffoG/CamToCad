"""Zet de afhankelijkheden van het installer-wheel vast op de versies uit uv.lock (V23, v0.15).

PyApp installeert bij de eerste start het ingebouwde wheel met alles wat dat wheel vraagt. Een gewoon wheel vraagt
"numpy>=1.24": dan komt de nieuwste versie, ook een die nog nooit met camtocad getest is. Dit script vervangt in een
kopie van pyproject.toml de lijst `dependencies` door de vastgezette lijst uit `uv export` (alle pakketten, ook die
van de servergroep, met hun platformvoorwaarden), zodat de installer de versies neemt waarmee de tests draaiden.

    uv export --frozen --no-hashes --no-emit-project --extra server -o vast.txt
    python installer/pin_dependencies.py vast.txt pyproject.toml
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REQUIRED = ("cadquery", "numpy", "opencv-python-headless", "fastapi", "uvicorn", "pi-heif")  # mag niet ontbreken
DEPS = re.compile(r"^dependencies = \[\n.*?^\]\n", re.MULTILINE | re.DOTALL)


def pinned(export: str) -> list[str]:
    """De vereisten uit de uitvoer van `uv export --no-hashes`: één per regel, zonder de commentaarregels."""
    reqs = []
    for line in export.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or line.endswith("\\") or "==" not in line.split(";")[0]:
            raise ValueError(f"onverwachte regel in de export (gebruik --no-hashes --no-emit-project): {line}")
        reqs.append(line)
    names = {re.split(r"[=;\s\[]", r, maxsplit=1)[0].lower() for r in reqs}
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise ValueError(f"de export mist {', '.join(missing)} (vergeten: --extra server?)")
    return reqs


def pin(pyproject: str, reqs: list[str]) -> str:
    """pyproject.toml met `dependencies` vervangen door de vastgezette lijst."""
    if len(DEPS.findall(pyproject)) != 1:
        raise ValueError("geen (of meer dan één) blok 'dependencies = [' in pyproject.toml")
    block = "dependencies = [\n" + "".join(f'    "{r}",\n' for r in reqs) + "]\n"
    out = DEPS.sub(lambda _: block, pyproject)
    assert tomllib.loads(out)["project"]["dependencies"] == reqs
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    export, toml = Path(argv[0]), Path(argv[1])
    reqs = pinned(export.read_text(encoding="utf-8"))
    toml.write_text(pin(toml.read_text(encoding="utf-8"), reqs), encoding="utf-8")
    print(f"{len(reqs)} afhankelijkheden vastgezet in {toml}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
