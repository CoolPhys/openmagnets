from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build" / "native"
PACKAGE_DIR = PROJECT_ROOT / "src" / "openmagnets"


def output_names() -> tuple[str, ...]:
    if sys.platform.startswith("win"):
        return ("openmagnets_native.dll", "libopenmagnets_native.dll")
    if sys.platform == "darwin":
        return ("libopenmagnets_native.dylib", "openmagnets_native.dylib")
    return ("libopenmagnets_native.so", "openmagnets_native.so")


def find_meson() -> str:
    env_value = os.environ.get("OPENMAGNETS_MESON")
    if env_value:
        return env_value
    path = shutil.which("meson")
    if path:
        return path
    raise SystemExit(
        "meson was not found on PATH. Install the build requirements first or set OPENMAGNETS_MESON."
    )


def run(cmd: list[str]) -> None:
    print("[openmagnets]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def find_built_library() -> Path:
    candidates = set(output_names())
    matches = [path for path in BUILD_DIR.rglob("*") if path.name in candidates]
    if not matches:
        raise SystemExit("build finished but no native library was found under build/native")
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    meson = find_meson()
    run([meson, "setup", str(BUILD_DIR), "--wipe", "--buildtype=release"])
    run([meson, "compile", "-C", str(BUILD_DIR)])

    built_library = find_built_library()
    out_path = PACKAGE_DIR / built_library.name
    shutil.copy2(built_library, out_path)

    print(f"[openmagnets] copied {built_library} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
