"""Generate installer icon files from the bundled PNG.

Writes dist/icon.ico (multi-resolution, consumed by the Inno Setup installer for
the Start Menu / desktop shortcuts and the Add/Remove Programs entry). macOS
.icns generation lives in scripts/make_app_bundle.sh (sips + iconutil need a
macOS host). Run with Pillow available, e.g.:

    uvx --with pillow python scripts/make_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "deepreefmap" / "resources" / "icon.png"
SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> None:
    out = REPO / "dist" / "icon.ico"
    out.parent.mkdir(exist_ok=True)
    image = Image.open(SOURCE).convert("RGBA")
    image.save(out, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
