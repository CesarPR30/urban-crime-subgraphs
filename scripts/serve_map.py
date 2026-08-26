"""Sirve `dashboard/public/` en localhost y abre el mapa de verificación.

    python scripts/serve_map.py [--port 8000] [--no-browser]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "dashboard" / "public"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not (ROOT / "data" / "crimes.bin").exists():
        print("Falta el artefacto. Ejecuta primero:  python scripts/build_map_data.py")
        return 1

    optional = {
        "hotspots.geojson": "crimepipe export",
        "snapping.bin": "crimepipe export",
        "param_sweep.json": "crimepipe calibrate",
        "evaluation.json": "crimepipe evaluate",
        "similarity.json": "crimepipe features",
    }
    missing = [(f, cmd) for f, cmd in optional.items()
               if not (ROOT / "data" / f).exists()]
    for f, cmd in missing:
        print(f"  aviso: falta data/{f} — ejecuta `{cmd}` para esa sección")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Sirviendo {ROOT}\n  -> {url}\nCtrl+C para parar.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDetenido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
