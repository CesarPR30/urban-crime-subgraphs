"""Sirve `dashboard/public/` en localhost y abre el dashboard.

    python scripts/serve_map.py [--port 8000] [--no-browser] [--dataset lima]

Antes de servir comprueba qué artefactos hay por ciudad, para que la ausencia de
una sección del dashboard se explique aquí y no como un panel vacío sin motivo.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "dashboard" / "public"
DATA = ROOT / "data"
INDEX = DATA / "datasets.json"

#: Artefacto -> comando que lo produce. El dashboard funciona sin ellos, pero
#: la sección correspondiente queda apagada.
OPTIONAL = {
    "hotspots.geojson": "crimepipe export",
    "snapping.bin": "crimepipe export",
    "pois.bin": "crimepipe pois && crimepipe export",
    "param_sweep.json": "crimepipe calibrate",
    "evaluation.json": "crimepipe evaluate",
    "similarity.json": "crimepipe features",
    "casestudies.json": "crimepipe casestudies",
    "benchmark.json": "crimepipe benchmark",
}


def check(entries: list[dict]) -> None:
    """Informa de lo que falta en cada ciudad, sin impedir servir."""
    for e in entries:
        d = DATA / e["dir"]
        missing = [(f, c) for f, c in OPTIONAL.items() if not (d / f).exists()]
        head = f"  {e['id']:<8} {e.get('n', 0):>9,} hechos · {e.get('window', '?')}"
        print(head + ("" if missing else "  · completo"))
        for f, cmd in missing:
            print(f"      falta {f:<20} -> `{cmd} --dataset {e['id']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dataset", default=None,
                        help="ciudad con la que abrir el navegador")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if not INDEX.exists():
        print("Falta dashboard/public/data/datasets.json. Ejecuta:\n"
              "  python scripts/build_map_data.py --dataset <ciudad>\n"
              "  python scripts/build_datasets_index.py")
        return 1

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    entries = index.get("datasets") or []
    if not entries:
        print("datasets.json no lista ninguna ciudad con artefactos.")
        return 1

    ids = [e["id"] for e in entries]
    if args.dataset and args.dataset not in ids:
        print(f"dataset {args.dataset!r} no indexado. Disponibles: {', '.join(ids)}")
        return 1

    print(f"Ciudades indexadas ({len(entries)}):")
    check(entries)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/"
        if args.dataset:
            url += f"?dataset={args.dataset}"
        print(f"\nSirviendo {ROOT}\n  -> {url}\nCtrl+C para parar.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDetenido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
