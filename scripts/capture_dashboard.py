"""Capturas reales del dashboard para la presentación.

    python scripts/capture_dashboard.py [--port 8000] [--out reports/shots]

Usa el Chrome instalado y no el Chromium de Playwright: el mapa es MapLibre y
necesita WebGL, que el Chromium headless empaquetado resuelve por software y
tarda o falla. Con `channel="chrome"` se usa el navegador real y el mapa sale
como se ve.

Cada captura espera a que su contenido esté **de verdad** en el DOM antes de
disparar: sin eso salen paneles a medio pintar, que es peor que no tenerlos
porque parecen un fallo del dashboard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Recorte de la captura de mapa, en fracción del ancho y alto: la vista por
#: defecto abarca desierto y sierra, y en una diapositiva sobra.
#:
#: **Se recorta la imagen en vez de acercar el mapa.** `MAP.jumpTo` mata el
#: renderizador: con el basemap vectorial y 4 160 subgrafos encima, forzar el
#: cambio de zoom hace que MapLibre rasterice todo de golpe y Chrome cierra la
#: pestaña. Se comprobó paso a paso que es esa llamada y no la carga ni los
#: interruptores. Recortar da el mismo encuadre y no puede fallar.
CROP = {
    "lima": (0.00, 0.00, 0.62, 0.90),
    "chicago": (0.10, 0.00, 0.78, 0.90),
}

#: `(nombre, dataset, pestaña o None, selector a esperar, selector a recortar)`.
#: `None` en el recorte captura la ventana entera. Se recortan **grupos
#: concretos** y no la pestaña entera: un panel de 2 000 px de alto metido en una
#: diapositiva no se lee, y además se corta por el scroll del cajón.
SHOTS = [
    ("01-mapa-lima",        "lima",    None,     "#hs-count",        None),
    ("02-extraccion",       "lima",    "hs",     "#tab-hs table",    "#tab-hs .group:nth-of-type(1)"),
    ("03-parametros",       "lima",    "params", "#tab-params svg",  "#tab-params .group:nth-of-type(1)"),
    ("04-evaluacion",       "lima",    "eval",   "#tab-eval table",  "#tab-eval .group:nth-of-type(1)"),
    ("05-similitud",        "lima",    "sim",    "#tab-sim .group",  "#tab-sim .group:nth-of-type(1)"),
    ("06-intensidad-frec",  "lima",    "traj",   "#ch-if circle",    "#tab-traj .group:nth-of-type(1)"),
    ("07-frec-estabilidad", "lima",    "traj",   "#ch-fs circle",    "#tab-traj .group:nth-of-type(2)"),
    ("08-topologia",        "lima",    "traj",   "#tab-traj table",  "#tab-traj .group:nth-of-type(3)"),
    ("09-pois-contraste",   "lima",    "traj",   "#tab-traj table",  "#tab-traj .group:nth-of-type(4)"),
    ("10-matching",         "lima",    "traj",   "#tab-traj table",  "#tab-traj .group:nth-of-type(5)"),
    ("11-sintetico",        "lima",    "bench",  "#tab-bench table", "#tab-bench .group:nth-of-type(1)"),
    ("12-mannwhitney",      "lima",    "bench",  "#tab-bench table", "#tab-bench .group:nth-of-type(2)"),
    ("13-mapa-chicago",     "chicago", None,     "#hs-count",        None),
]


def _crop(path: Path, box) -> None:
    """Recorta la captura a la fracción indicada, si se dio alguna."""
    if not box:
        return
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        x0, y0, x1, y1 = box
        im.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))).save(path)


def run(port: int, out: Path, headless: bool) -> int:
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}/"
    ok, fail = 0, []

    with sync_playwright() as pw:
        # **Con ventana visible, y no es un descuido.** El mapa base es
        # vectorial (OpenFreeMap) y MapLibre lo dibuja con WebGL; el Chrome
        # headless lo tumba al primer `evaluate`, con o sin SwiftShader —se
        # probó—. Con ventana real el proceso aguanta las trece capturas.
        # `--headless` queda como opción por si algún día deja de hacer falta.
        browser = pw.chromium.launch(channel="chrome", headless=headless)
        # `device_scale_factor` 1 y no 2: con 2 el lienzo de MapLibre pasa a
        # 3400x2800 px y, con 890 000 puntos encima, el renderizador muere. A
        # 1700 px de ancho las capturas siguen siendo legibles en diapositiva.
        page = browser.new_page(viewport={"width": 1700, "height": 1400},
                                device_scale_factor=1)
        page.on("pageerror", lambda e: print(f"    [JS] {e}", file=sys.stderr))

        loaded: str | None = None
        for name, ds, tab, wait_for, clip_sel in SHOTS:
            try:
                if loaded != ds:
                    page.goto(f"{base}?dataset={ds}", wait_until="networkidle",
                              timeout=120_000)
                    # El dashboard descarga varios MB de binarios y GeoJSON antes
                    # de pintar nada; `networkidle` no basta porque el mapa sigue
                    # trayendo teselas después.
                    page.wait_for_selector("#hs-count", timeout=120_000)
                    page.wait_for_timeout(6000)

                    # **Se apagan los puntos antes de acercar.** Con los 890 000
                    # crímenes dibujados, un `jumpTo` a zoom 12 hace que MapLibre
                    # los rasterice todos de golpe y el renderizador muere; se
                    # comprobó que es esa llamada y no la carga. Además, la capa
                    # interesante para una diapositiva son los subgrafos, no la
                    # nube de puntos.
                    page.click('#mode [data-mode="points"]')     # apaga puntos
                    page.wait_for_timeout(600)
                    page.click("#btn-hs")                        # enciende subgrafos
                    page.wait_for_timeout(3500)
                    loaded = ds

                if tab:
                    # El botón del cajón es un **conmutador**: pulsarlo cuando ya
                    # está abierto lo cierra, y entonces la pestaña no es
                    # visible y `wait_for_selector` expira. Solo se pulsa si
                    # está cerrado.
                    if not page.eval_on_selector(
                            "#drawer-info", "el => el.classList.contains('open')"):
                        page.click("#btn-info")
                        page.wait_for_timeout(600)
                    page.click(f'#info-tabs button[data-tab="{tab}"]')
                    page.wait_for_timeout(1800)

                page.wait_for_selector(wait_for, timeout=30_000)
                page.wait_for_timeout(800)

                dst = out / f"{name}.png"
                if clip_sel:
                    el = page.query_selector(clip_sel)
                    el.screenshot(path=str(dst))
                else:
                    page.screenshot(path=str(dst))
                    _crop(dst, CROP.get(ds))
                kb = dst.stat().st_size / 1024
                print(f"  {name:<22} {kb:>7.0f} KB")
                ok += 1
            except PWTimeout as e:
                print(f"  {name:<22} TIMEOUT esperando {wait_for}")
                fail.append(name)
            except Exception as e:                    # pragma: no cover
                print(f"  {name:<22} ERROR {type(e).__name__}: {str(e)[:60]}")
                fail.append(name)

        browser.close()

    print(f"\n{ok} capturas en {out}")
    if fail:
        print(f"fallaron: {', '.join(fail)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "shots")
    ap.add_argument("--headless", action="store_true",
                    help="sin ventana; hoy tumba el mapa vectorial")
    a = ap.parse_args()
    return run(a.port, a.out, a.headless)


if __name__ == "__main__":
    raise SystemExit(main())
