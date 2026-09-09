"""Presentación de lo implementado, con capturas reales del dashboard.

    python scripts/capture_dashboard.py     # primero las capturas
    python scripts/deck_implementacion.py   # luego la presentación

-> `reports/presentacion_implementacion.pptx`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_decks import (  # noqa: E402
    OUT, SHOTS, figure_slide, new_deck, section_slide, stat_slide, title_slide,
)


def build() -> Path:
    prs = new_deck()

    title_slide(
        prs,
        "Subgrafos urbanos de criminalidad",
        "Detección topológica de hotspots sobre la red vial\n"
        "Chicago 2024-2025  ·  Lima Metropolitana 2018-2025",
        "César Pajuelo  ·  capturas reales del dashboard del pipeline",
    )

    # ── 1 · qué se construyó ──────────────────────────────────────────────
    section_slide(
        prs, "punto de partida", "Dos ciudades, un mismo método",
        "El pipeline se valida sobre Chicago —que tiene tabla de referencia— y "
        "se aplica a Lima sin recalibrar ningún parámetro.")

    stat_slide(prs, "Lima Metropolitana", [
        ("887 137", "hechos retenidos", "de 3.1 M denuncias del CSV"),
        ("90", "meses", "2018-01 a 2025-06"),
        ("35", "tipos de delito", "subtipo_hecho, no 5 familias"),
        ("135 633", "nodos de red", "OpenStreetMap"),
    ], "Chicago: 213 389 hechos · 24 meses · 4 tipos · 29 537 nodos.")

    figure_slide(
        prs, "El dashboard: 887 137 hechos sobre la red vial de Lima",
        SHOTS / "01-mapa-lima",
        caption="Selector de ciudad arriba a la izquierda · línea de tiempo "
                "arrastrable para filtrar el periodo.")

    # ── 2 · K adaptativo ──────────────────────────────────────────────────
    section_slide(
        prs, "encargo 1", "El número de hotspots deja de ser un parámetro",
        "Antes se sacaban siempre los top-20. Ahora cada mes devuelve los que "
        "son estadísticamente significativos.")

    stat_slide(prs, "K adaptativo: Monte Carlo sobre la red", [
        ("3 – 70", "hotspots por mes", "en Lima; mediana 46"),
        ("99", "réplicas nulas", "por mes, α = 0.05"),
        ("0", "meses en el tope", "max_k = 80 no se satura"),
        ("ρ = 0.66", "correlación con volumen", "sigue a los datos sin ser función de ellos"),
    ], "Kulldorff (1997) adaptado a red vial por Shiode & Shiode (2020). Se compara "
       "contra el máximo por réplica, así que no hace falta corregir por test múltiple.")

    figure_slide(
        prs, "El estadístico no puede ser el crimen capturado",
        SHOTS / "02-extraccion",
        bullets=[
            ("Con crimen capturado", "el nulo uniforme da regiones enormes: "
             "máximo 291 contra hotspots observados de ~55. Nada era significativo."),
            ("Con log-razón de Poisson", "normalizada por tamaño, el nulo cae a "
             "media 23 y el contraste discrimina."),
            ("El fallo", "era comparar concentración contra tamaño."),
        ])

    figure_slide(
        prs, "Sensibilidad de parámetros: 180 combinaciones",
        SHOTS / "03-parametros",
        bullets=[
            ("σ = 120 m", "es lo que recomienda el barrido sobre Lima."),
            ("Y sobre Chicago también", "dos ciudades con 29 537 y 135 633 nodos "
             "convergen al mismo ancho de banda."),
            ("α", "mueve la cobertura solo un 15.5 %."),
        ])

    figure_slide(
        prs, "Contra las líneas base", SHOTS / "04-evaluacion",
        bullets=[
            ("+14.2 %", "sobre BFS voraz con la misma huella."),
            ("+40.7 %", "sobre anchura pura."),
            ("Huella igualada", "sin eso, comparar cobertura no mide nada."),
            ("La base recibe", "tantas regiones como sacó el extractor ese mes."),
        ])

    figure_slide(
        prs, "Similitud entre subgrafos", SHOTS / "05-similitud",
        bullets=[
            ("graph2vec, WWL, scattering", "tres embeddings comparables."),
            ("§5.1", "el crimen no entra en la similitud: `Subgraph` no lo lleva."),
        ])

    # ── 3 · trayectorias ──────────────────────────────────────────────────
    section_slide(
        prs, "encargo 2", "Intensidad × Frecuencia",
        "Un punto por sitio, no por subgrafo-mes. Frequency(H) = meses en que "
        "aparece ÷ meses analizados.")

    figure_slide(
        prs, "Intensidad × Frecuencia", SHOTS / "06-intensidad-frec",
        bullets=[
            ("Arriba a la derecha", "los crónicos: frecuentes e intensos."),
            ("Arriba a la izquierda", "los que resaltaron por algo puntual."),
            ("Identidad", "por índice de Jaccard sobre nodos, no por recuento: "
             "el recuento premia a los subgrafos grandes."),
            ("Clic", "pone la trayectoria en foco sobre el mapa."),
        ])

    figure_slide(
        prs, "¿Persistencia implica estabilidad espacial? No",
        SHOTS / "07-frec-estabilidad",
        bullets=[
            ("213 frecuentes y estables", ""),
            ("180 frecuentes y móviles", "casi mitad y mitad"),
            ("Conclusión", "un hotspot puede repetirse mes tras mes y estar "
             "desplazándose. Un ranking mensual no lo puede mostrar."),
            ("Cortes", "medianas observadas, no valores fijos."),
        ])

    # ── 4 · estudios de caso ──────────────────────────────────────────────
    section_slide(
        prs, "estudios de caso", "¿Qué distingue a un hotspot persistente?",
        "1 432 persistentes contra 1 538 episódicos, por terciles de frecuencia.")

    figure_slide(
        prs, "Topología: la respuesta honesta es «poco»",
        SHOTS / "08-topologia",
        bullets=[
            ("Ningún |δ| llega a 0.2", "solape casi total entre las dos "
             "distribuciones."),
            ("Dirección coherente", "más grandes, más ramificados, menos densos, "
             "menor intermediación."),
            ("δ es la delta de Cliff", "no diferencia de medias: los descriptores "
             "no son normales."),
            ("p ajustado", "Benjamini-Hochberg sobre 15 pruebas."),
        ])

    figure_slide(
        prs, "POIs: el entorno separa mejor que la forma",
        SHOTS / "09-pois-contraste",
        bullets=[
            ("lq:finance", "14.15 vs 9.74"),
            ("lq:retail", "10.88 vs 5.85"),
            ("entropía funcional", "0.605 vs 0.533"),
            ("Lectura", "lo que sostiene un hotspot en el tiempo se parece más a "
             "la actividad que a la forma de la calle."),
        ])

    figure_slide(
        prs, "Matching: misma forma, muy distinto crimen",
        SHOTS / "10-matching",
        bullets=[
            ("2 % más cercano", "en distancia estructural, con crimen ≥3×."),
            ("Caso extremo", "134 hechos contra 3, a distancia 0.0."),
            ("Se excluyen", "pares que solapan en el espacio: serían el mismo "
             "sitio medido dos veces."),
        ])

    # ── 5 · validación ────────────────────────────────────────────────────
    section_slide(
        prs, "encargo 3", "Validación con datos sintéticos",
        "Réplica de Shiode & Shiode (2020), §3 y §5. La única métrica absoluta "
        "del proyecto.")

    figure_slide(
        prs, "PPV, sensibilidad y F1 sobre verdad conocida",
        SHOTS / "11-sintetico",
        bullets=[
            ("Proceso de Poisson", "padres → hijos → fondo uniforme."),
            ("σ controla el sobredisparo", "PPV 0.29 con σ=40 y 0.10 con σ=120, "
             "con sensibilidad ~1.0 en todo el rango."),
            ("La red se recorta", "Shiode usa 900×750 m; sobre la ciudad entera "
             "el problema es trivial y los tres métodos empatan."),
        ])

    figure_slide(
        prs, "El extractor no bate a la línea base en localización exacta",
        SHOTS / "12-mannwhitney",
        bullets=[
            ("F1", "topológico 0.172 · voraz 0.189 · p = 0.011"),
            ("No es un fallo", "la verdad sintética son picos puntuales, que es "
             "lo que persigue una base que crece desde los nodos calientes."),
            ("Dos preguntas distintas", "¿captura más crimen? sí, +14.2 %. "
             "¿localiza mejor la cuadra? no, con σ = 120 m."),
        ])

    # ── 6 · POIs temporales ───────────────────────────────────────────────
    section_slide(
        prs, "datos", "POIs con eje temporal",
        "Ocho instantáneas anuales de OpenStreetMap, una por año de la ventana.")

    stat_slide(prs, "De Overpass a instantáneas anuales de Geofabrik", [
        ("450 105", "POIs", "8 instantáneas, 2018-2025"),
        ("0.4 %", "sin clasificar", "la taxonomía cubre Lima mejor que Chicago"),
        ("4 min", "de ingesta", "sin cuota ni clave de API"),
        ("×3.4", "education 2018→2025", "el 75 % en 2018-19: una importación"),
    ], "Por eso el contraste va sobre cociente de localización con la red como base: "
       "una importación nacional multiplica numerador y denominador y se cancela.")

    figure_slide(
        prs, "Chicago: el dataset de validación", SHOTS / "13-mapa-chicago",
        caption="Mismo pipeline, mismos parámetros. Chicago conserva K = 20 fijo "
                "porque §7.3 tabula sus valores esperados.")

    # ── cierre ────────────────────────────────────────────────────────────
    stat_slide(prs, "Estado", [
        ("136", "tests", "pasan"),
        ("8", "artefactos", "por ciudad"),
        ("σ = 120 m", "confirmado", "el barrido lo recomienda en las dos ciudades"),
        ("2025", "no comparable", "solo el 4.5 % de las denuncias tiene coordenada"),
    ], "σ es distancia sobre la red (Dijkstra), no radio geográfico. "
       "La red es `drive`: no incluye escaleras ni pasajes peatonales.")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "presentacion_implementacion.pptx"
    prs.save(dst)
    return dst


if __name__ == "__main__":
    p = build()
    print(f"escrito {p}  ({p.stat().st_size / 1e6:.1f} MB)")
