"""Presentación del análisis exploratorio del dataset de Lima.

    # ejecutar antes el cuaderno, que es quien genera las figuras
    python scripts/deck_eda.py

-> `reports/eda_delitos_lima.pptx`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_decks import (  # noqa: E402
    FIGS, OUT, figure_slide, new_deck, section_slide, stat_slide, title_slide,
)


def build() -> Path:
    prs = new_deck()

    title_slide(
        prs,
        "Delitos de Lima Metropolitana",
        "Análisis exploratorio del dataset 2018-2025\n"
        "Policía Nacional del Perú  ·  3 137 221 denuncias, 58 columnas",
        "César Pajuelo  ·  figuras de notebooks/eda_delitos_lima.ipynb",
    )

    stat_slide(prs, "El dataset de un vistazo", [
        ("3.14 M", "denuncias", "enero 2018 – junio 2025"),
        ("58", "columnas", "de las que el pipeline usa 6"),
        ("52", "valores de tipo_hecho", "y 35 subtipos relevantes"),
        ("43", "distritos", "Lima Metropolitana"),
    ])

    # ── 1 · estructura ────────────────────────────────────────────────────
    section_slide(
        prs, "estructura", "Qué columnas hacen falta",
        "De 58 columnas, seis. El resto son identificadores administrativos y "
        "jerarquía policial redundante.")

    figure_slide(
        prs, "Completitud de las columnas que se usan", FIGS / "01-nulos",
        bullets=[
            ("Coordenadas", "el campo más incompleto, y el que decide si una "
             "denuncia sirve."),
            ("direccion_hecho", "sirve de respaldo cuando falta la coordenada."),
            ("Las demás", "esencialmente completas."),
        ])

    # ── 2 · duplicados ────────────────────────────────────────────────────
    section_slide(
        prs, "limpieza", "id_dgc no es una clave",
        "Una denuncia con varios delitos imputados aparece repetida, una fila "
        "por combinación.")

    figure_slide(
        prs, "Una denuncia puede llevar varios delitos",
        FIGS / "02-duplicados",
        bullets=[
            ("Deduplicar por id_dgc", "borraría delitos distintos que comparten "
             "expediente."),
            ("La clave correcta", "(id_dgc, subtipo_hecho, modalidad_hecho): "
             "identifica el hecho dentro de la denuncia."),
        ])

    # ── 3 · categorías ────────────────────────────────────────────────────
    section_slide(
        prs, "alcance", "Qué delitos entran",
        "La unidad de análisis es la calle: solo sirven los hechos con "
        "localización con sentido sobre la red vial.")

    figure_slide(
        prs, "tipo_hecho: cinco familias de vía pública", FIGS / "03-tipos",
        bullets=[
            ("En rojo", "lo que entra al pipeline."),
            ("Se descartan", "los administrativos y los que se localizan en el "
             "domicilio de la víctima, no en el lugar del hecho."),
        ])

    figure_slide(
        prs, "Pero tipo_hecho es demasiado grueso para ser el eje",
        FIGS / "04-subtipos",
        bullets=[
            ("PATRIMONIO (DELITO)", "mete hurto, robo, extorsión y estafa en la "
             "misma caja."),
            ("El nivel equivalente", "al Primary Type de Chicago es "
             "subtipo_hecho: 35 tipos."),
            ("Decisión", "el alcance se decide sobre tipo_hecho; el eje de "
             "categorías es subtipo_hecho."),
        ])

    # ── 4 · geocodificación ───────────────────────────────────────────────
    section_slide(
        prs, "calidad", "¿Son de fiar las coordenadas?",
        "No todas, y el propio dataset lo dice.")

    figure_slide(
        prs, "Coordenadas que se repiten miles de veces",
        FIGS / "05-puntos-repetidos",
        bullets=[
            ("Los puntos más repetidos", "son centroides de comisaría, no "
             "lugares del hecho."),
            ("Si se dejan dentro", "aparece un hotspot en la puerta de cada "
             "comisaría."),
            ("El campo observacion", "los marca explícitamente."),
        ])

    figure_slide(
        prs, "La geocodificación se degrada, y en 2025 colapsa",
        FIGS / "11-geocodificacion",
        bullets=[
            ("2018", "92.5 % con coordenada real."),
            ("2024", "34.8 %."),
            ("2025", "7.0 %."),
            ("El resto", "recibe el centroide de su comisaría."),
        ])

    figure_slide(
        prs, "Consecuencia: 2025 no es comparable",
        FIGS / "12-crudo-vs-utilizable",
        bullets=[
            ("Solo el 4.5 %", "de las denuncias de 2025 llega al pipeline."),
            ("Los hotspots de 2025", "se calculan sobre esa fracción."),
            ("Al escribir", "decirlo, o truncar la ventana en 2024."),
        ])

    # ── 5 · patrones ──────────────────────────────────────────────────────
    section_slide(
        prs, "patrones", "Cómo se distribuye el delito",
        "En el tiempo, en el día y en el espacio.")

    figure_slide(
        prs, "La serie no baja: termina", FIGS / "06-serie-mensual",
        bullets=[
            ("2020", "el corte de COVID-19 es real y brusco."),
            ("2021-2024", "recuperación y crecimiento sostenido."),
            ("Junio 2025", "último mes con datos, no una caída."),
        ])

    figure_slide(
        prs, "Perfil horario y semanal", FIGS / "07-perfil-temporal",
        bullets=[
            ("Picos en horas redondas", "señal de hora imputada, no observada."),
            ("Conviene", "no leer el perfil horario como si fuera exacto."),
        ])

    figure_slide(
        prs, "Concentración por distrito", FIGS / "08-distritos",
        bullets=[
            ("Muy desigual", "y es esperable: San Juan de Lurigancho tiene un "
             "millón de habitantes."),
            ("Sin normalizar por población", "el ranking mide tamaño, no riesgo."),
        ])

    figure_slide(
        prs, "120 000 denuncias geolocalizadas", FIGS / "09-nube",
        caption="La forma de Lima sale sola. Las líneas rectas y los grumos "
                "regulares son artefactos de geocodificación.")

    # ── 6 · embudo ────────────────────────────────────────────────────────
    figure_slide(
        prs, "Del CSV al pipeline", FIGS / "10-embudo",
        bullets=[
            ("3.14 M", "denuncias en el CSV."),
            ("887 137", "llegan al pipeline."),
            ("La mayor pérdida", "no es el filtro de delitos: es la "
             "geocodificación."),
        ])

    stat_slide(prs, "Las cinco decisiones de limpieza", [
        ("6 / 58", "columnas", "el resto es jerarquía administrativa"),
        ("3 campos", "clave de dedupe", "id + subtipo + modalidad"),
        ("5 familias", "de vía pública", "eje de 35 subtipos"),
        ("centroides", "descartados", "inventarían un hotspot por comisaría"),
    ], "Todo ello está implementado en pipeline/ingest/profiles.py como un perfil "
       "declarativo, auditable sin tocar el código del pipeline.")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "eda_delitos_lima.pptx"
    prs.save(dst)
    return dst


if __name__ == "__main__":
    p = build()
    print(f"escrito {p}  ({p.stat().st_size / 1e6:.1f} MB)")
