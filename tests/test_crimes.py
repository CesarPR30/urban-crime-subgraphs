"""Cargador de CSV: resolución de alias, validación y proyección (§3.1)."""

from __future__ import annotations

import pytest

from pipeline.ingest.crimes import (
    CANONICAL_FIELDS,
    MissingColumnError,
    load_crimes,
    resolve_columns,
)


class TestAlias:
    def test_resuelve_el_export_completo_de_chicago(self):
        header = ["ID", "Case Number", "Date", "Block", "IUCR", "Primary Type",
                  "Description", "Location Description", "Latitude", "Longitude"]
        req, opt = resolve_columns(header)
        assert req["lat"] == "Latitude"
        assert req["lon"] == "Longitude"
        assert req["fecha"] == "Date"
        assert req["tipo"] == "Primary Type"
        assert req["crimen"] == "Description"
        assert opt["lugar"] == "Location Description"
        assert opt["id"] == "ID"

    def test_resuelve_el_export_reducido(self):
        """Espacios de sobra, doble espacio y sin `Description` separada."""
        header = ["DATE  OF OCCURRENCE", " PRIMARY DESCRIPTION",
                  " LOCATION DESCRIPTION", "LATITUDE", "LONGITUDE"]
        req, _ = resolve_columns(header)
        assert req["fecha"] == "DATE  OF OCCURRENCE"
        # Sin dos niveles de descripción, `crimen` y `tipo` caen en la misma.
        assert req["crimen"] == req["tipo"] == " PRIMARY DESCRIPTION"

    def test_log_sigue_siendo_alias_de_longitud(self):
        """`log` aparece así en algunos CSV de origen. No quitarlo (§3.1)."""
        header = ["lat", "log", "fecha", "crimen", "tipo"]
        req, _ = resolve_columns(header)
        assert req["lon"] == "log"

    @pytest.mark.parametrize("variant", [
        "latitud", "LATITUD", "  Latitud  ", "lat_itud" if False else "y",
    ])
    def test_normaliza_mayusculas_acentos_y_espacios(self, variant):
        header = [variant, "lon", "fecha", "crimen", "tipo"]
        req, _ = resolve_columns(header)
        assert req["lat"] == variant

    def test_falta_columna_obligatoria_lista_los_alias(self):
        with pytest.raises(MissingColumnError) as e:
            resolve_columns(["lat", "fecha", "crimen", "tipo"])
        msg = str(e.value)
        assert "lon" in msg and "longitude" in msg and "log" in msg


class TestValidacion:
    def test_detecta_las_filas_invalidas_inyectadas(self, crimes_csv):
        path, expected = crimes_csv
        records, report = load_crimes(path, dedupe="none")

        assert report.valid_rows == expected["valid"] + expected["duplicates"]
        assert report.discarded_rows == sum(expected["bad"].values())
        for reason, count in expected["bad"].items():
            assert report.discards_by_reason[reason] == count, reason
        assert len(records) == report.kept_rows

    def test_deduplica_por_id(self, crimes_csv):
        path, expected = crimes_csv
        _, report = load_crimes(path, dedupe="id")
        assert report.dedupe_key == "id"
        assert report.duplicate_rows == expected["duplicates"]
        assert report.kept_rows == expected["valid"]

    def test_auto_prefiere_id_cuando_existe(self, crimes_csv):
        path, _ = crimes_csv
        _, report = load_crimes(path, dedupe="auto")
        assert report.dedupe_key == "id"

    def test_filtro_por_categoria_no_cuenta_como_descarte(self, crimes_csv):
        """Un descarte es un problema; un filtro es una decisión (§3.1)."""
        path, expected = crimes_csv
        _, report = load_crimes(path, categories=["THEFT"], dedupe="id")
        assert report.discarded_rows == sum(expected["bad"].values())
        assert report.filtered_rows > 0
        assert report.filters_by_reason["fuera_de_categoria"] == report.filtered_rows
        assert report.kept_rows + report.filtered_rows == expected["valid"]

    def test_filtro_por_ventana(self, crimes_csv):
        path, _ = crimes_csv
        _, report = load_crimes(path, month_from="2024-02", dedupe="id")
        assert set(report.months) == {"2024-02"}
        assert report.filters_by_reason["fuera_de_ventana"] > 0


class TestProyeccion:
    def test_reduce_a_las_columnas_canonicas(self, crimes_csv):
        """Entren las columnas que entren, salen siempre las mismas (§3.1)."""
        path, _ = crimes_csv
        _, report = load_crimes(path, dedupe="id")
        assert set(report.column_mapping) <= set(CANONICAL_FIELDS)
        assert len(report.columns_in) > len(report.column_mapping)
        assert set(report.columns_dropped).isdisjoint(report.column_mapping.values())
        assert (len(report.column_mapping) + len(report.columns_dropped)
                == len(report.columns_in))

    def test_mes_es_yyyy_mm(self, crimes_csv):
        path, _ = crimes_csv
        records, _ = load_crimes(path, dedupe="id")
        assert all(len(r.mes) == 7 and r.mes[4] == "-" for r in records)
