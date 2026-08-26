"""Puntos de interés: taxonomía y perfil por subgrafo (§3.2, §5.6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.features.pois import PoiProfile, assign, hulls, shannon
from pipeline.features.structural import Subgraph, local_epsg
from pipeline.ingest.pois import Taxonomy

CATS = ["food", "nightlife", "retail", "education",
        "health", "finance", "park", "police"]


@pytest.fixture
def taxonomy():
    """La taxonomía real del repo: es lo que hay que auditar, no una copia."""
    cfg = load_config()
    return Taxonomy.load(cfg.path(cfg.pois.taxonomy))


# --------------------------------------------------------------------------- #
# §3.2 — taxonomía
# --------------------------------------------------------------------------- #


def test_son_las_ocho_categorias_de_la_spec(taxonomy):
    assert sorted(taxonomy.categories) == sorted(CATS)


def test_cada_categoria_tiene_etiqueta_legible(taxonomy):
    for c in taxonomy.categories:
        assert taxonomy.labels.get(c), f"falta la etiqueta de {c}"


def test_clasifica_por_amenity_shop_y_leisure(taxonomy):
    assert taxonomy.classify({"amenity": "restaurant"}) == "food"
    assert taxonomy.classify({"amenity": "bar"}) == "nightlife"
    assert taxonomy.classify({"shop": "bakery"}) == "food"
    assert taxonomy.classify({"leisure": "park"}) == "park"
    assert taxonomy.classify({"amenity": "bank"}) == "finance"


def test_shop_desconocido_cae_en_comercio(taxonomy):
    """`shop` tiene cientos de valores; el default evita listarlos todos."""
    assert taxonomy.classify({"shop": "un_comercio_rarisimo"}) == "retail"


def test_amenity_gana_a_shop(taxonomy):
    """Un pub que también vende alcohol es ocio nocturno por las dos vías,
    pero el desempate tiene que ser estable y por `amenity`."""
    assert taxonomy.classify({"amenity": "pub", "shop": "alcohol"}) == "nightlife"
    assert taxonomy.classify({"amenity": "cafe", "shop": "books"}) == "food"


def test_el_mobiliario_urbano_se_excluye_a_proposito(taxonomy):
    """Distinguir `excluded` de desconocido es lo que hace útil al indicador."""
    for tag in ("parking", "bench", "post_box", "bicycle_parking", "waste_basket"):
        assert taxonomy.classify({"amenity": tag}) == Taxonomy.EXCLUDED, tag


def test_lo_no_mapeado_devuelve_none(taxonomy):
    assert taxonomy.classify({"amenity": "no_existe_esta_etiqueta"}) is None
    assert taxonomy.classify({}) is None


def test_una_exclusion_no_tapa_una_clasificacion_por_otra_llave(taxonomy):
    """Un centro comercial con aparcamiento es comercio, no un aparcamiento."""
    assert taxonomy.classify({"amenity": "parking", "shop": "mall"}) == "retail"


def test_nan_de_pandas_no_cuenta_como_etiqueta(taxonomy):
    """`iterrows` sobre un GeoDataFrame rellena las columnas ausentes con NaN."""
    assert taxonomy.classify({"amenity": float("nan"), "shop": "bakery"}) == "food"


def test_una_etiqueta_no_puede_estar_clasificada_y_excluida(tmp_path):
    p = tmp_path / "tax.yaml"
    p.write_text(
        "categories: [food]\n"
        "tags:\n  amenity:\n    food: [cafe]\n"
        "excluded:\n  amenity: [cafe]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="clasificados y excluidos"):
        Taxonomy.load(p)


def test_categoria_desconocida_en_el_yaml_es_un_error(tmp_path):
    p = tmp_path / "tax.yaml"
    p.write_text("categories: [food]\ntags:\n  amenity:\n    inventada: [x]\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="no está en `categories`"):
        Taxonomy.load(p)


# --------------------------------------------------------------------------- #
# §5.6 — entropía
# --------------------------------------------------------------------------- #


def test_entropia_maxima_con_distribucion_uniforme():
    counts = np.full(8, 10, dtype=np.int64)
    assert shannon(counts) == pytest.approx(math.log(8))


def test_entropia_cero_si_es_monofuncional():
    counts = np.zeros(8, dtype=np.int64)
    counts[3] = 42
    assert shannon(counts) == 0.0


def test_entropia_de_zona_vacia_es_cero():
    assert shannon(np.zeros(8, dtype=np.int64)) == 0.0


def test_mezclar_usos_sube_la_entropia():
    mono = np.array([100, 0, 0, 0, 0, 0, 0, 0])
    mixto = np.array([40, 30, 20, 10, 0, 0, 0, 0])
    assert shannon(mixto) > shannon(mono)


# --------------------------------------------------------------------------- #
# §5.6 — asociación por envolvente bufferizada
# --------------------------------------------------------------------------- #


def _sub(xy, epsg=32616, sid="h"):
    return Subgraph(id=sid, month="2024-01", nodes=list(range(len(xy))),
                    edges=[], xy=np.asarray(xy, dtype=np.float64),
                    edge_class=[], epsg=epsg)


def test_el_buffer_se_aplica_en_metros():
    """Una zona de 100 m con buffer 50 m debe medir 200 m de largo."""
    sg = _sub([[0, 0], [100, 0]])
    zone = hulls([sg], 50.0)[0]
    x0, y0, x1, y1 = zone.bounds
    assert x1 - x0 == pytest.approx(200, abs=1)
    assert y1 - y0 == pytest.approx(100, abs=1)


def test_asigna_solo_lo_que_cae_dentro():
    """Un POI a 500 m del subgrafo no es de su zona."""
    # UTM 16N, cerca de Chicago: se elige el mismo plano que usaría el pipeline.
    epsg = local_epsg([41.88], [-87.63])
    from pipeline.features.structural import transformer_for
    to_m = transformer_for(epsg)
    cx, cy = to_m(-87.63, 41.88)

    from pyproj import Transformer
    to_deg = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform

    sg = _sub([[cx, cy], [cx + 100, cy]], epsg=epsg)
    dentro = to_deg(cx + 50, cy)          # en medio del subgrafo
    fuera = to_deg(cx + 600, cy)          # a 500 m del extremo

    prof = assign([sg], [dentro[1], fuera[1]], [dentro[0], fuera[0]],
                  ["food", "food"], CATS, 50.0)[0]
    assert prof.total == 1
    assert prof.counts[CATS.index("food")] == 1


def test_un_poi_puede_pertenecer_a_varias_zonas():
    """Las envolventes de meses distintos se solapan por construcción; descontar
    el solape haría que el perfil de una zona dependiera de qué otras existen."""
    epsg = local_epsg([41.88], [-87.63])
    from pipeline.features.structural import transformer_for
    from pyproj import Transformer
    to_m = transformer_for(epsg)
    to_deg = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    cx, cy = to_m(-87.63, 41.88)

    a = _sub([[cx, cy], [cx + 100, cy]], epsg=epsg, sid="a")
    b = _sub([[cx + 40, cy], [cx + 200, cy]], epsg=epsg, sid="b")
    lon, lat = to_deg(cx + 60, cy)

    profs = assign([a, b], [lat], [lon], ["retail"], CATS, 50.0)
    assert profs[0].total == 1
    assert profs[1].total == 1


def test_perfil_de_zona_sin_pois():
    sg = _sub([[0, 0], [100, 0]])
    p = assign([sg], [], [], [], CATS, 50.0)[0]
    assert p.total == 0
    assert p.entropy == 0.0
    assert p.per_node == 0.0
    assert np.all(p.distribution() == 0.0)


def test_densidad_por_nodo():
    epsg = local_epsg([41.88], [-87.63])
    from pipeline.features.structural import transformer_for
    from pyproj import Transformer
    to_m = transformer_for(epsg)
    to_deg = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    cx, cy = to_m(-87.63, 41.88)

    sg = _sub([[cx, cy], [cx + 100, cy]], epsg=epsg)     # 2 nodos
    pts = [to_deg(cx + d, cy) for d in (10, 30, 50, 70)]  # 4 POIs dentro
    p = assign([sg], [q[1] for q in pts], [q[0] for q in pts],
               ["food"] * 4, CATS, 50.0)[0]
    assert p.total == 4
    assert p.per_node == pytest.approx(2.0)


def test_categoria_fuera_de_la_taxonomia_no_se_cuenta():
    epsg = local_epsg([41.88], [-87.63])
    from pipeline.features.structural import transformer_for
    from pyproj import Transformer
    to_m = transformer_for(epsg)
    to_deg = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    cx, cy = to_m(-87.63, 41.88)
    sg = _sub([[cx, cy], [cx + 100, cy]], epsg=epsg)
    lon, lat = to_deg(cx + 50, cy)
    p = assign([sg], [lat], [lon], ["categoria_inventada"], CATS, 50.0)[0]
    assert p.total == 0


def test_los_pois_no_entran_en_la_similitud():
    """§5.1 otra vez: `Subgraph` no puede llevar POIs, así que el descriptor no
    los puede ver ni por accidente."""
    campos = set(Subgraph.__dataclass_fields__)
    assert not (campos & {"pois", "poi_counts", "entropy", "poi_profile"})
