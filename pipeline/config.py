"""Modelos pydantic y carga de `config.yaml` (PROJECT_SPEC §10).

Un solo archivo de configuración tipado. Ningún módulo del pipeline define
constantes propias: σ, α, K, `f_min`, el buffer de POIs y los pesos de fusión
salen todos de aquí, para que el análisis de sensibilidad sea posible sin tocar
código.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


class ProjectCfg(BaseModel):
    name: str = "urban-crime-subgraphs"
    random_state: int = 42


class DataCfg(BaseModel):
    crimes_csv: Path
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    month_from: str | None = None
    month_to: str | None = None
    categories: list[str] | None = None
    dedupe: Literal["auto", "id", "content", "none"] = "auto"

    #: Nombre del perfil de `pipeline/ingest/profiles.py`. Decide el anclaje de
    #: columnas, el identificador compuesto y las exclusiones propias de la
    #: fuente. `generic` = todo por alias, que es como funcionaba antes de que
    #: existieran los perfiles.
    source_profile: str = "generic"

    #: Reglas de limpieza que no pueden decidirse fila a fila. Hoy solo
    #: `fallback_points`. Se pasan tal cual al cargador para que el umbral viva
    #: en el config y pueda barrerse, no enterrado en el perfil.
    cleaning: dict = Field(default_factory=dict)

    @field_validator("month_from", "month_to")
    @classmethod
    def _month_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 7 or v[4] != "-" or not (v[:4] + v[5:]).isdigit():
            raise ValueError(f"mes debe ser YYYY-MM, recibido {v!r}")
        return v


class NetworkCfg(BaseModel):
    place: str = ""
    network_type: str = "drive"
    simplify: bool = True
    cache: Path = Path("data/interim/network.graphml")
    hierarchy: dict[str, list[str]]
    hierarchy_default: str = "street"

    @property
    def hierarchy_order(self) -> list[str]:
        """Clases viales de mayor a menor jerarquía, en el orden del YAML."""
        return list(self.hierarchy)

    @property
    def osm_to_class(self) -> dict[str, str]:
        """Índice inverso: etiqueta `highway` de OSM -> clase de la jerarquía."""
        return {tag: cls for cls, tags in self.hierarchy.items() for tag in tags}


class SnappingCfg(BaseModel):
    max_distance_m: float = 200.0
    earth_radius_m: float = 6_371_000.0


#: Un parámetro puede fijarse a un número o delegarse a la calibración
#: automática con la cadena `auto` (ver `pipeline/calibrate.py`).
AUTO = "auto"
Tunable = float | Literal["auto"]


class DensityCfg(BaseModel):
    sigma_m: Tunable = 120.0
    truncate_sigmas: float = Field(3.0, gt=0)

    @property
    def is_auto(self) -> bool:
        return self.sigma_m == AUTO

    def resolved_sigma(self, calibration: dict | None = None) -> float:
        """σ efectivo: el del YAML, o el calibrado si está puesto en `auto`."""
        if self.sigma_m != AUTO:
            return float(self.sigma_m)
        if not calibration:
            raise ValueError(
                "density.sigma_m está en 'auto' pero no hay calibración. "
                "Ejecuta `crimepipe calibrate` primero."
            )
        return float(calibration["recommended"]["sigma_m"])

    def radius_for(self, sigma: float) -> float:
        """Radio de truncamiento `r = k·σ`."""
        return sigma * self.truncate_sigmas


class SelectionCfg(BaseModel):
    """Cuántas regiones retiene cada mes (ver `pipeline/selection.py`)."""

    method: Literal["fixed", "percentile", "montecarlo"] = "fixed"

    #: `fixed`: número de regiones. Se lee de `hotspots.top_k`, que sigue
    #: siendo el parámetro de la especificación; aquí solo se refleja.
    top_k: int = Field(20, gt=0)

    #: `percentile`: cuantil de la puntuación dentro del propio mes.
    percentile: float = Field(0.90, ge=0.0, lt=1.0)

    #: `montecarlo`: nivel de significancia y número de réplicas.
    alpha_sig: float = Field(0.05, gt=0.0, lt=1.0)
    replicates: int = Field(99, gt=0)
    null: Literal["uniform", "permutation"] = "uniform"

    #: Topes de seguridad, no objetivos. `max_k: null` = sin tope superior.
    min_k: int = Field(1, ge=0)
    max_k: int | None = 50


class HotspotsCfg(BaseModel):
    alpha: Tunable = 0.3
    top_k: int = Field(20, gt=0)
    f_min_ratio: Tunable = 0.10
    selection: SelectionCfg = SelectionCfg()

    @model_validator(mode="after")
    def _sync_top_k(self) -> "HotspotsCfg":
        """`top_k` vive en `hotspots`, no dentro de `selection`.

        Es el parámetro que nombra §4.2 y no tendría sentido duplicarlo en el
        YAML. Se copia hacia dentro para que `selection.resolve` reciba un
        objeto autosuficiente.
        """
        if "top_k" not in self.selection.model_fields_set:
            self.selection.top_k = self.top_k
        return self

    def resolved(self, key: str, calibration: dict | None = None) -> float:
        value = getattr(self, key)
        if value != AUTO:
            return float(value)
        if not calibration:
            raise ValueError(
                f"hotspots.{key} está en 'auto' pero no hay calibración. "
                "Ejecuta `crimepipe calibrate` primero."
            )
        return float(calibration["recommended"][key])


class BaselineCfg(BaseModel):
    method: str = "bfs"
    #: `greedy` es el `region growing voraz` de §7.2: la frontera se ordena por
    #: crimen del vecino. `bfs` es anchura pura, la variante ingenua.
    growth: Literal["greedy", "bfs"] = "greedy"
    #: Iguala la huella de cada región BFS a la media de las topológicas del
    #: mismo mes. Sin esto, comparar cobertura no mide nada (§7.2).
    match_footprint: bool = True
    #: Presupuesto fijo de nodos por región, solo si `match_footprint` es false.
    node_budget: int = Field(18, gt=0)


class PoisCfg(BaseModel):
    taxonomy: Path = Path("pipeline/ingest/poi_taxonomy.yaml")
    cache: Path = Path("data/interim/pois.parquet")
    buffer_m: float = 50.0

    #: `overpass` consulta el servicio en vivo; `geofabrik` descarga extractos
    #: estáticos por país. Overpass raciona y ha llegado a bloquear la IP
    #: durante días; Geofabrik es reproducible y además tiene histórico.
    source: Literal["overpass", "geofabrik"] = "overpass"

    #: Ruta del extracto en Geofabrik, sin extensión: `south-america/peru`.
    region: str | None = None

    #: Años cuyas instantáneas del 1 de enero se descargan. Vacío o `None`
    #: significa una sola instantánea, la más reciente disponible.
    years: list[int] = []

    #: Dónde se guardan los `.osm.pbf`. Pesan cientos de MB y no son artefactos
    #: del pipeline, así que viven fuera de `data/processed`.
    snapshots_dir: Path = Path("data/interim/osm")

    @model_validator(mode="after")
    def _check_geofabrik(self) -> "PoisCfg":
        if self.source == "geofabrik" and not self.region:
            raise ValueError(
                "pois.source: geofabrik exige `pois.region`, p.ej. "
                "'south-america/peru'"
            )
        return self


_EMBED_METHODS = Literal["graph2vec", "gl2vec", "feather", "gcn", "netlsd", "wwl", "scattering"]


class EmbeddingCfg(BaseModel):
    method: _EMBED_METHODS = "graph2vec"
    dimensions: int = 128
    wl_iterations: int = 2
    epochs: int = 50
    #: Métodos extra a calcular en la misma corrida para compararlos en el
    #: dashboard. El principal (`method`) va siempre; estos se añaden.
    compare: list[_EMBED_METHODS] = []


class FeaturesCfg(BaseModel):
    embedding: EmbeddingCfg = EmbeddingCfg()


class WeightsCfg(BaseModel):
    structural: float = 1.0
    embedding: float = 1.0
    hierarchy: float = 0.5


class UmapCfg(BaseModel):
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"


class HdbscanCfg(BaseModel):
    min_cluster_size: int = 5


class SimilarityCfg(BaseModel):
    weights: WeightsCfg = WeightsCfg()
    top_k_similar: int = 5
    umap: UmapCfg = UmapCfg()
    hdbscan: HdbscanCfg = HdbscanCfg()


class ReferenceCfg(BaseModel):
    """Los números de §7.3 contra los que se valida una ciudad.

    Son propiedad del dataset, no del método: §7.3 tabula lo que debe salir
    sobre Chicago 2024-2025. Una ciudad sin tabla de referencia —Lima— no tiene
    contra qué contrastarse, y el reporte debe decir eso en vez de restar
    contra los números de otra ciudad y publicar un delta sin sentido.
    """

    crimes_snapped: int
    network_nodes: int
    hotspots: int
    crimes_captured: int
    coverage: float
    nodes_in_hotspots: int
    density_hotspots: float
    hottest_node_crimes: int
    baseline_captured: int
    baseline_coverage: float
    baseline_density: float
    gain_absolute: int
    gain_relative: float


class Config(BaseModel):
    project: ProjectCfg = ProjectCfg()
    data: DataCfg
    network: NetworkCfg
    snapping: SnappingCfg = SnappingCfg()
    density: DensityCfg = DensityCfg()
    hotspots: HotspotsCfg = HotspotsCfg()
    baseline: BaselineCfg = BaselineCfg()
    pois: PoisCfg = PoisCfg()
    features: FeaturesCfg = FeaturesCfg()
    similarity: SimilarityCfg = SimilarityCfg()

    #: Tabla de validación de §7.3, si la ciudad tiene una. `None` = no hay
    #: contra qué contrastar y los reportes omiten la columna «Esperado».
    reference: ReferenceCfg | None = None

    #: Raíz del repo. No viene del YAML; la fija `load_config`.
    root: Path = ROOT

    #: Dataset activo. Lo fija `load_config` al fusionar; no está en el YAML
    #: dentro del bloque del dataset, sino que es la clave que lo nombra.
    dataset: str = "default"
    dataset_label: str = ""
    dataset_sublabel: str = ""

    def path(self, p: Path | str) -> Path:
        """Resuelve una ruta del config contra la raíz del repo."""
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def dashboard_data(self) -> Path:
        """Carpeta de artefactos que sirve el dashboard, propia del dataset.

        Cada ciudad escribe en la suya. Compartirlas fue el bug latente que
        hacía imposible tener dos ciudades a la vez: `crimes.bin` y
        `snapping.bin` se indexan por posición y llevan una huella para negarse
        a mezclarse, pero `hotspots.geojson` y `similarity.json` no, y se
        habrían pisado en silencio.
        """
        return self.root / "dashboard" / "public" / "data" / self.dataset


def _deep_merge(base: dict, over: dict) -> dict:
    """Fusiona `over` sobre `base` recursivamente, sin mutar ninguno.

    Un dict se funde clave a clave; cualquier otra cosa —incluidas las listas—
    se reemplaza entera. Reemplazar la lista es lo correcto para lo que hay
    aquí: `categories` es el recorte del estudio, y una ciudad que declara las
    suyas quiere *esas*, no las suyas añadidas a las de Chicago.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def available_datasets(path: Path | str | None = None) -> dict[str, dict]:
    """Los datasets declarados en el YAML, sin validar el resto del config."""
    path = Path(path) if path else DEFAULT_CONFIG
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("datasets") or {}


@lru_cache(maxsize=8)
def load_config(
    path: Path | str | None = None, dataset: str | None = None
) -> Config:
    """Carga y valida `config.yaml` para un dataset. Cacheado por (ruta, dataset).

    El YAML tiene dos mitades: los parámetros del método, comunes, y el bloque
    `datasets:`, donde cada ciudad declara lo suyo. Aquí se funde la segunda
    sobre la primera y se descarta `datasets:` del resultado, de modo que el
    `Config` que ve el pipeline sigue siendo el de una sola ciudad y ningún
    paso aguas abajo tiene que saber que hay más de una.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    datasets = raw.pop("datasets", None) or {}
    name = dataset or raw.pop("active_dataset", None) or "default"
    raw.pop("active_dataset", None)

    if datasets:
        if name not in datasets:
            raise ValueError(
                f"dataset {name!r} no está en config.yaml. "
                f"Disponibles: {', '.join(sorted(datasets))}"
            )
        block = dict(datasets[name])
        label = block.pop("label", name)
        sublabel = block.pop("sublabel", "")
        raw = _deep_merge(raw, block)
    else:
        label, sublabel = name, ""

    cfg = Config.model_validate(raw)
    cfg.root = Path(path).resolve().parent
    cfg.dataset = name
    cfg.dataset_label = label
    cfg.dataset_sublabel = sublabel
    return cfg
