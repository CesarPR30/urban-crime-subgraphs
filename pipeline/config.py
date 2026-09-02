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
from pydantic import BaseModel, Field, field_validator

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

    @field_validator("month_from", "month_to")
    @classmethod
    def _month_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 7 or v[4] != "-" or not (v[:4] + v[5:]).isdigit():
            raise ValueError(f"mes debe ser YYYY-MM, recibido {v!r}")
        return v


class NetworkCfg(BaseModel):
    place: str
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


class HotspotsCfg(BaseModel):
    alpha: Tunable = 0.3
    top_k: int = Field(20, gt=0)
    f_min_ratio: Tunable = 0.10

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

    #: Raíz del repo. No viene del YAML; la fija `load_config`.
    root: Path = ROOT

    def path(self, p: Path | str) -> Path:
        """Resuelve una ruta del config contra la raíz del repo."""
        p = Path(p)
        return p if p.is_absolute() else self.root / p


@lru_cache(maxsize=4)
def load_config(path: Path | str | None = None) -> Config:
    """Carga y valida `config.yaml`. Cacheado por ruta."""
    path = Path(path) if path else DEFAULT_CONFIG
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config.model_validate(raw)
    cfg.root = Path(path).resolve().parent
    return cfg
