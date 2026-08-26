"""Convenciones de rutas, caché de artefactos y manifest (PROJECT_SPEC §2).

Cada paso del pipeline cachea su salida y detecta si su entrada cambió, para
saltarse el recómputo. La detección se basa en una **clave de caché**: el hash
de los parámetros de config que afectan a ese paso, más el `mtime` y el tamaño
de sus archivos de entrada. Si la clave coincide con la registrada en el
manifest, la salida en disco sirve.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------------------- #
# Claves de caché
# --------------------------------------------------------------------------- #


def hash_obj(obj: Any) -> str:
    """Hash estable (sha256, 16 hex) de cualquier objeto serializable a JSON."""
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def stamp(path: Path) -> dict[str, Any]:
    """Huella barata de un archivo: existencia, tamaño y mtime."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}


def cache_key(params: Any, inputs: list[Path] | None = None) -> str:
    """Clave de caché de un paso: sus parámetros más la huella de sus entradas."""
    return hash_obj({"params": params, "inputs": [stamp(p) for p in inputs or []]})


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


class Manifest:
    """Registro en disco de qué produjo cada paso y con qué clave.

    Es lo que permite que `crimepipe <paso>` sea idempotente: si la clave del
    paso no cambió y su salida sigue existiendo, no se recomputa.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Manifest corrupto en %s; se reinicia", self.path)
                self.data = {}

    def get(self, step: str) -> dict[str, Any] | None:
        return self.data.get(step)

    def is_fresh(self, step: str, key: str, outputs: list[Path]) -> bool:
        """¿La salida cacheada de `step` sigue siendo válida para `key`?"""
        rec = self.data.get(step)
        if not rec or rec.get("key") != key:
            return False
        return all(Path(p).exists() for p in outputs)

    def record(self, step: str, key: str, outputs: list[Path], **extra: Any) -> None:
        self.data[step] = {
            "key": key,
            "outputs": [str(p) for p in outputs],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Paths:
    """Rutas de todos los artefactos del pipeline, derivadas del config."""

    root: Path
    raw: Path
    interim: Path
    processed: Path

    @classmethod
    def from_config(cls, cfg: Any) -> "Paths":
        return cls(
            root=cfg.root,
            raw=cfg.path(cfg.data.raw_dir),
            interim=cfg.path(cfg.data.interim_dir),
            processed=cfg.path(cfg.data.processed_dir),
        )

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def manifest(self) -> Path:
        return self.interim / MANIFEST_NAME

    # Artefactos por paso
    @property
    def network(self) -> Path:
        return self.interim / "network.graphml"

    @property
    def snapped(self) -> Path:
        return self.interim / "snapped.parquet"

    @property
    def counts(self) -> Path:
        return self.interim / "node_month_counts.parquet"

    @property
    def density(self) -> Path:
        return self.interim / "density.parquet"

    @property
    def hotspots(self) -> Path:
        return self.interim / "hotspots.json"

    @property
    def calibration(self) -> Path:
        return self.interim / "calibration.json"

    @property
    def evaluation(self) -> Path:
        return self.processed / "evaluation_report.json"

    @property
    def evaluation_csv(self) -> Path:
        return self.processed / "evaluation_monthly.csv"

    @property
    def similarity(self) -> Path:
        return self.processed / "similarity.json"

    @property
    def embedding_2d(self) -> Path:
        return self.processed / "embedding_2d.json"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def save_json(path: Path, obj: Any, *, indent: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def timed(label: str, logger_: logging.Logger | None = None) -> Iterator[None]:
    """Cronometra un bloque y lo reporta (§10: cada paso reporta cuánto tardó)."""
    log = logger_ or logger
    t0 = time.perf_counter()
    log.info("%s…", label)
    yield
    log.info("%s: %.2f s", label, time.perf_counter() - t0)
