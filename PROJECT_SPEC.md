# Urban Crime Subgraphs — Especificación de construcción

> Documento de arranque para Claude Code. Reconstrucción desde cero del pipeline
> de análisis de crimen urbano sobre red vial. Léelo completo antes de escribir
> código. Al final hay un plan por fases: **construye una fase a la vez y no
> avances hasta que pasen sus criterios de aceptación.**

---

## 0. Decisiones ya tomadas

Si alguna no coincide con lo que quiero, la corrijo antes de empezar.

| Decisión | Valor |
|---|---|
| Lenguaje del pipeline | Python 3.11+ |
| Gestor de dependencias | `uv` (fallback: `pip` + `requirements.txt`) |
| Grafos | `networkx` + `osmnx` |
| Geometría | `shapely`, `geopandas`, `pyproj` |
| Datos tabulares | `pandas`, `pyarrow` (formato de intercambio: **Parquet**) |
| Embeddings | `karateclub` (graph2vec, GL2Vec, FEATHER), `gensim` (Doc2Vec) |
| Proyección / clustering | `umap-learn`, `hdbscan`, `scikit-learn` |
| Dashboard | Frontend separado: **React + Vite + TypeScript + Mapbox GL JS**, consumiendo artefactos estáticos |
| Repo | Monorepo: `pipeline/` (Python) + `dashboard/` (TS) |
| Tests | `pytest` |
| Config | Un solo `config.yaml` en la raíz, tipado con `pydantic` |

**Regla de oro del proyecto:** el pipeline produce **artefactos estáticos**
(Parquet + JSON). El dashboard **solo lee** esos artefactos. No hay backend, no
hay base de datos, no hay cómputo en el navegador más allá de renderizar.

---

## 1. Qué se está construyendo y por qué

El objetivo científico es aislar el efecto de los **puntos de interés (POIs)**
sobre la distribución espacial del crimen, controlando la **topología de la red
vial**. La forma de hacerlo es un **diseño comparativo**: encontrar pares de
zonas urbanas estructuralmente equivalentes y contrastar su composición de POIs
contra su intensidad de crimen.

El pipeline tiene tres fases y seis pasos:

```
FASE 1 · Detección de hotspots
  (1) Campo de densidad de crimen sobre la red vial
  (2) Extracción de los top-20 subgrafos por mes

FASE 2 · Búsqueda de hotspots similares
  (3) Atributos: topología + historial de crímenes + POIs
  (4) Feature vector (embedding) por hotspot
  (5) Agrupamiento y proyección 2D

FASE 3 · Análisis de similitud
  (6) Contraste crimen vs. infraestructura entre zonas emparejadas
```

### Restricción de diseño que NO se puede violar

**Ningún dato de crimen entra en la representación estructural usada para medir
similitud entre hotspots.** El crimen es la variable de resultado a contrastar.
Si entra en el descriptor, dos zonas salen "similares" *porque* tienen crimen
parecido, y todo el diseño comparativo se vuelve circular.

Esto tiene una consecuencia concreta que hay que respetar en el código: el
descriptor de similitud y el descriptor de reporte son **dos objetos distintos**.
Ver §5.1.

---

## 2. Estructura del repositorio

```
.
├── config.yaml
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/            # descargas crudas (gitignored)
│   ├── interim/        # artefactos intermedios cacheados (gitignored)
│   └── processed/      # artefactos finales que consume el dashboard
├── pipeline/
│   ├── __init__.py
│   ├── config.py       # modelos pydantic + carga de config.yaml
│   ├── io.py           # lectura/escritura de artefactos, convenciones de path
│   ├── ingest/
│   │   ├── crimes.py   # carga + validación del CSV de crímenes
│   │   ├── pois.py     # descarga + categorización de POIs
│   │   └── network.py  # descarga + simplificación de la red vial
│   ├── snapping.py     # proyección de crímenes al grafo (arista → nodo)
│   ├── density.py      # paso 1: campo escalar gaussiano geodésico
│   ├── hotspots.py     # paso 2: join tree + persistencia
│   ├── features/
│   │   ├── structural.py   # paso 3: descriptor manual de 14 dims
│   │   ├── embeddings.py   # paso 4: graph2vec y alternativas
│   │   ├── hierarchy.py    # huella de jerarquía vial
│   │   └── poi_context.py  # enriquecimiento con POIs
│   ├── similarity.py   # paso 5: fusión, coseno, top-K, UMAP, HDBSCAN
│   ├── compare.py      # paso 6: contraste entre zonas emparejadas
│   ├── evaluate.py     # métricas + línea base BFS
│   └── cli.py          # entrypoint: `crimepipe <comando>`
├── tests/
│   ├── fixtures/       # grafos sintéticos pequeños + CSVs mínimos
│   └── test_*.py
└── dashboard/
    ├── package.json
    ├── public/data/    # symlink o copia de data/processed/
    └── src/
```

### CLI

Cada paso es invocable de forma independiente y cachea su salida:

```bash
crimepipe ingest        # descarga red vial + POIs, valida CSV de crímenes
crimepipe snap          # proyecta crímenes al grafo
crimepipe density       # calcula el campo escalar por mes
crimepipe hotspots      # extrae los top-K subgrafos por mes
crimepipe features      # descriptores + embeddings
crimepipe similarity    # top-K similares, UMAP, HDBSCAN
crimepipe evaluate      # métricas + comparación contra BFS
crimepipe export        # escribe data/processed/ para el dashboard
crimepipe all           # todo en orden
```

Cada comando debe ser **idempotente** y detectar si su entrada no cambió
(hash del config relevante + mtime de las entradas) para saltarse el recómputo,
con `--force` para ignorar la caché.

---

## 3. Datos

### 3.1 Crímenes

Fuente: portal de datos abiertos de la Ciudad de Chicago
(`Crimes – 2001 to Present`). Ventana de trabajo: **2024-01 a 2025-12**.

El cargador debe aceptar **cualquier CSV** que tenga columnas equivalentes a
las cinco canónicas, resolviendo alias sin distinguir mayúsculas ni espacios:

| Canónica | Alias aceptados | Tipo |
|---|---|---|
| `lat` | lat, latitude, latitud, y | float, −90..90 |
| `lon` | lon, **log**, lng, long, longitude, longitud, x | float, −180..180 |
| `fecha` | fecha, date, datetime, fecha_hora, occurred_at | datetime parseable |
| `crimen` | crimen, crime, delito, incidente, incident | string no vacío |
| `tipo` | tipo, type, crime_type, tipo_delito, categoria | string no vacío |

`log` es alias real de longitud: aparece así en algunos de mis CSVs. No lo quites.

El cargador devuelve `(df_limpio, ValidationReport)`. El reporte incluye:
total de filas, filas válidas, filas descartadas, mapeo de columnas resuelto, y
un conteo de descartes **por razón** (`lat_invalida`, `lon_invalida`,
`fecha_invalida`, `crimen_vacio`, `tipo_vacio`). Si falta una columna
obligatoria, lanza `ValueError` listando los alias aceptados.

Categorías de análisis: `theft`, `assault`, `robbery`, `motor vehicle theft`.
Agregación temporal: **mes calendario** (`YYYY-MM`).

### 3.2 Puntos de interés

Fuente: OpenStreetMap vía OSMnx, sobre un vocabulario fijo de etiquetas
`amenity`, `shop`, `leisure`. Cada POI se reduce a un punto (centroide si la
geometría no es puntual) y se mapea a una de **ocho categorías funcionales**:

`food`, `nightlife`, `education`, `health`, `police`, `finance`, `retail`, `park`

El mapeo etiqueta-OSM → categoría vive en un YAML versionado
(`pipeline/ingest/poi_taxonomy.yaml`), no hardcodeado. Debe ser fácil de auditar
y de extender a otra ciudad.

### 3.3 Red vial

Red **transitable** (`network_type="drive"`) de OSMnx, con topología
simplificada: cada arista es un segmento real entre dos puntos de decisión
(intersección o cul-de-sac).

- Nodos: coordenadas geográficas (`x` = lon, `y` = lat), convención OSMnx.
- Aristas: longitud en metros + clase `highway` de OSM, plegada a una jerarquía
  ordenada: `expressway`, `avenue`, `collector`, `street`, `service`.

La descarga es lenta y no determinista en el tiempo. **Cachea el grafo en disco
(GraphML) y no lo vuelvas a descargar salvo `--force`.** Registra la fecha de
descarga en un `manifest.json`.

Todas las fuentes comparten marco de referencia WGS84. Para cualquier cálculo
métrico, proyecta a un CRS local en metros (UTM de la zona, o plano
equirrectangular local centrado en el bbox).

---

## 4. Fase 1 — Detección de hotspots

### 4.0 Snapping de crímenes al grafo

Dos pasos, **y el orden importa**:

1. Localizar la **arista** geométricamente más cercana al punto y proyectar el
   punto sobre ella.
2. Asignar el crimen al **más cercano de los dos nodos extremos** de esa arista.

Ir directo al nodo más cercano asignaría crímenes a nodos de otra calle. Esto no
es un detalle: es una fuente sistemática de error en la construcción del campo.

Distancias con haversine:

```
d(p₁, p₂) = 2R · arctan(√a / √(1−a))
a = sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)
```

con `φ` latitud y `λ` longitud en radianes, `R = 6_371_000 m`.

**Rendimiento:** una búsqueda lineal sobre todas las aristas por cada crimen es
O(n·m) y no escala a 213 mil crímenes. Usa un índice espacial (STRtree de
shapely, o `osmnx.distance.nearest_edges` que ya está vectorizado). Este paso
debe correr en minutos, no en horas.

Salida: por cada nodo `v` y cada mes `m`, un conteo `c_m(v)`.

### 4.1 Campo de densidad escalar (paso 1)

Para cada mes, kernel gaussiano difundido **sobre el grafo**:

```
f(v) = Σ_{s∈V} c(s) · exp( −d_G(s,v)² / (2σ²) )    para  d_G(s,v) ≤ r
```

| Símbolo | Significado |
|---|---|
| `c(s)` | crímenes asignados al nodo `s` ese mes |
| `d_G(s,v)` | distancia geodésica **a lo largo de las aristas** (Dijkstra acotado), **no euclidiana** |
| `σ` | ancho de banda, `120 m` (≈ una cuadra corta) |
| `r` | truncamiento, `3σ` (más allá la gaussiana aporta <1 %) |

`d_G` es lo que hace que la difusión respete la red: no atraviesa manzanas ni
cruza el río. Implementa con `networkx.single_source_dijkstra_path_length` con
`cutoff=r`, iterando solo sobre nodos con `c(s) > 0`.

### 4.2 Segmentación por join tree (paso 2)

Barrido descendente sobre `f`, equivalente al join tree de los conjuntos de
super-nivel. Procesa nodos en orden decreciente de `f`, con union-find:

- **sin vecino procesado** → máximo local, abre una componente (hotspot candidato)
- **exactamente un vecino** → punto regular, extiende esa componente
- **dos o más** → silla de unión; sobrevive la rama de pico más alto, las otras mueren ahí

Por construcción cada región es un **subgrafo conexo**.

### 4.3 Simplificación por persistencia

```
π(m) = f(m) − f(s_m)          persistencia del máximo m
τ    = α · maxᵥ f(v)          umbral, con α = 0.3
```

Todo máximo con `π(m) < τ` se absorbe en la rama más profunda con la que se
fusionó; sigue la cadena de fusión hasta llegar a un máximo persistente. Los
nodos por debajo de un piso de densidad `f_min` quedan fuera del barrido para
mantener las regiones compactas.

Cada región sobreviviente es un hotspot. Se puntúan por crimen crudo capturado
(`Σ_{v∈región} c(v)`), se rankean, y se retienen los **top-K = 20 por mes**. La
semilla del hotspot es su nodo de crimen más denso.

**Salida por hotspot:** id, mes, conjunto de nodos, conjunto de aristas, crímenes
totales, desglose por categoría, nodo semilla.

---

## 5. Fase 2 — Caracterización y similitud

### 5.1 Dos descriptores distintos — no los mezcles

```python
# Entra en el cálculo de similitud. SIN datos de crimen.
StructuralDescriptor   # 14 dims: conectividad + geometría métrica
GraphEmbedding         # graph2vec sobre la estructura del subgrafo
RoadHierarchyFingerprint

# Solo para reporte y para el contraste final. NUNCA en similitud.
CrimeProfile           # totales, desglose por categoría, serie temporal
POIProfile             # distribución sobre 8 categorías, densidad, entropía
```

Escribe un test que falle si alguna feature derivada de crimen aparece en el
vector que alimenta la similitud coseno. Es la invariante más importante del
proyecto.

> **Nota sobre el código anterior:** el descriptor de 14 dims incluía dos
> dimensiones derivadas de crimen (fracción de crímenes en el nodo pico e
> intensidad log por nodo). Eso contamina la similitud. En esta reconstrucción,
> **quítalas del vector de similitud** y muévelas a `CrimeProfile`. El bloque de
> conectividad queda en 8 dims y el descriptor total en **12 dims**. Si al
> comparar contra los resultados anteriores esto mueve los emparejamientos, el
> resultado nuevo es el correcto.

### 5.2 Descriptor estructural

**Bloque de conectividad (8 dims)**
- `log |V_h|`, `log |E_h|`
- densidad `2|E_h| / (|V_h|(|V_h|−1))`
- grado medio
- histograma de grados: fracción de nodos con grado 1, 2, 3, 4+

**Bloque de geometría métrica (4 dims)** — calculado tras proyectar las
coordenadas de los nodos a un plano local en metros
- `log` de la longitud media de arista
- coeficiente de variación de las longitudes
- `log` del radio de giro
- **elongación**: `√(λ₂/λ₁)` con `λ₁ ≥ λ₂ ≥ 0` los autovalores de la matriz de
  covarianza de las coordenadas. `≈0` = huella lineal (una avenida),
  `≈1` = isótropa o acodada (un damero).

El bloque métrico existe porque la conectividad sola no distingue una cadena
recta de una en forma de L: ambas son el grafo camino `P_n` con idéntico
histograma de grados.

Normalización **min–max a [0,1]** por columna sobre todo el corpus, antes de
comparar.

### 5.3 Embedding aprendido

**graph2vec** = reetiquetado Weisfeiler–Lehman + Doc2Vec (PV-DBOW).

```
ℓ⁽ᵗ⁺¹⁾(v) = hash( ℓ⁽ᵗ⁾(v) ‖ sort{ ℓ⁽ᵗ⁾(u) : u ∈ N(v) } )
```

con `ℓ⁽⁰⁾(v) = deg(v)`, `‖` concatenación y `sort` el orden canónico del
multiconjunto de vecinos — que es lo que da **invariancia a la numeración de los
nodos**.

El multiconjunto de etiquetas WL sobre todas las iteraciones es el "documento"
del hotspot. El corpus son **todos** los hotspots de la ventana, y el ajuste es
**conjunto sobre todo el corpus a la vez** — no hotspot por hotspot. Solo así
los motivos estructurales recurrentes reciben vectores cercanos.

Fija `random_state` en todo: Doc2Vec, UMAP y HDBSCAN son estocásticos y el
proyecto tiene que ser reproducible.

**Métodos alternativos**, todos detrás de la misma interfaz
`Embedder.fit_transform(graphs) -> np.ndarray`, seleccionables por config:

| Método | Qué captura |
|---|---|
| `graph2vec` | subárboles enraizados de radio *t* (**por defecto**) |
| `gl2vec` | extiende graph2vec al *line graph*: incorpora info de aristas |
| `feather` | funciones características de atributos de nodo difundidos |
| `gcn` | representación aprendida por convolución + readout global |
| `netlsd` | espectro del laplaciano vía firma heat-kernel (**fallback**) |

El fallback a `netlsd` + PCA debe activarse automáticamente si falta una
dependencia opcional, con un warning claro. Nunca debe fallar la generación de
coordenadas para el dashboard.

### 5.4 Huella de jerarquía vial

Distribución normalizada de las aristas del subgrafo sobre las cinco clases
viales, más la participación de la clase dominante. Separa dos subgrafos con la
misma forma pero construidos con arterias frente a calles residenciales.

### 5.5 Fusión, similitud y proyección

1. Estandariza cada bloque por separado (estructural, embedding, jerarquía).
2. Pondera cada bloque con pesos de config (por defecto `1.0`, `1.0`, `0.5`).
3. Concatena.
4. Similitud coseno: `sim(hᵢ,hⱼ) = (xᵢ·xⱼ) / (‖xᵢ‖‖xⱼ‖)`. Mide ángulo, no
   magnitud: dos zonas de distinto tamaño con la misma forma siguen siendo
   similares.
5. Pre-calcula y almacena los **top-5 más similares** de cada hotspot sobre toda
   la ventana.
6. Proyecta a 2D con **UMAP** bajo métrica coseno; agrupa con **HDBSCAN**.

### 5.6 Enriquecimiento con POIs

- Los POIs se asocian a un subgrafo por contención en la **envolvente convexa
  bufferizada** de sus nodos (buffer configurable, por defecto 50 m).
- Distribución normalizada `p_k` sobre las 8 categorías.
- Densidad de POIs por nodo.
- **Entropía de Shannon**: `H = −Σ_{k=1..8} p_k log p_k` — diversidad funcional.
  `H` alto = mezcla de usos; `H` bajo = zona monofuncional.

Estas son las variables que se contrastan en la fase 3.

---

## 6. Fase 3 — Comparación

Dado un hotspot de referencia y uno de sus similares, produce el contraste:

- delta de crímenes totales y por categoría
- evolución temporal del *footprint* de ambos
- distribución de POIs lado a lado, con las categorías presentes en solo uno de
  los dos marcadas explícitamente (`solo A` / `solo B`)
- delta de entropía funcional

La pregunta que responde: manteniendo la estructura aproximadamente constante,
¿cuánto de la variación restante en intensidad de crimen se alinea con
diferencias en composición de POIs?

---

## 7. Evaluación y criterios de aceptación

### 7.1 Métricas

Por mes y en total, para el extractor y para la línea base:

1. crímenes crudos dentro de la unión de los top-20 subgrafos
2. cobertura como fracción de todos los crímenes de ese mes
3. **huella total de nodos** (compacidad)
4. **densidad de crímenes por nodo**

(3) y (4) no son opcionales. Sin ellas, cualquier método gana la métrica de
cobertura simplemente haciendo los subgrafos más grandes.

### 7.2 Línea base obligatoria

Implementa un `region growing` voraz por BFS desde los nodos más calientes, con
el mismo `K = 20` por mes, sobre los mismos datos snappeados. Se compara cabeza a
cabeza contra el extractor topológico.

### 7.3 Números que debe reproducir

El código anterior produjo estos resultados sobre Chicago 2024-01 → 2025-12. La
reconstrucción debe llegar a valores **equivalentes dentro de un margen
razonable** (±2 % en los agregados). Si se desvía mucho, hay un bug —
investígalo antes de seguir.

| Métrica | Valor esperado |
|---|---|
| Crímenes snappeados a la red | 213 602 |
| Nodos en la red vial | 29 537 |
| Subgrafos extraídos (20 × 24 meses) | 480 |
| Crímenes en los top-20/mes | 26 645 |
| Cobertura | 12.5 % |
| Nodos en subgrafos (suma 24 meses) | 8 562 (≈357/mes = 1.2 % de la red) |
| Densidad en hotspots | 3.11 cr/nodo |
| Media de la ciudad | ≈0.30 cr/nodo/mes |
| *Lift* | ≈10× |
| Nodo más caliente | 59 crímenes |
| Cobertura mensual | media 12.4 %, mín 9.9 % (2025-01), máx 13.7 % (verano) |

**Comparación contra BFS:**

| Método | Crímenes | Cobertura | cr/nodo |
|---|---|---|---|
| Topológico | 26 645 | 12.5 % | 3.11 |
| BFS | 23 064 | 10.8 % | 3.02 |
| Ganancia | +3 581 (+15.5 %) | +1.7 pts | ≈ igual |

La lectura correcta de esa tabla: a **igual huella de nodos y densidad
prácticamente idéntica**, el método topológico captura 15.5 % más crimen. La
mejora no viene de inflar los subgrafos.

**Advertencia:** el conteo de nodos de la red y algunos agregados pueden moverse
si OSM cambió desde la última descarga. Si difieren, no ajustes parámetros para
forzar el número — documenta la fecha del snapshot de OSM en el manifest y
reporta ambos.

### 7.4 Salida de evaluación

`crimepipe evaluate` escribe un `evaluation_report.json` + un CSV con la serie
mensual completa (24 filas: mes, crímenes del mes, capturados, cobertura, nodos,
densidad, para ambos métodos). Esa serie mensual me falta y la quiero.

---

## 8. Artefactos para el dashboard

`crimepipe export` escribe en `data/processed/`:

| Archivo | Contenido |
|---|---|
| `network.json` | nodos y aristas de la red (o tiles si pesa demasiado) |
| `hotspots.json` | un registro por hotspot: id, mes, nodos, aristas, crímenes, desglose, bbox |
| `similarity.json` | top-5 similares por hotspot, con score |
| `embedding_2d.json` | coordenadas UMAP + etiqueta HDBSCAN, por método de embedding |
| `poi_profiles.json` | perfil de POIs por hotspot |
| `timeline.json` | serie mensual global de crímenes |
| `manifest.json` | versión, fecha de snapshot OSM, config usada, hashes |

Vigila el peso: 213 mil puntos crudos no van al navegador sin agregación.
Pre-agrega lo que se pueda y considera formato binario o tiles si `network.json`
supera unos pocos MB.

### Los cinco paneles

- **A · Mapa principal** — subgrafo seleccionado sobre la red real, con los
  eventos. Alterna puntos / mapa de calor / mapa de nodos. Toggle de POIs.
- **B · Panel descriptivo** — crímenes, nodos, donut de tipos de delito, donut de
  POIs de la zona.
- **C · Espacio de embeddings** — scatter UMAP, coloreable por tipo / nº de
  delitos / clúster, conmutable entre métodos de embedding, con selección por
  lazo.
- **D · Línea de tiempo global** — serie mensual, con marcadores del hotspot
  seleccionado y sus similares; arrastrable para filtrar.
- **E · Comparación A vs B** — el corazón del diseño. Dos subgrafos contrastados
  por nº de delitos, evolución del footprint, tipos de crimen y distribución de
  POIs por uso de suelo, marcando las categorías exclusivas de cada uno.

---

## 9. Plan por fases

Construye en este orden. **No pases a la siguiente fase hasta que la actual pase
sus criterios.**

### Fase A · Cimientos
Estructura del repo, `config.yaml` + pydantic, `io.py`, CLI esqueleto, harness de
tests con fixtures sintéticas (una grilla vial pequeña + un CSV de ~200
crímenes generado con seed fijo).
✅ `pytest` pasa · `crimepipe --help` funciona.

### Fase B · Ingesta y snapping
Cargador de CSV con alias y `ValidationReport`. Descarga y caché de red vial y
POIs. Snapping arista→nodo con índice espacial.
✅ Test de que un punto entre dos calles paralelas se asigna a la correcta ·
snapping de los 213 602 crímenes en < 5 min · el `ValidationReport` detecta filas
inválidas inyectadas a propósito.

### Fase C · Detección de hotspots
Campo de densidad geodésico, join tree con union-find, persistencia, top-K.
✅ Sobre la fixture sintética, las regiones son conexas y no se solapan ·
sobre Chicago, los números de §7.3 caen dentro del margen.

### Fase D · Evaluación
Línea base BFS, las cuatro métricas, serie mensual, reporte.
✅ Reproduce la tabla de comparación de §7.3 · genera el CSV mensual de 24 filas.

### Fase E · Features y similitud
Descriptor de 12 dims, graph2vec, jerarquía vial, POIs, fusión, coseno, top-5,
UMAP, HDBSCAN.
✅ **El test de no-contaminación por crimen pasa** · dos subgrafos isomorfos con
distinta numeración de nodos obtienen el mismo embedding · resultados
reproducibles entre corridas con el mismo seed.

### Fase F · Exportación y dashboard
Artefactos estáticos, luego los cinco paneles.
✅ El dashboard carga sin backend y los cinco paneles funcionan.

---

## 10. Convenciones

- **Type hints en todo.** `mypy` en modo razonable.
- **Sin notebooks en el repo.** Si necesito explorar, es un script en `scripts/`.
- **Logging con `logging`**, no `print`. Cada paso reporta cuántas entidades
  procesó y cuánto tardó.
- **Seeds fijos y explícitos.** Todo lo estocástico toma `random_state` del
  config.
- **Nada de constantes mágicas en el código.** `σ`, `α`, `K`, `f_min`, el buffer
  de POIs, los pesos de fusión y las dimensiones del embedding viven en
  `config.yaml`.
- **Docstrings con la fórmula** en las funciones que implementan una: densidad,
  persistencia, elongación, WL, entropía, coseno. Las voy a citar en la tesis y
  tienen que coincidir con lo que hace el código.
- **Sin dependencias pesadas nuevas** sin decírmelo antes.

## 11. Errores del código anterior que no quiero repetir

1. **Contaminación por crimen** en el descriptor de similitud (§5.1). El más
   grave.
2. **Snapping directo al nodo** en vez de arista→nodo. Asigna crímenes a la calle
   equivocada.
3. **Búsqueda lineal de la arista más cercana.** Inviable a escala real.
4. **Parámetros hardcodeados** dispersos por los módulos, lo que hizo imposible
   el análisis de sensibilidad de `σ` y `α`.
5. **Sin seeds fijos**, de modo que la proyección UMAP cambiaba entre corridas y
   las capturas del dashboard no eran reproducibles.
6. **Cómputo pesado en el navegador**, en lugar de artefactos pre-calculados.
7. **Sin línea base**, así que durante meses no hubo forma de saber si el método
   topológico era mejor que lo obvio.

---

## 12. Primera tarea

Empieza por la **Fase A**. Antes de escribir código:

1. Propón la estructura de `config.yaml` completa, con todos los parámetros de
   este documento y sus valores por defecto.
2. Propón las firmas de los objetos de dominio principales (`Hotspot`,
   `StructuralDescriptor`, `ValidationReport`, `EvaluationResult`).
3. Espera mi visto bueno antes de implementar.

Si algo de este documento es ambiguo o contradictorio, pregúntame en vez de
elegir por tu cuenta.
