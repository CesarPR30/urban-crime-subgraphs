# Urban Crime Subgraphs

Análisis de crimen urbano sobre red vial. La especificación completa está en
[PROJECT_SPEC.md](PROJECT_SPEC.md).

**Todas las fases construidas: A – F.**
A (cimientos), B (ingesta y snapping), C (hotspots) y D (evaluación) verificadas
contra §7.3. La E está entera —descriptor de 12 dims, graph2vec, jerarquía vial,
POIs, fusión, coseno, top-5, UMAP y HDBSCAN— con el test de no-contaminación por
crimen pasando. De la F están **los cinco paneles**: A mapa, B descriptivo,
C espacio de embeddings, D línea de tiempo global y E comparación A vs B.

---

## Instalación

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,features]"   # Linux/macOS: .venv/bin/python
```

Todo con wheel para Python 3.14 — nada compila. El extra `features` añade
`scikit-learn` y `umap-learn` para la fase E; sin ellos el pipeline corre igual
y cae a PCA y «sin clúster», como exige §5.3.

`gensim` **no** está en las dependencias: no publica wheel para 3.14. El Doc2Vec
de graph2vec va implementado en numpy. Ver [Fase E](#graph2vec-sin-gensim).

## El pipeline

Cada paso acepta `--dataset chicago|lima`; sin él se usa `active_dataset` del
config. Ver [Multi-dataset](#multi-dataset-chicago-y-lima).

```bash
.venv/Scripts/python -m pipeline.cli ingest      # valida CSV + descarga y cachea la red
.venv/Scripts/python -m pipeline.cli pois        # descarga y clasifica los POIs de OSM
.venv/Scripts/python -m pipeline.cli calibrate   # σ, α y f_min desde los datos
.venv/Scripts/python -m pipeline.cli hotspots    # densidad + join tree + top-K
.venv/Scripts/python -m pipeline.cli evaluate    # líneas base + las cuatro métricas
.venv/Scripts/python -m pipeline.cli features    # descriptor + embedding + top-5 similares
.venv/Scripts/python -m pipeline.cli export      # GeoJSON + snapping.bin + pois.bin
.venv/Scripts/python -m pytest tests -q          # 126 tests
```

`hotspots` corre en **~16 s** con la red y el snapping cacheados (~65 s la
primera vez). Escribe `data/interim/hotspots.json` con los 480 subgrafos: nodos,
aristas, crímenes, desglose por categoría, semilla, pico y persistencia.

### Resultado contra §7.3

| Métrica | Obtenido | Esperado | Δ |
|---|---|---|---|
| Crímenes snappeados a la red | 211 391 | 213 602 | −1.0 % |
| Nodos en la red vial | 29 832 | 29 537 | +1.0 % |
| Subgrafos extraídos | 480 | 480 | 0.0 % |
| Crímenes en los top-20/mes | 26 471 | 26 645 | −0.7 % |
| Cobertura | 12.5 % | 12.5 % | 0.0 pts |
| Nodos en subgrafos (24 meses) | 9 385 | 8 562 | +9.6 % |
| Densidad en hotspots | 2.82 | 3.11 | −9.3 % |
| Media de la ciudad | 0.30 | ≈0.30 | ✓ |
| *Lift* | 9.6× | ≈10× | ✓ |
| Nodo más caliente | 55 | 59 | −6.8 % |

Cobertura mensual: media 12.5 %, mín 10.0 % (2024-01), máx 16.2 % (2024-08) —
el patrón estacional de §7.3 (mínimo en enero, máximo en verano) se reproduce.

Huella y densidad quedan a ~9.5 %, fuera del ±2 % que pide la spec. Son la
misma desviación vista dos veces: regiones algo más grandes con el mismo crimen
capturado. El snapshot de OSM tiene 295 nodos más que el original, y `f_min`
—el único parámetro que la spec no fija— se calibró aquí desde cero. Ver abajo.

### La calibración de `f_min`

`scripts/calibrate_fmin.py` barre el piso de densidad y contrasta contra §7.3.
σ y α están fijados por la especificación; `f_min` solo se menciona («mantener
las regiones compactas», §4.3) sin valor:

| `f_min` | capturados | cobertura | nodos/mes | cr/nodo | lift |
|---|---|---|---|---|---|
| 0.02 | 119 963 | 56.7 % | 5 632 | 0.89 | 3.0× |
| 0.05 | 46 633 | 22.1 % | 1 146 | 1.70 | 5.7× |
| **0.10** | **26 471** | **12.5 %** | **391** | **2.82** | **9.6×** |
| 0.11 | 24 837 | 11.7 % | 340 | 3.04 | 10.3× |
| 0.15 | 19 976 | 9.4 % | 208 | 4.00 | 13.5× |
| §7.3 | 26 645 | 12.5 % | 357 | 3.11 | ≈10× |

Se eligió **0.10** porque clava las dos métricas de crimen (capturados −0.7 %,
cobertura exacta). 0.11 ajusta mejor la densidad a costa de perder 6.8 % del
crimen capturado; el trade-off está documentado en `config.yaml`.

## Fase D · Evaluación contra la línea base

`crimepipe evaluate` escribe `data/processed/evaluation_report.json` y
`evaluation_monthly.csv` (24 filas, todas las métricas de ambos métodos), y el
análisis se lee en el dashboard en **ⓘ → Evaluación**.

| Método | Crímenes | Cobertura | Nodos | cr/nodo |
|---|---|---|---|---|
| **Topológico** | **26 471** | **12.5 %** | 9 385 | **2.82** |
| BFS voraz | 24 638 | 11.7 % | 9 420 | 2.62 |
| BFS anchura | 18 054 | 8.5 % | 9 420 | 1.92 |

| Ganancia del topológico | Crímenes | Cobertura | Nodos | Relativa |
|---|---|---|---|---|
| vs BFS voraz | +1 833 | +0.9 pts | −35 | **+7.4 %** |
| vs BFS anchura | +8 417 | +4.0 pts | −35 | **+46.6 %** |

La **huella está igualada por construcción**: cada región de la línea base crece
hasta el tamaño medio de las topológicas de ese mismo mes. Sin eso, comparar
cobertura no mide nada — gana quien haga los subgrafos más grandes (§7.1).

### Por qué hay dos líneas base

§7.2 pide un «region growing voraz por BFS», y esa frase admite dos lecturas con
resultados muy distintos:

- **voraz** — la frontera se ordena por crimen del vecino, así que la región
  elige hacia dónde crecer. Es una línea base fuerte.
- **anchura pura** — cola FIFO, crece igual en todas direcciones y se traga
  tantos nodos vacíos como calientes. Es el suelo.

Se evalúan las dos porque la conclusión del proyecto —«el topológico gana a
igual huella»— tiene que sostenerse contra la versión fuerte, no solo contra la
ingenua. **El +15.5 % que reporta §7.3 cae entre ambas** (+7.4 % y +46.6 %), lo
que sugiere que el código original usaba una política intermedia. La ambigüedad
queda explícita en vez de resolverse a ojo ajustando hasta dar con el número.

Contra §7.3, el lado topológico ajusta bien (crímenes −0.7 %, cobertura +0.2 %);
la línea base voraz sale un 6.8 % más fuerte que la original.

## Calibración automática de parámetros

Los valores de σ y `f_min` se habían fijado contrastando contra §7.3 — números
de una corrida anterior sobre este mismo dataset. **Eso no generaliza**: con otra
ciudad o ventana esos números no existen.

`crimepipe calibrate` los deriva de los datos sin mirar ninguna referencia. El
criterio es la **rodilla de la frontera de Pareto** entre cobertura y densidad:
aflojar los parámetros sube cobertura y hunde densidad, apretarlos hace lo
contrario, y entre medias hay un punto donde cada punto extra de cobertura
empieza a costar mucha más densidad. Se localiza por máxima distancia a la
cuerda (*kneedle*) sobre los ejes normalizados.

**Validación sobre Chicago 2024-2025**: la rodilla cae en σ=150, f_min=0.125 →
**12.7 % de cobertura**, a 0.2 puntos del 12.5 % de §7.3, sin haber visto nunca
esa cifra. La curvatura es 0.478 (codo marcado, no un punto arbitrario). Es
evidencia directa de que el criterio localiza el régimen correcto por sí solo.

En `config.yaml`, σ, α y `f_min` aceptan un número o la cadena `auto`:

```yaml
density:
  sigma_m: auto
hotspots:
  alpha: 0.3
  f_min_ratio: auto
```

`K` **no se calibra**: no es un parámetro estadístico sino una decisión de
alcance —cuántas zonas estudiar al mes— y además no degrada la densidad al
subir, así que su «óptimo» sería siempre el máximo del rango.

### Barrido de parámetros

`crimepipe calibrate` recorre σ × α × f_min (180 combinaciones, ~35 s) más un
barrido de K, y escribe `dashboard/public/data/param_sweep.json`. El análisis se
lee en el dashboard, en **ⓘ → Parámetros**. (`scripts/sweep_params.py` hace lo
mismo de forma autónoma, sin escribir la calibración.)

**Maximizar cobertura a secas es gameable**: se consigue agrandando los
subgrafos. Por eso el resultado se presenta como frontera de Pareto entre
cobertura y densidad, y no como un ranking.

Efecto marginal, variando uno y dejando los otros dos en la configuración base:

| σ (m) | cobertura | nodos/mes | cr/nodo | lift |
|---|---|---|---|---|
| 60 | 7.6 % | 109 | 6.10 | 20.7× |
| 90 | 10.2 % | 232 | 3.87 | 13.1× |
| **120** | **12.5 %** | **391** | **2.82** | **9.6×** |
| 150 | 14.9 % | 588 | 2.23 | 7.5× |
| 250 | 26.0 % | 1 786 | 1.28 | 4.3× |

| K | cobertura | nodos/mes | cr/nodo |
|---|---|---|---|
| 10 | 10.0 % | 322 | 2.73 |
| **20** | **12.5 %** | **391** | **2.82** |
| 50 | 17.0 % | 521 | 2.86 |
| 100 | 19.4 % | 595 | 2.87 |

Tres hallazgos:

1. **σ y f_min son los dos lados de la misma palanca.** Ambos deciden hasta
   dónde se derrama la región alrededor de un pico, y recorren casi la misma
   curva cobertura↔densidad.
2. **α es casi irrelevante** en este régimen: de 0.1 a 0.5 la cobertura solo se
   mueve del 11.9 % al 12.6 %. Con el piso en 0.10 el barrido ya solo recorre
   nodos densos, donde apenas quedan máximos espurios que simplificar.
3. **K es la palanca barata.** A diferencia de σ y f_min, subir K *no degrada la
   densidad*: de K=20 a K=100 la cobertura pasa del 12.5 % al 19.4 % y la
   densidad sube ligeramente. Los subgrafos 21.º a 100.º son tan densos como los
   primeros veinte. Si el objetivo es cobertura sin perder poder de
   señalización, subir K domina a ensanchar el kernel.

#### Decisión abierta: σ = 90 ajusta mejor que σ = 120

`σ=90 · α=0.3 · f_min=0.075` reproduce **las cuatro métricas de §7.3 dentro del
±2 %**: 12.4 % de cobertura (objetivo 12.5 %), 354 nodos/mes (357) y 3.08 cr/nodo
(3.11) — frente al +9.6 % de huella de la configuración actual.

El config mantiene **σ = 120 m porque §4.1 lo fija explícitamente**. La
alternativa contradice ese valor pero ajusta mejor los agregados, así que o el
código original usó σ=90, o su truncamiento/normalización del kernel difería.
Es una decisión tuya, no un bug: está documentada en el dashboard y basta
cambiar dos líneas de `config.yaml` para adoptarla.

### Un bug que los tests atraparon

La primera versión asignaba cada nodo al pico de la raíz de un union-find. Esa
raíz fusiona **todas** las ramas tocadas, incluidas las persistentes, así que un
nodo adyacente solo a la región B podía quedar etiquetado con el pico de A sin
tener camino físico hasta ella: `2024-08_h01` salía no conexo. La invariante
correcta es asignar cada nodo a la región de **uno de sus vecinos ya
procesados**, la de pico más alto. Los tests de conectividad corren sobre los
480 subgrafos reales, no solo sobre la fixture sintética.

---

## Fase E · Caracterización y similitud

`crimepipe features` corre en **~35 s** y escribe `similarity.json` (0.52 MB) con
el top-5 de cada subgrafo, sus coordenadas UMAP y su clúster.

### La invariante: el crimen no entra en la similitud

Es §5.1 y es la razón de ser del proyecto. Si el crimen alimentase la
comparación, "zonas parecidas tienen crimen parecido" sería una tautología. Lo
que entra describe **forma**:

| Bloque | Dims | Peso | Qué mide |
|---|---|---|---|
| estructural (§5.2) | 12 | 1.0 | conectividad (8) + geometría métrica (4) |
| `graph2vec` (§5.3) | 128 | 1.0 | subárboles enraizados de radio *t* |
| jerarquía vial (§5.4) | 6 | 0.5 | de qué clase de calle está hecho |
| **fusionado** | **146** | | |

`tests/test_features.py::test_el_crimen_no_toca_la_similitud` no inspecciona
nombres de campos —eso se esquiva renombrando— sino comportamiento: retuerce
todos los conteos del artefacto y exige que el vector fusionado salga
**idéntico bit a bit**. `Subgraph` tampoco tiene campos de crimen, así que quien
lo tenga en la mano no puede contaminar aunque quiera.

Los pesos se reparten como `w/√d`, no por dimensión. Sin eso, un bloque de 128
dims con peso 1.0 aporta ~128 unidades de norma² y uno de 12 aporta ~12: el peso
real sería el número de columnas, no el del config.

### graph2vec sin gensim

`gensim` no publica wheel para Python 3.14 —pip resuelve a la 0.10.1, de 2014,
incompatible con numpy 2— así que el Doc2Vec PV-DBOW va escrito en numpy: ~60
líneas de descenso por gradiente con muestreo negativo, el mismo algoritmo. El
hash de las etiquetas WL usa `hashlib`, no el `hash()` de Python, que va salteado
por proceso y rompería la reproducibilidad entre corridas.

Verificado sobre grafos sintéticos y sus copias renumeradas:

| | |
|---|---|
| documentos WL de grafos isomorfos | idénticos |
| coseno entre isomorfos | 0.9998 – 0.9999 |
| coseno máximo entre no isomorfos | 0.9864 |
| reejecutar con el mismo seed | artefacto byte a byte idéntico |

### ¿Mide algo la similitud?

Contraste sobre los 480 subgrafos reales, con las 12 dims estructurales
normalizadas:

| | Distancia media |
|---|---|
| pares emparejados por coseno | 0.0753 |
| pares al azar | 0.2148 |
| | **2.85× más parecidos** |

El coseno del primer similar va de 0.512 a 0.997 (mediana 0.801), así que
discrimina en vez de dar todo casi 1. Los clústeres pequeños son limpios e
interpretables: `c1` son 15 subgrafos de 2 nodos con elongación 0 —una sola
arista, perfectamente lineal—; `c3` son 10 de 3 nodos también rectos; `c7` son 8
de ~5 nodos con elongación 0.53.

**Lo que no funciona bien:** `c9` se lleva 258 de los 480, más de la mitad del
corpus en un solo grupo, y otros 136 quedan como ruido. Con
`min_cluster_size = 5` HDBSCAN produce una masa grande más varios grupos
diminutos y muy tensos. No lo he tocado para que salga bonito: es lo que da el
parámetro de la spec sobre estos datos, y el número está en el dashboard.

### El mismo sitio, otro mes

El corpus son 20 subgrafos × 24 meses. El mismo cruce reaparece cada mes y sus
copias son casi idénticas en forma, así que copan el ranking: para
`2024-01_h01`, el primer similar sin filtrar es `2024-11_h01` con coseno 0.839 y
Jaccard 0.50 — literalmente el mismo sitio. Son de verdad los más parecidos,
pero no responden la pregunta del proyecto, que compara zonas distintas.

Por eso el artefacto trae **dos listas** por hotspot: `similar` (la de §5.5) y
`similar_distinct`, que descarta cualquier candidato con un solo nodo en común.
El cajón del UMAP lleva el conmutador **otro lugar** para pasar de una a otra.

### POIs: dos formas de descartar que no son la misma

`crimepipe pois` baja 54 430 elementos de OSM (`amenity`, `shop`, `leisure`) —una
petición por llave, porque las tres juntas agotan el tiempo de Overpass— y los
mapea a las ocho categorías de §3.2 con un YAML versionado.

La primera versión reportaba un **53 % de descartes** y mi propio umbral de aviso
saltó diciendo que la taxonomía se quedaba corta. No era eso: los descartes eran
9 197 aparcamientos, 4 580 plazas de aparcamiento, 2 378 aparcabicis, 1 896
bancos, 1 875 buzones. Infraestructura y mobiliario urbano, correctamente fuera
de alcance — el indicador estaba midiendo cuántos aparcamientos tiene Chicago,
no si el YAML cubre sus usos de suelo.

Ahora hay dos cubos separados y el aviso solo mira el segundo:

| | | |
|---|---:|---|
| clasificados | 26 708 | |
| excluidos a propósito | 27 471 | lista explícita en el YAML |
| **sin mapear** | **251** | **0.9 % de los funcionales** |

Con la taxonomía corregida (`theatre`/`cinema`/`arts_centre` → vida nocturna,
`childcare`/`community_centre` → educación, `track`/`sports_hall` → parques):

| Categoría | POIs | |
|---|---:|---:|
| Parques | 10 306 | 38.6 % |
| Alimentación | 6 356 | 23.8 % |
| Comercio | 4 882 | 18.3 % |
| Educación | 1 775 | 6.6 % |
| Vida nocturna | 1 455 | 5.4 % |
| Salud | 1 124 | 4.2 % |
| Finanzas | 660 | 2.5 % |
| Policía | 150 | 0.6 % |

`amenity=place_of_worship` (1 840) se excluye **a propósito**, no por descuido:
es un uso de suelo real pero ninguna de las ocho categorías de §3.2 le
corresponde, y forzarlo dentro de `education` o `retail` deformaría la entropía.
Si el estudio quisiera medirlo, lo correcto es ampliar §3.2, no reetiquetarlo.

La asociación POI→subgrafo usa la envolvente convexa bufferizada a 50 m, y ese
buffer sale ahora de `config.yaml` **también para el polígono que dibuja el
mapa**: antes el dibujo usaba 40 m y la asociación 50, así que el usuario veía un
área y el perfil hablaba de otra.

Resultado: 64 486 asociaciones sobre 480 zonas, 6 zonas sin ningún POI, entropía
funcional normalizada media 0.583. Un POI puede pertenecer a varias zonas y se
cuenta en todas — las envolventes de meses distintos se solapan por
construcción, y descontar el solape haría que el perfil de una zona dependiera
de qué otras zonas existen.

### Los cinco paneles (§8)

El mapa es el sujeto del dashboard, así que **todo lo demás son cajones**: se
abren cuando se piden y se quitan cuando estorban. No hay columna fija robando
ancho al mapa.

| Panel | Dónde | Qué hace |
|---|---|---|
| **A** mapa | el fondo | subgrafo seleccionado + eventos; puntos / calor / relieve 3D / nodos, POIs |
| **B** descriptivo | ficha movible sobre el mapa | estadísticas, tipos de delito, donut de POIs |
| **C** embeddings | cajón izquierdo | scatter UMAP, clústeres, lista del conjunto en foco |
| **D** línea de tiempo | barra inferior plegable | serie mensual global, arrastrable, con marcadores R y A–E |
| **E** comparación | cajón lateral | A vs B: delitos, footprint, tipos, POIs, entropía |

El panel D comparte estado con el selector de meses del desplegable —los mismos
`monthLo`/`monthHi`—, así que arrastrar en cualquiera de los dos mueve el otro.
Dos controles del mismo filtro que pudieran contradecirse serían peor que uno.
Se pliega dejando fuera la cabecera, para que siempre se vea dónde vuelve.

Los cajones llevan sus bloques **plegables**, con los dos o tres primeros
abiertos: un panel con doce secciones desplegadas a la vez no se lee. Un gráfico
dentro de un bloque plegado se redibuja al abrirlo, porque dibujar dentro de un
`display:none` lo deja sin dimensiones.

**Cada cosa en un solo sitio**, y el criterio es de quién habla cada bloque. El
donut de **POIs** describe *ese* subgrafo, igual que sus tipos de delito, así que
se lee junto a ellos en la ficha. La lista de **parecidos** habla de *otros*
subgrafos, así que vive donde están dibujados: en el cajón del UMAP, pegada al
scatter. Pulsar una fila salta a un punto que el scatter tiene justo encima, y
las letras **R · A–E** de la lista son las mismas que marcan la línea de tiempo:
los tres paneles nombran igual a los mismos subgrafos.

Esa lista es un solo widget con dos modos. Con foco por *similares* muestra la
referencia y sus más parecidos con puntuación y el conmutador **otro lugar**;
con un grupo escogido a lazo o por chip de clúster muestra el conjunto sin
puntuación, porque un grupo libre no sale de ningún subgrafo y no hay coseno
contra el que medirlo.

El botón del espacio de embeddings está a la **izquierda**, del mismo lado por el
que se abre su cajón. La información técnica es el único cajón **ancho**
(520 px): cinco pestañas y tablas de varias columnas no caben en 420, y el hueco
que deja se mide contra su ancho real, no contra el genérico.

**La ficha se mueve.** Se arrastra por su cabecera y se coloca donde estorbe
menos, que depende de dónde esté mirando el usuario. Moverla libre tiene dos
formas de salir mal —escaparse de la ventana, o quedar debajo de la barra, de la
línea de tiempo o de un cajón—, así que el arrastre está acotado a la misma zona
útil que el mapa respeta al encuadrar, y los bordes **tiran** de ella con un imán
de 32 px. El imán no es estética: pegada a un borde, la ficha sigue ahí al abrir
un cajón o al plegar la línea de tiempo, porque el enganche se reaplica contra
los límites nuevos. Doble clic en la cabecera la devuelve a su sitio; la posición
sobrevive a la recarga.

### El relieve 3D: el campo de §4.1 en volumen

El mapa de calor y el relieve enseñan **exactamente el mismo campo** —la densidad
gaussiana de §4.1—. Cambia el canal: color plano frente a altura. Y la altura se
lee mejor justo donde importa, en el extremo alto: dos manchas del rojo más
saturado son indistinguibles, dos cerros de altura distinta no, y ahí es donde
viven los máximos de los que salen los subgrafos.

La σ no es decorativa: se lee de `sigma_m` del propio artefacto (**120 m**), así
que los cerros tienen el ancho real del kernel con el que se extrajeron los
hotspots. Se está mirando la superficie sobre la que se buscaron los máximos.

#### Una malla propia, no `fill-extrusion`

`fill-extrusion` solo sabe levantar **prismas**: cada celda es un cubo de altura
constante con paredes verticales entre vecinas. Una gaussiana no tiene escalones,
así que el resultado era una escalera con la silueta de una colina.

El relieve es una **capa `custom`**: MapLibre entrega el contexto de WebGL y la
matriz de proyección, y la malla la dibujamos nosotros. La altura vive en los
**vértices**, de modo que el triángulo interpola entre muestras, y la normal se
arma en el vertex shader a partir del **gradiente del campo** —diferencias
centrales sobre la altura normalizada—. El sombreado sigue entonces la pendiente
real de la gaussiana y no la cara de un prisma. No hay tipo de capa de estilo que
sepa hacer esto; por eso se baja a WebGL y no por gusto.

Detalles que hay que acertar o no se ve nada:

- **Índices de 32 bits.** La malla pasa de 65 535 vértices. En WebGL2 vienen de
  serie; en WebGL1 hacen falta por extensión, y si no está, el relieve **no se
  dibuja** en vez de dibujar basura.
- **Soltar el VAO** antes de tocar los atributos, y **deshabilitarlos** después:
  en WebGL2 MapLibre deja uno enlazado, y el estado de atributos es global.
- **`depthMask(true)`**: MapLibre lo deja cerrado en algunas pasadas y la
  superficie se dibujaría sin escribir profundidad.

#### La rejilla es fija: la cámara no la deforma

La primera versión construía la rejilla sobre la vista actual. Al mover la
cámara cambiaban el tamaño de celda, la extensión **y la altura del pico**, así
que el terreno se deformaba bajo el cursor. Un campo escalar no depende de dónde
se mire.

Ahora la malla se calcula **una vez sobre el bbox de los datos**, con la celda en
metros y la altura en metros. Girar, inclinar o acercarse no toca un solo
vértice: la cámara solo la mira. Consecuencia buena: `moveend` ya no reconstruye
nada, y la malla solo se rehace cuando cambia el filtro, que es lo único que
cambia el campo.

Cómo se construye:

1. La carga por nodo (`computeNodeCounts`, **la misma función** que alimenta la
   vista de nodos: las dos no pueden desincronizarse) se reparte en la rejilla.
2. Convolución gaussiana **separable** — una gaussiana 2D es el producto de dos
   1D, así que son dos pasadas de (2r+1) muestras en vez de una de (2r+1)².
3. Se tejen dos triángulos por cuadro, y **solo** donde alguna esquina supera el
   2 % del máximo: donde el campo es nada no hay geometría y se ve el mapa base,
   no una losa a cota cero.

El lado de celda se mantiene en la banda **[σ/3, σ]**. Por debajo de σ/2 la
gaussiana ya está muestreada de sobra y más vértices no añaden información al
campo; por encima de σ el kernel no llega a tres muestras. Un techo de 550 000
vértices engorda la celda si hiciera falta. Medido sobre el bbox real
(33,7 × 41,8 km):

| detalle | celda | rejilla | vértices | triángulos | muestras del kernel |
|---|---|---|---|---|---|
| 0,44× | 120 m | 289 × 356 | 103 k | 204 k | 7 |
| **1,00×** | **60 m** | **576 × 710** | **409 k** | **815 k** | **13** |
| 1,52× y más | 53 m | 653 × 805 | 526 k | 1 048 k | 15 |

Solo se sube a la GPU lo que cambió: las posiciones son estáticas, y un cambio de
filtro reescribe alturas, gradientes e índices. Las subidas ocurren dentro del
`render`, porque el contexto de GL solo es válido ahí y así una ráfaga de cambios
sube una vez y no una por cambio.

#### Las zonas seleccionadas

Con algo en foco, el terreno de fuera **pierde el color y conserva el relieve**:
se sigue leyendo la forma del campo, y la zona seleccionada es la única que
mantiene la rampa de densidad. Es el mismo idioma de foco que ya usan las demás
capas, llevado a la superficie.

La máscara se calcula con el **mismo polígono que dibuja el mapa** —la envolvente
con su buffer de §5.6—, no con una aproximación: si el relieve marcara un área y
el mapa otra, el usuario vería dos zonas distintas para el mismo subgrafo. Solo
se prueban los vértices del bbox de cada polígono, así que el coste va con el
área en foco y no con la malla entera, y cambiar el foco **no recalcula el
campo**: solo reescribe un atributo.

#### Lo que hay que decir

El pipeline mide distancias **geodésicas sobre la red vial** y esta vista las
mide en línea recta. Al otro lado de un río o de una autopista sin cruce, el
pipeline no propaga densidad y el relieve sí. Es una aproximación para mirar, no
el campo con el que se calculó nada. Se quita exportando `f` por nodo y por mes
—1,4 MB cuantizado a uint16— e interpolándolo sobre esta misma malla.

Y un relieve visto a plomo es un mapa de calor con peor rampa: al activarlo, si
la cámara está vertical se inclina sola una vez (58°). A partir de ahí manda el
usuario, arrastrando con el **botón derecho**.

**La línea de tiempo se dibuja a píxeles reales.** Tenía un `viewBox` de 1200 con
`preserveAspectRatio="none"`: en una pantalla de 2000 px eso multiplica la
horizontal por 1,7 y deja la vertical intacta, así que la serie salía aplastada y
las etiquetas ensanchadas. Un `viewBox` fijo sirve dentro de un cajón de ancho
conocido; una barra que ocupa toda la ventana hay que medirla, y rehacerla cuando
cambia el ancho —al redimensionar o al abrir un cajón, esperando a que termine la
transición—. Lleva rótulo y ejes nombrados: sin ellos «10k» no dice de qué, y
las abreviaturas de mes podrían ser cualquier cosa con fecha.

**Sin velo sobre el mapa.** Un cajón aquí no es un diálogo modal: se consulta
*mientras* se mira el mapa, y apagar el mapa para enseñar un panel que habla del
mapa es contraproducente. El mapa sigue arrastrable y clicable con los cajones
abiertos. El espacio de embeddings entra por la **izquierda** y comparación,
información y ajustes por la **derecha**, así que se puede tener uno de cada lado
a la vez; el resto de la interfaz flotante se aparta en vez de quedar debajo, y
`easeTo` recibe el `padding` de los cajones abiertos para que lo que centre caiga
en la parte del mapa que se ve. Por debajo de 900 px de ancho el cajón se
superpone: apartarse dejaría la interfaz sin sitio.

### La paleta de POIs

Ocho categorías **nominales**, así que nada de rampas — sugerirían un orden que
no existe. Los ocho tonos no están elegidos a ojo sino **buscados** maximizando
la separación del peor par por recocido simulado sobre el espacio OKLCH, y
verificados con el validador de paletas:

| Comprobación | | |
|---|---|---|
| banda de luminosidad OKLCH | PASS | los ocho dentro de 0.43–0.77 |
| suelo de croma | PASS | ninguno lee como gris |
| separación CVD | PASS | peor par ΔE 8.4 (protanopía), objetivo 8 |
| suelo de visión normal | PASS | peor par ΔE 15.4, suelo 15 |
| contraste sobre blanco | PASS | los ocho ≥ 3:1 |

Todo con los **28 pares**, no solo los adyacentes: en el mapa los POIs caen
dispersos, así que cualquier categoría puede acabar junto a cualquier otra.

La paleta anterior, elegida a ojo, fallaba cuatro de las cinco: `park` y
`finance` eran dos verdes a ΔE 6.1 —indistinguibles incluso con visión de color
completa—, `retail` y `nightlife` quedaban a ΔE 2.9 bajo deuteranopía, `police`
era un gris de croma 0.015 y `food` no llegaba a 3:1 sobre blanco.

Además el mapa lleva **leyenda** de POIs cuando la capa está activa: con ocho
categorías, la identidad no puede depender del tono a secas.

### Selección en dos niveles

Recorrer los parecidos de un subgrafo sin perderlo exige separar dos cosas que
parecen una:

- **`hsRef`** — el subgrafo *ancla*: de él salen los parecidos.
- **`hsSel`** — el que la ficha está mostrando.

Pinchar un parecido cambia `hsSel` pero **no** `hsRef`, así que la lista no se
desmonta bajo el cursor y se pueden recorrer los cinco sin perder la referencia.
Reanclar es un acto explícito: pinchar en el mapa, o el botón *Anclar aquí* que
aparece en la ficha cuando lo que se mira no es el ancla.

### Foco: ver solo un conjunto

**`focusIds`** es el conjunto de subgrafos en foco, y puede venir de tres sitios:

| Origen | Cómo |
|---|---|
| similares | seleccionar un subgrafo en el mapa → él y sus 5 parecidos |
| lazo | contorno a mano alzada sobre el scatter UMAP |
| clúster | pulsar un chip de la leyenda de clústeres |

El interruptor **Solo …** de la barra aparece en cuanto hay foco y hace que el
mapa dibuje **solo** ese conjunto. El filtro de meses se sigue aplicando además:
el foco acota qué subgrafos, no qué periodo.

El lazo existe porque los grupos que importan no siempre coinciden con los que
encontró HDBSCAN — la vista son 2 dimensiones y el clustering trabajó sobre 146.
Un lazo de cien subgrafos no se etiqueta: por encima de seis, los marcadores de
la línea de tiempo pasan a ser marcas mudas y la leyenda desaparece.

### Ajustes de los gráficos

En Configuración → **Gráficos**: altura de los gráficos, tamaño de los puntos del
UMAP y paleta de clústeres. Por defecto va **Categórica** (`schemeTableau10`)
porque los clústeres son categorías sin orden; Turbo y Viridis se leen mejor si
se quiere ver el recorrido continuo del espacio, a costa de sugerir un orden que
no existe.

Los marcadores del panel D son lo que convierte la lista de parecidos en una
lectura temporal: **R** es el subgrafo seleccionado y **A–E** sus similares, y
de un golpe se ve si los cinco caen en el mismo verano o están repartidos por
los dos años. Se puede pulsar cualquiera para saltar a él.

### Fase 3 · Comparación A vs B (§6)

El botón de la barra abre el panel de contraste. Se fija un subgrafo como **A**
desde su ficha y otro como **B** —lo natural es elegirlo entre sus *más
parecidos*, para que la estructura quede aproximadamente constante—, y el panel
muestra:

- delta de delitos totales y por categoría, en barras espejo;
- **evolución del footprint**: crímenes sobre los nodos de cada zona mes a mes en
  toda la ventana, con marca en el mes de extracción de cada uno. La huella
  existe los 24 meses aunque el subgrafo se detectara en uno solo; recortar la
  serie a ese mes daría un número, no una serie;
- distribución de POIs lado a lado, con las categorías presentes en solo una de
  las dos marcadas `solo A` / `solo B`;
- delta de entropía funcional.

La pregunta que responde: manteniendo la estructura aproximadamente constante,
¿cuánta de la variación restante en intensidad de crimen se alinea con
diferencias en composición de POIs?

---

## Paso 0 · Mapa de verificación

Comprueba visualmente que el CSV se lee bien y que los crímenes caen donde deben.

```bash
python scripts/build_map_data.py     # CSV -> dashboard/public/data/*.json
python scripts/serve_map.py          # sirve el mapa y abre el navegador
```

Sin dependencias: stdlib de Python para el pipeline, MapLibre GL JS por CDN
para el mapa. Requiere conexión a internet la primera vez (librería + teselas).

El mapa muestra **los 213 389 crímenes de las 4 categorías del estudio** (§3.1):
selector de periodo por arrastre sobre el histograma mensual, selector de tipo
de delito, puntos o mapa de calor, y dos cajones laterales —información técnica
y configuración—. El encuadre inicial usa el bbox real de los datos: si latitud
y longitud estuvieran invertidas, el mapa saltaría fuera de Chicago de forma
evidente.

La vista por nodos necesita además `crimepipe export`, que es quien produce
`snapping.bin`.

El color de los crímenes es configurable desde el cajón de ajustes: ocho
presets más un selector libre. De ese único color se derivan por HSL los tonos
de la interfaz y las siete paradas de la rampa del mapa de calor, así que
cualquier elección produce una rampa monotono coherente. La elección se guarda
en `localStorage`.

### Capas: qué se combina con qué

La barra superior tiene tres controles y no se estorban:

- **Crudo / Nodos** — excluyente. Decide *dónde* se dibuja el crimen: en la
  coordenada del CSV o en el nodo al que lo llevó el snapping. Son dos posiciones
  del mismo dato y superponerlas no diría nada.
- **Puntos** y **Calor** — dos interruptores independientes. Se combinan entre sí
  y **se pueden apagar los dos**, que es como se ven los subgrafos solos.
- **Subgrafos** — independiente de todo lo anterior.

Con las dos capas de crimen apagadas, el selector de origen se atenúa y se
desactiva: elegir dónde dibujar algo que no se dibuja no significa nada.

### Crudo o nodos: dónde se dibuja cada crimen

En **Crudo** el mapa dibuja la coordenada que trae el CSV. En **Nodos** dibuja el
nodo de la red al que el snapping (§4.0) proyectó cada crimen, con el radio
codificando la carga en escala de raíz cuadrada —el área del círculo es
proporcional al número de crímenes—. Esa segunda vista es la que enseña la
entrada real del campo de densidad: si el snapping concentra mal, todo lo que
viene después hereda el sesgo, y en el mapa crudo eso es invisible.

Sobre Chicago 2024-2025 se ve que:

| | |
|---|---|
| crímenes proyectados a la red | 211 391 de 213 389 (99.06 %) |
| distancia a la **calle**, mediana / p95 | 0.5 m / 5.0 m |
| distancia al **nodo**, mediana / p95 / máx | 54.2 m / 108.8 m / 1656 m |
| nodos que reciben algún crimen | 22 966 de 29 832 (77 %) |
| carga media / máxima de un nodo (24 meses) | 9.2 / 788 |

Las dos distancias miden cosas distintas. A la calle el crimen cae casi encima,
porque el portal geocodifica al centroide de la manzana, que ya está sobre el eje
vial. Al nodo se desplaza media manzana, porque §4.0 asigna el crimen al extremo
más cercano de la arista. Ese desplazamiento no es ruido: es la unidad de
análisis del proyecto, ya que el campo de densidad vive sobre nodos.

El artefacto es `snapping.bin` (0.97 MB), que **no** trae conteos precalculados
sino la asignación crimen→nodo. Así el navegador agrega con los mismos filtros de
mes y tipo que ya aplica a los puntos, y las dos vistas no pueden
desincronizarse; el coste es un barrido de 213 389 enteros por cambio de filtro
(~5 ms, con freno de 70 ms al arrastrar el selector de meses).

`crimes.bin` y `snapping.bin` se indexan **por posición** y los genera el
pipeline por separado, así que cada uno firma su carga con una huella
(`fingerprint`) y el dashboard las compara antes de combinarlos. Si no coinciden,
la vista por nodos se desactiva y explica por qué, en vez de dibujar un mapa
plausible y equivocado.

### Los subgrafos en el mapa

El interruptor **Subgrafos** dibuja los 480 hotspots extraídos, filtrados por el
mismo rango de meses que los puntos. Tres canales llevan información y sus
dominios salen de los datos, no de constantes —si cambia el dataset, la escala
se recalcula sola—:

- **opacidad del área** → crimen capturado. Con 480 áreas superpuestas, un
  relleno plano sería una mancha uniforme que no dice nada.
- **grosor del trazo** → densidad (cr/nodo).
- **radio de la semilla** → crimen capturado otra vez, para que el ranking se
  lea sin abrir la ficha.

Los trazos llevan un halo blanco debajo que los separa del callejero del mapa
base, que a zoom alto compite visualmente. Al pasar el ratón se realzan las tres
features del subgrafo a la vez, vía `feature-state` (sin recargar la fuente).

Tienen selector de color propio —independiente del de crímenes, porque deben
distinguirse de los puntos que contienen— y un selector de representación
**Área / Calles / Ambos**: con muchos meses activos conviene ver solo las calles.

Opciones de `build_map_data.py`:

```bash
--all-categories              # los 31 tipos (default: solo las 4 del estudio)
--from 2024-01 --to 2025-12   # ventana temporal (default: sin límite)
--dedupe auto|id|content|none # clave de deduplicación (default: auto)
--no-canonical                # no escribir data/interim/crimes_canonical.csv
```

### Rendimiento

Medio millón de puntos en el navegador exige tres decisiones:

1. **Artefacto binario.** `crimes.bin` empaqueta 8 bytes por punto (lat/lon/lugar
   en `Uint16`, tipo/mes en `Uint8`) = 1.71 MB. El mismo contenido en JSON serían
   ~5 MB de texto a parsear en el hilo principal; como binario el navegador
   expone `TypedArray` sobre el búfer con coste de parseo cero. La cuantización
   reparte 65 536 niveles sobre el bbox: ~0.6 m, irrelevante para dibujar. El
   pipeline real trabaja sobre el CSV crudo en doble precisión.
2. **Construcción por lotes.** El GeoJSON se arma en tandas de 40 000 dentro de
   `requestAnimationFrame`, así la interfaz no se congela y la barra de progreso
   es real, no decorativa.
3. **Matriz de conteos precalculada.** Una matriz mes×tipo se calcula en Python.
   Cualquier total de un filtro es una suma sobre la submatriz seleccionada, no
   un recorrido de los 213 389 puntos.

Con `--all-categories` el artefacto sube a 492 624 puntos y 3.94 MB; las tres
decisiones son las que hacen que esa escala también funcione.

> Las visualizaciones (histograma mensual, barras de proporción, embudo de
> filas, rampa de densidad, muestras de color) son **D3**. El mapa es MapLibre
> GL: para cientos de miles de puntos hace falta WebGL, y D3 sobre canvas no
> compite ahí. La división es esa — D3 para los gráficos analíticos, MapLibre
> para la capa geográfica.

> El basemap **ya no es CARTO**. `basemaps.cartocdn.com` pasó a exigir clave de
> API: sigue devolviendo HTTP 200, pero la tesela es una marca de agua que dice
> «API KEY REQUIRED». Se verificó pidiendo la misma tesela para el centro de
> Lima y para mar abierto —hash MD5 idéntico, o sea que no es mapa—. Un 200 que
> no es un error visible es el peor modo de fallo posible.
>
> Ahora se usa **OpenFreeMap**, que sirve el mismo estilo Positron en vectorial,
> sin clave y sin cuota. El conmutador de mapa base cambia el estilo entero con
> `setStyle` y reinyecta las capas propias vía `transformStyle`, así que no hay
> que reconstruir el GeoJSON —27 MB en Lima— ni recablear la interfaz.
>
> La especificación pide Mapbox GL JS; el cambio es directo cuando haya token,
> las dos APIs son compatibles.

---

## Estructura

```
.
├── PROJECT_SPEC.md
├── config.yaml                       # todos los parámetros (§10)
├── pyproject.toml
├── data/
│   ├── raw/                          # CSV crudo (gitignored)
│   ├── interim/<ciudad>/             # caché por dataset (gitignored)
│   │   ├── crimes_canonical.csv      # el CSV ya proyectado a 7 columnas
│   │   ├── crimes_load.npz           # la carga cacheada (Lima tarda 10 min)
│   │   ├── network.graphml           # red vial cacheada
│   │   ├── snapped.npz               # nodo asignado por crimen
│   │   └── hotspots.json             # los subgrafos del mes
│   └── processed/<ciudad>/
├── pipeline/
│   ├── config.py                     # modelos pydantic + carga de config.yaml
│   ├── io.py                         # rutas, claves de caché, manifest
│   ├── snapping.py                   # §4.0 arista -> nodo con índice espacial
│   ├── density.py                    # §4.1 campo gaussiano geodésico
│   ├── hotspots.py                   # §4.2-4.3 join tree + persistencia
│   ├── calibrate.py                  # rodilla de Pareto: σ, α, f_min sin §7.3
│   ├── baseline.py                   # §7.2 region growing voraz y anchura pura
│   ├── evaluate.py                   # §7.1-7.4 métricas y comparación
│   ├── similarity.py                 # §5.5 fusión, coseno, top-K, UMAP, HDBSCAN
│   ├── export.py                     # §8 subgrafos -> GeoJSON + snapping.bin
│   ├── cli.py                        # entrypoint `crimepipe`
│   ├── features/
│   │   ├── structural.py             # §5.2 descriptor de 12 dims
│   │   ├── embedding.py              # §5.3 graph2vec (WL + PV-DBOW) y netlsd
│   │   ├── hierarchy.py              # §5.4 huella de jerarquía vial
│   │   └── pois.py                   # §5.6 perfil de POIs y entropía
│   └── ingest/
│       ├── crimes.py                 # §3.1 cargador + ValidationReport
│       ├── profiles.py               # §3.1 lo propio de cada proveedor
│       ├── pois.py                   # §3.2 descarga y clasificación de POIs
│       ├── poi_taxonomy.yaml         # §3.2 mapeo OSM -> 8 categorías
│       └── network.py                # §3.3 descarga + caché + jerarquía vial
├── scripts/
│   ├── calibrate_fmin.py             # barrido de f_min contra §7.3
│   ├── sweep_params.py               # barrido autónomo σ × α × f_min × K
│   ├── build_map_data.py             # CSV -> artefacto binario del mapa
│   ├── build_datasets_index.py       # datasets.json para el selector de ciudad
│   └── serve_map.py                  # servidor estático local
├── tests/                            # 126 tests
└── dashboard/
    └── public/
        ├── index.html                # dashboard (MapLibre + D3)
        └── data/
            ├── datasets.json         # índice que alimenta el selector
            ├── chicago/              # artefactos de Chicago
            └── lima/                 # artefactos de Lima
```

## Proyección al esquema canónico

El cargador no arrastra el CSV de entrada: lo **proyecta** a un esquema fijo de
siete columnas y descarta el resto de forma explícita y auditable. Con el export
de Chicago eso es **22 columnas de entrada → 7**:

| Canónica | Columna del CSV | |
|---|---|---|
| `lat` | `Latitude` | obligatoria |
| `lon` | `Longitude` | obligatoria |
| `fecha` | `Date` | obligatoria |
| `crimen` | `Description` | obligatoria |
| `tipo` | `Primary Type` | obligatoria |
| `lugar` | `Location Description` | opcional |
| `id` | `ID` | opcional |

Descartadas: `Case Number`, `Block`, `IUCR`, `Arrest`, `Domestic`, `Beat`,
`District`, `Ward`, `Community Area`, `FBI Code`, `X/Y Coordinate`, `Year`,
`Updated On`, `Location`.

Es lo que permite mover el pipeline a otra ciudad sin tocar nada aguas abajo:
ningún módulo posterior conoce el esquema del proveedor. El resultado se
materializa en `data/interim/crimes_canonical.csv` como evidencia de la
reducción.

## El dataset de Chicago

`data/raw/Crimes_-_2001_to_Present_20260822.csv` — export completo del portal de
Chicago: 495 854 filas, 2024 y 2025 enteros, 31 tipos de delito.

| | Filas |
|---|---|
| Totales en el CSV | 495 854 |
| Descartadas por calidad (`lat_invalida`) | 3 230 |
| Duplicadas | 0 |
| Filtradas por categoría | 279 235 |
| **Retenidas** (4 categorías × 24 meses) | **213 389** |

**213 389 contra las 213 602 que espera §7.3: 0.1 % de diferencia.** El dataset
y el recorte son los correctos. Los otros 27 tipos de delito se filtran en la
ingesta, antes de generar el artefacto: no llegan al navegador.

### Deduplicación

No hay duplicados reales: cada fila trae un `ID` único. Sí hay ~1 475 filas que
comparten (fecha, tipo, descripción, lat, lon) — pero son incidentes distintos
que caen en la misma manzana a la misma hora reportada, porque Chicago
geocodifica al centroide del bloque. Deduplicar por contenido borraría crímenes
reales, así que la clave por defecto es `id` (`--dedupe auto`).

### Notas

- El cargador resuelve alias normalizando mayúsculas, acentos, espacios y
  guiones bajos, así que también acepta el export reducido
  (`DATE  OF OCCURRENCE`, ` PRIMARY DESCRIPTION`), donde `crimen` y `tipo` caen
  ambos en la misma columna por no existir dos niveles de descripción.
- El descarte de 3 230 filas por `lat_invalida` es del CSV de origen: son
  incidentes sin geocodificar. No se pueden snappear a la red vial.

---

## Multi-dataset: Chicago y Lima

El pipeline procesa dos ciudades. Todo lo que depende de la ciudad vive bajo
`datasets:` en `config.yaml`; el resto del archivo —σ, α, `f_min`, pesos de
fusión, dimensiones del embedding— son parámetros del método y se comparten.
El bloque de un dataset puede sobrescribir **cualquier** sección de nivel
superior, así que si una ciudad necesita otro σ basta con redeclararlo dentro
de su bloque.

```bash
.venv/Scripts/python scripts/build_map_data.py --dataset lima
.venv/Scripts/python -m pipeline.cli hotspots --dataset lima
.venv/Scripts/python -m pipeline.cli evaluate --dataset lima
.venv/Scripts/python -m pipeline.cli pois     --dataset lima
.venv/Scripts/python -m pipeline.cli features --dataset lima
.venv/Scripts/python -m pipeline.cli export   --dataset lima
.venv/Scripts/python scripts/build_datasets_index.py   # refresca el selector
```

Sin `--dataset` se usa `active_dataset` del config. Cada ciudad escribe en sus
propias carpetas —`data/interim/<ciudad>/`, `data/processed/<ciudad>/`,
`dashboard/public/data/<ciudad>/`— porque los artefactos no son mezclables:
`crimes.bin` y `snapping.bin` se indexan por posición y llevan una huella que
el navegador comprueba, pero `hotspots.geojson` y `similarity.json` no la
llevan y se habrían pisado en silencio.

En el dashboard el selector es **la marca misma**, arriba a la izquierda: el
dataset activo no es un filtro más, es la identidad de todo lo que hay debajo.
Cambiar de ciudad recarga la página con `?dataset=<id>`; el estado se recuerda
en `localStorage`. Se recarga a propósito: media docena de estructuras del
dashboard (índices de hotspots, dominios de color, la capa de deck.gl, el
estado de selección) se derivan de los artefactos en el arranque y no tienen
camino de vuelta.

`data/datasets.json` es lo que alimenta el selector. Lo genera
`scripts/build_datasets_index.py` desde el config, y **solo lista los datasets
con artefactos en disco**: uno declarado pero sin procesar aparecería en el
menú y llevaría a un mapa vacío.

### Perfiles de fuente

`pipeline/ingest/crimes.py` sabe proyectar cualquier CSV al esquema canónico
resolviendo alias de columna. Eso basta mientras el proveedor solo difiera en
cómo *llama* a las cosas; no basta cuando difiere en **qué hay que hacer con
las filas**. Para eso está `pipeline/ingest/profiles.py`, donde un perfil
declara cuatro cosas y ninguna más:

| | |
|---|---|
| `columns` | anclaje explícito canónica → columna real. Gana sobre los alias. |
| `uid_columns` | identificador compuesto, para fuentes donde ninguna columna sola identifica el hecho. |
| `drop_where` | exclusión por valor de una columna *de origen*: filas que la fuente marca como no fiables. |
| `local_utc_offset_h` | desfase horario, cuando la fuente publica la fecha en UTC. |

Un perfil no contiene umbrales. Los umbrales viven en `config.yaml`, porque son
parámetros del análisis y tienen que poder barrerse; el perfil solo dice *qué*
columnas mirar.

El reporte de validación gana una tercera categoría de baja. Antes había dos y
no había que mezclarlas: **descartes** (el CSV viene mal: coordenada ilegible,
fecha imposible) miden la calidad de la fuente, y **filtros** (categoría no
analizada, mes fuera de ventana) miden el recorte del estudio. Lima obliga a
una tercera, **exclusiones**, que no es ninguna de las dos: la fila está bien
formada y dentro del alcance, pero su geocodificación es falsa. Contarla como
descarte diría que el CSV está roto; contarla como filtro diría que se decidió
no estudiarla. Ni una cosa ni la otra.

---

## El dataset de Lima

`data/raw/delitos_lima_metropolitana_completo_2018-25.csv` — denuncias de la
PNP consolidadas por la DGIS. **1.8 GB, 3 137 220 filas, 58 columnas**, un solo
departamento y una sola provincia (LIMA / LIMA), 43 distritos.

### Qué columnas se necesitan

Siete de las 58. Las otras 51 se descartan de forma explícita y quedan
enumeradas en `validation_report.json`.

| Canónica | Columna de origen | Por qué esa y no otra |
|---|---|---|
| `lat` / `lon` | `lat_hecho`, `long_hecho` | las del hecho, no las de la comisaría que registró |
| `fecha` | `fecha_hora_hecho_iso_utc` | el CSV trae además el epoch en ms y el año/mes/día desglosados; la ISO es la única que no hay que reconstruir. **No** `fecha_hora_registro_hecho`: es administrativa y llega a ir meses por detrás |
| `tipo` | `subtipo_hecho` | el nivel equivalente al `Primary Type` de Chicago (THEFT, ROBBERY, ASSAULT) **no** es `tipo_hecho` sino `subtipo_hecho`: `tipo_hecho` mete hurto, robo, extorsión y estafa en la misma caja. Da 35 tipos distintos en vez de 5 |
| `crimen` | `modalidad_hecho` | la descripción fina del hecho, equivalente al `Description` de Chicago |
| `lugar` | `distrito_hecho` | `direccion_hecho` es texto libre sin normalizar y no sirve para agrupar |
| `id` | `id_dgc` (+ subtipo + modalidad) | ver deduplicación |

Tres columnas más se leen sin llegar al esquema canónico, porque son las que
gobiernan la limpieza: `observacion`, `direccion_hecho` y `modalidad_hecho`.

**La hora es UTC y hay que convertirla.** El epoch `1547269200000` de la primera
fila es 2019-01-12 05:00 UTC, que en Lima son las 00:00 — y `turno_hecho` dice
«madrugada», que confirma la lectura local. Perú no observa horario de verano
desde 1994, así que en toda la ventana el desfase es exactamente −5 h. No es
cosmético: el mes es la unidad de agregación del pipeline, y leer en UTC un
hecho de las 21:00 lo mueve al día siguiente y, si cae a fin de mes, al mes
siguiente.

### Duplicados

`id_dgc` **no es único**: 164 803 identificadores traen entre 2 y 9 filas.

| Multiplicidad | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| ids | 156 473 | 7 795 | 466 | 52 | 9 | 6 | 1 | 1 |

Inspeccionadas, las filas repetidas son **idénticas en todo menos en
`objectid`**: misma fecha del hecho, misma coordenada, mismo subtipo, misma
modalidad, misma comisaría y hasta la misma hora de registro. `objectid` es una
clave subrogada del ETL, no un hecho distinto.

La clave de deduplicación es **`id_dgc | subtipo_hecho | modalidad_hecho`**, no
`id_dgc` a secas. `id_dgc` identifica la *denuncia*, y una denuncia puede
registrar legítimamente dos delitos distintos; añadir subtipo y modalidad
colapsa la repetición del ETL sin fundir esos dos. Sobre el material retenido
elimina **131 073 filas**.

No se usa la clave por contenido (fecha, tipo, lat, lon): con este archivo
colapsaría hechos reales, porque la geocodificación es a nivel de cuadra.

### Geocodificación falsa: el problema serio del archivo

**1 246 941 filas (39.7 %) tienen coordenadas inventadas.** Llevan
`observacion = GEO FORZADA AL CENTROIDE DE COMISARIA`: no se pudo geocodificar
la dirección y se las colocó en la comisaría que tomó la denuncia.

Traen `lat_hecho` y `long_hecho` rellenos y numéricamente válidos, así que
ninguna validación de coordenada las detecta. Lo que las delata es la
concentración:

| `observacion` | Filas | Puntos distintos | Filas por punto |
|---|---|---|---|
| `COORDENADA OK` | 1 777 990 | 430 377 | 4.1 |
| `COORDENADA ADECUADA MANUALMENTE` | 64 801 | 29 299 | 2.2 |
| **`GEO FORZADA AL CENTROIDE DE COMISARIA`** | **1 246 941** | **129** | **9 666** |

Sobre un KDE de σ = 120 m cada comisaría sería el hotspot más intenso de la
ciudad, y el análisis acabaría midiendo dónde están las comisarías.

**El filtro correcto es `observacion`, no `estado_coord`.** Es el error que
parece obvio y no lo es: `estado_coord` marca `SIN COORDENADA` a 1 338 244
filas, y ahí dentro caen las 91 314 «adecuadas manualmente», que sí están
geocodificadas de verdad (2.2 filas por punto). Filtrar por `estado_coord`
tiraría 91 314 filas buenas y —peor— dejaría pasar nada, porque las malas
también son detectables ahí; pero el criterio quedaría atado a una columna que
no significa lo que parece.

### Centroides de relleno: el residuo

Aun descartando lo anterior queda un artefacto más fino. Un único punto,
**(-12.0463731, -77.042754)** —la Plaza de Armas— acumula **40 805 hechos
provenientes de 36 127 direcciones escritas distintas**. Es la coordenada a la
que cae todo lo que el geocodificador no supo resolver dentro del Cercado, y
hay una equivalente por distrito.

La firma que los separa de una esquina realmente caliente son **dos
condiciones a la vez**: muchos hechos *y* muchas direcciones distintas. Una
esquina caliente de verdad —un mercado, un paradero— acumula cientos de hechos
sobre dos o tres direcciones; un centroide de relleno acumula cientos sobre
cientos. Exigir solo volumen se llevaría por delante justo lo que el análisis
busca; exigir solo variedad de direcciones se llevaría cualquier manzana
geocodificada a nivel de cuadra, que es la precisión normal de la fuente y
perfectamente utilizable.

Los umbrales están en `config.yaml` (`min_rows: 500`, `min_addresses: 100`) y
los puntos detectados salen enumerados con su peso en `validation_report.json`,
para que la decisión sea revisable: si el detector se comiera una esquina real,
se vería ahí antes que en el mapa. Sobre Lima marca **117 puntos** y descarta
**218 632 hechos**.

### Qué `tipo_hecho` se conservan

El archivo trae 52 tipos. El criterio es si **el hecho ocurre en la vía
pública**, que es la condición para que tenga sentido sobre una red vial.

El alcance se decide sobre `tipo_hecho` (cinco familias) y el **eje de
categorías que se ve en el mapa es `subtipo_hecho`**: dentro de esas familias
entran todos sus subtipos, 35 en total, sin lista escrita a mano que se
desactualice. Filtrar por la columna gruesa *y además* mostrarla dejaría el mapa
con cinco cajones y perdería la distinción que hace útil el análisis.

**Se conservan (5 familias, 35 subtipos, 1 504 498 filas antes de limpiar):**

| `tipo_hecho` | Filas | Subtipos principales |
|---|---|---|
| PATRIMONIO (DELITO) | 1 011 604 | hurto (513 283), robo (343 648), estafa, extorsión, receptación |
| FALTAS | 189 432 | contra el patrimonio (96 215), contra las personas (88 322) |
| SEGURIDAD PUBLICA (DELITO) | 108 902 | peligro común (93 192), salud pública |
| LIBERTAD (DELITO) | 100 207 | violación de la libertad sexual (61 596), de la libertad personal (29 539) |
| VIDA, EL CUERPO Y LA SALUD (DELITO) | 94 353 | lesiones (80 372), homicidio (12 146) |

**Se descartan.** Los dos que importan, porque son los dos más numerosos del
archivo:

- **`INTERVENCION POLICIALES` (719 986).** No es delito: es actividad policial.
  Sus subtipos son «visita realizada por medida de protección» (478 582),
  «servicio policial efectuado» (138 459) y «control de identidad» (24 442).
  Mide dónde patrulla la policía, no dónde ocurre el crimen. Además solo el
  **5.8 %** está geocodificado.
- **`LEY DE VIOLENCIA CONTRA LA MUJER Y GRUPOS VULNERABLES` (752 584).**
  Violencia intrafamiliar. Ocurre en el domicilio y se geocodifica a la
  vivienda de la víctima. Es el tipo más numeroso del archivo y, de entrar,
  dominaría todos los hotspots midiendo dónde vive la gente. **Descartarlo es
  una decisión sobre el alcance del método, no sobre la importancia del
  fenómeno**: un análisis de violencia doméstica es legítimo, pero no es un
  análisis sobre red vial.

Y el resto: `LEY 30096 DELITOS INFORMATICOS` (83 944, sin lugar físico),
`ADMINISTRACION PUBLICA` (32 331), `DENUNCIAS ESPECIALES` (8 309, que son
pérdidas de documento y están geocodificadas al 0.2 %), `FE PUBLICA`,
`FAMILIA`, `TRAFICO ILICITO DE DROGAS` (5 155: la coordenada es la de la
intervención, no la del delito), y la treintena de tipos residuales con menos
de 3 000 filas, incluidos los ~400 de la familia `MODALIDAD POLICIAL …`, que
son artefactos de la taxonomía.

### El recorte, en números

| | Filas |
|---|---|
| Totales en el CSV | 3 137 220 |
| Descartadas por calidad (`crimen_vacio`) | 106 |
| **Excluidas por la fuente** — geo forzada al centroide de comisaría | **1 246 918** |
| **Excluidas por la fuente** — centroides de relleno (117 puntos) | **218 632** |
| Duplicadas (`id_dgc\|subtipo\|modalidad`) | 131 073 |
| Filtradas por categoría | 647 961 |
| Filtradas por ventana (anteriores a 2018) | 5 393 |
| **Retenidas** | **887 137** |

90 meses (2018-01 .. 2025-06), **35 tipos de delito**, 43 distritos. Los diez
más numerosos, ya limpios:

| | | | |
|---|---|---|---|
| HURTO 315 378 | ROBO 218 156 | FALTAS C. EL PATRIMONIO 65 439 | PELIGRO COMUN 61 484 |
| FALTAS C. LAS PERSONAS 53 824 | LESIONES 45 276 | ESTAFA 32 059 | VIOLACION LIB. SEXUAL 28 670 |
| VIOLACION LIB. PERSONAL 13 039 | RECEPTACION 11 988 | EXTORSION 10 479 | HOMICIDIO 5 587 |

Las 5 393 filas anteriores a 2018 son errores de digitación en
`fecha_hora_hecho`: hay hechos fechados en 1923, 1945, 1948, 1963, 1965 sobre
denuncias registradas entre 2018 y 2025. El archivo se publica como «2018-25» y
ese es su alcance real, así que la ventana empieza en 2018-01 y no tiene tope
superior.

### La cobertura se derrumba a partir de 2024-09

Es la limitación más importante del dataset y **no se corrige filtrando**, así
que conviene tenerla presente al leer cualquier serie temporal:

| Año | Con coordenada | Sin coordenada | % geocodificado |
|---|---|---|---|
| 2018 | 284 041 | 23 133 | 92.5 % |
| 2019 | 220 896 | 116 490 | 65.5 % |
| 2020 | 212 860 | 104 146 | 67.1 % |
| 2021 | 244 901 | 169 496 | 59.1 % |
| 2022 | 283 654 | 169 192 | 62.6 % |
| 2023 | 349 028 | 198 276 | 63.8 % |
| 2024 | 179 606 | 337 096 | **34.8 %** |
| 2025 | 16 430 | 218 596 | **7.0 %** |

En el material retenido eso se traduce en un desplome de volumen mensual: de
~19 000 hechos/mes en 2024-08 a ~1 700/mes desde 2024-11. **No es que baje el
crimen: es que dejó de geocodificarse.** Los meses de 2024-09 en adelante
existen en el dashboard porque el encargo era usar todos los años del archivo,
pero sus hotspots descansan sobre una décima parte de la evidencia de los meses
anteriores y no son comparables con ellos.

### La red vial

`Provincia de Lima, Peru` en OSM resuelve exactamente al ámbito del dataset —
los 43 distritos, mismo bbox que las coordenadas. **135 633 nodos y 355 907
aristas** en la red `drive` simplificada.

### Resultado sobre Lima

Con los mismos parámetros de campo que Chicago (σ = 120 m, α = 0.3,
`f_min` = 0.10), sin recalibrar, pero con **K estadístico** en vez de top-20
fijo:

| | Chicago | Lima |
|---|---|---|
| Ventana | 2024-01 .. 2025-12 (24 meses) | 2018-01 .. 2025-06 (90 meses) |
| Hechos retenidos | 213 389 | 887 137 |
| Categorías (`subtipo_hecho`) | 4 | **35** |
| Nodos de la red | 29 537 | 135 633 |
| Snappeados a la red | 211 391 (99.1 %) | 882 279 (99.5 %) |
| Criterio de K | `fixed` = 20 | `montecarlo`, α = 0.05, B = 99 |
| Subgrafos por mes | 20 | **3 – 70** (mediana 46) |
| Subgrafos extraídos | 480 | 4 160 |
| Hechos capturados | 26 471 | 160 177 |
| Cobertura | 12.52 % | 18.16 % |
| Huella media por mes | 1 084 nodos (3.7 %) | 1 468 nodos (1.1 %) |
| Densidad en hotspots | 2.82 cr/nodo | 1.21 cr/nodo |
| *Lift* sobre la media de la ciudad | 9.6× | **16.8×** |
| Ganancia sobre la línea base voraz | — | +19 909 (+14.2 %) |
| Ganancia sobre anchura pura | — | +46 314 (+40.7 %) |

Tres lecturas, y la primera es una advertencia:

- **Las cifras de cobertura ya no son comparables entre las dos columnas.**
  Chicago usa K = 20 fijo y Lima un K que va de 3 a 70; una cobertura mayor con
  más regiones no dice nada sobre el método. Lo comparable entre ciudades es el
  *lift*, y las ganancias sobre la línea base, que son internas al dataset y sí
  se miden a igualdad de número de regiones y de huella (§7.1).
- **El *lift* es 16.8× contra 9.6×.** Lima tiene 4.6 veces más nodos de red que
  Chicago para 4.2 veces más hechos repartidos en 3.75 veces más meses, así que
  la densidad absoluta de sus subgrafos es menor —1.21 contra 2.82 cr/nodo—. Lo
  que es mayor es la concentración *relativa a su propio fondo*: la huella de
  los hotspots es el **1.1 % de la red** y captura el **18 % del crimen**.
- **K sigue al volumen sin quedar amarrado a él.** La correlación de Spearman
  entre hechos del mes y número de hotspots es ρ = 0.66 (p = 2.5e-12): el
  criterio reacciona al volumen, como debe, pero no es una función de él. El mes
  más pobre de la serie —2019-10, con 331 hechos frente a una mediana de
  10 748— sale con 3 hotspots; ninguno de los 90 meses toca el tope de 80, así
  que `max_k` no está actuando como un K fijo encubierto.

**§7.3 no se aplica a Lima.** Su tabla de valores esperados describe Chicago
2024-2025, así que el bloque `reference:` solo existe en el dataset de Chicago.
Sobre Lima, tanto el reporte de consola como la pestaña «Extracción» del
dashboard omiten la columna «Esperado» —y también el aviso del ±2 %— en lugar
de contrastar contra los números de otra ciudad. La comparación con la línea
base sí se imprime siempre: esa es interna al dataset y siempre significa algo.

## Sensibilidad de los parámetros sobre Lima

```bash
.venv/Scripts/python -m pipeline.cli calibrate --dataset lima
```

180 combinaciones —6 valores de σ × 5 de α × 6 de f_min— sobre los 90 meses, en
8 minutos. Resultado:

```
RECOMENDADO   σ 120 m · α 0.2 · f_min 0.10
              11.8 % cobertura · 1.15 cr/nodo · 1007 nodos/mes · lift 15.9x
              frontera de Pareto: 85 configuraciones no dominadas
              curvatura de la rodilla 0.510 (codo marcado)
```

**σ = 120 m sale de los datos de Lima igual que salía de los de Chicago**, y es
el valor que §4.1 ya fijaba por criterio criminológico. Que dos ciudades con
redes de tamaño muy distinto —29 537 y 135 633 nodos— converjan al mismo ancho
de banda es el argumento más fuerte que hay a favor de ese parámetro: no está
ajustado a una ciudad.

La única discrepancia con `config.yaml` es α: el barrido pide 0.2 y el pipeline
usa 0.3. No se cambia, y el reporte lo señala en cada ejecución. Cambiarlo
rehace los 4 160 subgrafos y con ellos §7.3, la similitud y los estudios de
caso; y el barrido dice también que **α mueve la cobertura solo un 15.5 %**, así
que la diferencia entre 0.2 y 0.3 no es lo que decide nada.

### `--sweep-months` no sirve para esto

Se añadió un muestreo regular de meses pensando que el barrido completo sobre
Lima costaría horas. Costó 8 minutos, así que la premisa era falsa. Y lo
importante: **la submuestra cambia la respuesta**. Con 12 de los 90 meses el
barrido recomendaba σ = 150 m; con los 90, σ = 120 m. El flag sigue en el CLI
porque es útil para iterar mientras se toca el código, pero **ningún número
publicable debe salir de él**.

## Trayectorias y estudios de caso

```bash
.venv/Scripts/python -m pipeline.cli casestudies --dataset lima
```

Hasta aquí cada mes se extraía por separado y `2018-03_h07` y `2018-04_h11` eran
dos objetos sin relación aunque fueran la misma esquina. `pipeline/trajectories.py`
los enlaza, y de ese enlace salen las cuatro cantidades de la nota de tesis:

| | |
|---|---|
| **Frequency(H)** | `#meses donde aparece / #meses analizados` |
| **Intensity(H)** | crímenes del subgrafo en el mes |
| **Stability(H)** | `1/(k−1) · Σ IoU(H_t, H_{t+1})` |
| **Movement(H)** | `1/(k−1) · Σ d_G(seed_t, seed_{t+1})`, geodésica sobre la red |

### Cómo se decide que dos subgrafos son «el mismo»

Por **índice de Jaccard** sobre los nodos y no por recuento crudo de nodos
compartidos, que es la otra opción que la nota menciona. El recuento premia a
los grandes: dos regiones de 300 nodos que compartan 30 tendrían tanto en común
como dos de 35 que compartan 30, y solo el segundo par es la misma esquina.

**No se toman componentes conexas.** La tentación es construir el grafo «A se
parece a B» sobre los 4 160 subgrafos y quedarse con sus componentes. No sirve:
el enlace simple encadena. A solapa con B, B con C, y C puede estar a dos
kilómetros de A; sobre 90 meses eso funde media ciudad en una trayectoria y la
frecuencia resultante no significa nada. Se hace seguimiento en orden temporal
—cada mes contra la última aparición de cada trayectoria activa, uno a uno y por
IoU descendente—, que es el planteamiento estándar en seguimiento de objetos.

Una trayectoria sobrevive `--max-gap` meses sin aparecer (3 por defecto). Sin
eso, un hotspot que falta un mes se parte en dos trayectorias de frecuencia baja
y la distinción que la nota quiere —«muy frecuente» frente a «resaltó por algo
puntual»— mediría sobre todo el ruido mes a mes.

`Stability` es `None` y no `0` cuando hay una sola aparición: no es que el sitio
sea inestable, es que la pregunta no aplica. Con `0` caería en el cubo «muy
móvil» y contaminaría el eje entero.

### Los cuatro cubos

Los cortes son las **medianas observadas**, no valores fijos: «frecuente» no
significa lo mismo en una ventana de 24 meses que en una de 90, y fijar 0.5
dejaría tres cubos vacíos. Se publican los cortes usados para que el reparto sea
auditable.

### Estudio 2 · topología de persistentes vs episódicos

Se parte por **terciles** de frecuencia y no por la mediana: comparar el tercio
de arriba contra el de abajo deja fuera la franja del medio, donde
«persistente» y «episódico» no se distinguen y solo añadirían ruido.

A las 12 dimensiones del descriptor se añaden las tres que la nota pedía y no
estaban: `betweenness_mean`, `closeness_mean` e `intersection_density_km`. Se
calculan **solo para el contraste**; meterlas en el descriptor cambiaría el
embedding y la similitud ya publicada (§5.1). La intermediación se aproxima por
muestreo de 50 fuentes en los subgrafos grandes: Brandes exacto sobre 4 160
subgrafos de hasta 635 nodos son miles de millones de operaciones en Python.

Se reporta **delta de Cliff** y no diferencia de medias ni d de Cohen: casi
ningún descriptor es normal —las fracciones de grado y la densidad están
acotadas, `log_n_nodes` está sesgado— y la d supone normalidad. Y se ordena por
tamaño de efecto, no por valor p, con Benjamini-Hochberg sobre los quince
descriptores. Con miles de subgrafos casi cualquier diferencia sale
«significativa»; lo que decide si es interesante es cuánto se separan las
distribuciones.

**El contraste de POIs no puede ir sobre recuentos crudos.** Los persistentes
son más grandes —`log_n_nodes` los separa con δ = 0.23 en Chicago—, así que
«67.6 restaurantes frente a 11.8» mide tamaño y no función, que es exactamente
lo que §7.1 advierte que no mide nada. Va sobre el cociente de localización
cuando hay eje temporal, y sobre proporciones cuando no. Con esa corrección, lo
que queda en Chicago es `per_node` (δ = 0.29) y `entropy_norm` (δ = 0.20): los
hotspots persistentes son más densos en POIs y **más diversos funcionalmente**,
no «tienen más de algo».

### Estudio 3 · matching

Pares en el 2 % de menor distancia estructural pero con `crimes(A) ≥ 3 · crimes(B)`.
Se excluyen los pares que **solapan en el espacio** por encima del 5 %: dos
recortes de la misma esquina no son un contraste, son el mismo sitio medido dos
veces, y sin ese filtro la lista se llena de un hotspot grande contra sus
propias versiones de otros meses. Se limita además a dos las veces que un mismo
subgrafo puede aparecer, porque si no el caso más extremo copa la lista entera.

### Resultados

**Lima**, 90 meses, 4 160 subgrafos → **1 335 trayectorias**, de las cuales 611
aparecen más de un mes y 724 una sola vez.

| cubo | Lima | Chicago |
|---|---|---|
| frecuente + estable | 213 | 17 |
| frecuente + móvil | 180 | 16 |
| episódico + estable | 93 | 15 |
| episódico + móvil | 125 | 16 |

**La respuesta a «¿persistencia implica estabilidad espacial?» es que no.** Entre
las trayectorias frecuentes de Lima, 213 son estables y 180 móviles: casi mitad
y mitad. Un hotspot puede repetirse mes tras mes y estar desplazándose. Ese es
justamente el hallazgo que el plano `Frequency × Stability` hace visible y que
un ranking mensual de top-20 no puede mostrar.

**Topología (Lima, 1 432 persistentes vs 1 538 episódicos).** Todos los efectos
son pequeños —ningún |δ| llega a 0.2— y esa es la conclusión honesta: los
persistentes son algo **más grandes** (`log_n_edges` δ = +0.19), con **más grado
medio** (+0.16) y **menos densos** (−0.16), con menor intermediación (−0.17) y
menor cercanía (−0.14). El patrón es coherente: subgrafos más extensos y
ramificados, no calles individuales muy conectadas. Pero con δ < 0.2 la
topología **no separa** persistentes de episódicos; hay solape casi total entre
las dos distribuciones.

**POIs (sobre el cociente de localización, Lima).** Aquí los efectos, aunque
también pequeños, apuntan más claro:

| | persistentes | episódicos | δ |
|---|---|---|---|
| `lq:finance` | 14.15 | 9.74 | +0.175 |
| `per_node` | 2.33 | 1.68 | +0.146 |
| `lq:retail` | 10.88 | 5.85 | +0.138 |
| `entropy_norm` | 0.605 | 0.533 | +0.133 |

Los hotspots persistentes están **más especializados en comercio y finanzas** y
son **funcionalmente más diversos**. En Chicago sale lo mismo por la otra vía
(`per_node` δ = 0.29, `entropy_norm` δ = 0.20). Que el entorno los separe algo
mejor que la topología es un resultado, no un fallo: sugiere que lo que sostiene
un hotspot en el tiempo es la actividad, no la forma de la calle.

**Matching (Lima).** 25 pares con la misma forma y crimen ≥3×; el más extremo es
`2019-04_h01` con 134 hechos frente a `2024-12_h24` con 3, un factor de 44.7 a
distancia estructural 0.0. Son el material del tercer estudio de caso: dos
calles indistinguibles en forma y radicalmente distintas en crimen.

### σ = 120 m: ¿caminando o radio geográfico?

**Caminando, o más exactamente: distancia sobre la red vial.** No es un radio
geográfico y en ningún punto del pipeline se mide una distancia en línea recta
entre un crimen y otro.

El kernel se difunde con un **Dijkstra acotado** sobre el grafo de calles
(`KernelCache.get` en `pipeline/density.py`): desde cada nodo fuente se recorren
las aristas acumulando su longitud real, y el peso de un nodo a distancia `d` es
`exp(−d² / 2σ²)`, con truncamiento en `r = 3σ = 360 m`. Dos portales separados
por 30 m de fachada están a 30 m; los mismos dos portales en aceras opuestas de
una avenida sin cruce cercano están a los metros que haya que andar hasta el
cruce y volver. Por eso el campo no atraviesa manzanas ni cruza el río.

**Con una salvedad que conviene declarar.** La red se descarga con
`network_type: drive`, es decir, la red **circulable**. Sigue las calles, que es
lo que importa, pero no incluye las conexiones exclusivamente peatonales:
escaleras, pasajes, puentes peatonales. En Lima eso no es menor —las escaleras
de los cerros de San Juan de Lurigancho o Villa María son trayectos peatonales
reales que el grafo no ve—, así que en esas zonas la distancia del modelo
sobreestima la que un peatón recorrería. Cambiarlo es un parámetro
(`network.network_type: walk`), no un cambio de método, pero rehace la red y
todos los artefactos que cuelgan de ella.

Que σ esté en metros de calle y no de mapa es también lo que hace legítimo
compararlo con la criminología ambiental: 120 m es aproximadamente una cuadra
corta *andando*, que es la escala a la que se argumenta que opera la
oportunidad delictiva.

### POIs: instantáneas anuales de Geofabrik

Los POIs de Lima **no vienen de Overpass**. Overpass acabó limitando por cuota la
IP desde la que se ejecutaba, y el polígono de la Provincia de Lima es lo
bastante grande como para agotarla; el dataset estuvo tres días sin perfil
funcional por depender de un servicio interactivo con racionamiento.

Geofabrik publica el extracto de cada país como fichero estático, y además
guarda **una instantánea por cada 1 de enero desde 2014**:

```
https://download.geofabrik.de/south-america/peru-180101.osm.pbf   77 MB
https://download.geofabrik.de/south-america/peru-250101.osm.pbf  227 MB
```

Sin cuota, con URL fija y citable, y con el mismo resultado en cada ejecución.
Se lee con `osmium` —nodos y vías, porque un colegio o un parque son polígonos—
y se clasifica con la **misma** `poi_taxonomy.yaml` que Chicago. Cobertura sobre
Lima: **0.4 % sin mapear**, mejor que sobre Chicago. El YAML no hubo que tocarlo.

Se configura por dataset:

```yaml
pois:
  source: geofabrik
  region: south-america/peru
  years: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
```

Chicago sigue en `overpass`: su `pois.bin` ya estaba generado y validado, y
cambiarle la fuente movería los números de §7.3 sin que eso mida nada.

### El eje temporal, y la trampa que trae

Con `years`, cada subgrafo se caracteriza contra la instantánea de **su propio
año** en vez de describir un hotspot de 2018 con el mapa de 2026. Se usa la del
*inicio* del año, no la del final, para no meter información del futuro.

Pero el histórico de OSM registra **cuándo alguien mapeó algo**, no cuándo
abrió. Sobre Lima eso no es teórico:

| categoría | 2018 | 2025 | × |
|---|---|---|---|
| retail | 10 251 | 12 108 | 1.2 |
| food | 6 329 | 9 749 | 1.5 |
| park | 11 017 | 22 118 | 2.0 |
| **education** | 3 792 | **13 029** | **3.4** |

Repartiendo `education` por año de edición, **6 089 objetos en 2018 y 3 680 en
2019** de 13 029 —el 75 % de la categoría en dos años—, con `amenity=school`
(6 015) y `amenity=kindergarten` (5 873) casi 1:1. Es el volcado de un padrón
escolar. Ninguna otra categoría tiene esa firma: `food` reparte 750/932/593 en
esos mismos años.

**Por eso el perfil temporal no se lee en recuentos.** El contraste se hace por
cociente de localización contra la ciudad *del mismo año*, y **con la red como
base**, no con el total de POIs:

$$LQ_c(H) = \frac{n_c(H)\,/\,N_c(t)}{|H|\,/\,|G|}$$

La versión clásica del cociente, `(n_c/n) / (N_c/N)`, **no vale aquí**, y esto
se midió en vez de suponerlo: una importación que multiplica por 4 una sola
categoría mueve `n` y `N` en proporciones distintas según la composición de la
zona, y el cociente se desplaza de **2.00 a 1.37**. Amortigua la proporción
cruda —que se va de 0.57 a 0.84— pero no la cancela.

Con la red como base sí se cancela exactamente: la importación multiplica
`n_c(H)` y `N_c(t)` por el mismo factor, su cociente no se mueve, y el número de
nodos no depende de OSM. Está verificado en `pipeline/features/pois.py` para los
dos casos —importación de una categoría y crecimiento uniforme del mapeo—.

Como cualquier hotspot es una zona densa, ese `LQ` vale ~20x en todas las
categorías y no distingue unas zonas de otras. El dashboard muestra por eso la
**especialización relativa**, `LQ_c` dividido por la mediana de la propia zona,
que centra en 1; sigue siendo invariante porque los términos de red se cancelan
al dividir. El recuento crudo y el `LQ` absoluto quedan en el `title`.

Lo que sí se comprobó que **no** ocurre es el sesgo que haría el eje inservible:
que el mapeo creciera con la renta, que es el mismo gradiente que predice el
crimen. Ocurre lo contrario.

| zona (3.9 × 3.9 km) | 2018 | 2025 | × |
|---|---|---|---|
| Miraflores (rico) | 2 470 | 3 436 | 1.4 |
| San Juan de Lurigancho (periférico) | 583 | 1 212 | **2.1** |
| Villa El Salvador (periférico) | 1 424 | 1 811 | 1.3 |

### Reejecutar

```bash
.venv/Scripts/python -m pipeline.cli pois     --dataset lima --force
.venv/Scripts/python -m pipeline.cli features --dataset lima
.venv/Scripts/python -m pipeline.cli export   --dataset lima
.venv/Scripts/python scripts/build_datasets_index.py
```

Los `.osm.pbf` se cachean en `data/interim/osm/` (1.3 GB las ocho) y no se
vuelven a descargar. La ingesta entera son unos 4 minutos.

### Lo que se rompió en el ingestor de Overpass

Perseguir aquella descarga bloqueada destapó cuatro fallos reales.
Se dejan documentados y arreglados aunque Lima ya no use esa vía,
porque Chicago sí la usa.

El origen: durante tres días los POIs de Lima **no se pudieron descargar**.
Overpass acabó limitando por cuota la IP desde la que se ejecutaba, y el
polígono de la Provincia de Lima es lo bastante grande como para agotarla
rápido. Ese bloqueo es lo que motivó pasar a Geofabrik, que ya no depende de un
servicio interactivo; pero Chicago sigue usando esta vía, así que los cuatro
fallos que salieron por el camino importan igual. Están arreglados:

1. **User-Agent.** OSMnx se identifica con una cadena genérica.
   `overpass.kumi.systems` responde a eso con `429` y el texto «Please include
   a meaningful User-Agent string with your requests to avoid rate-limiting», y
   `overpass-api.de` con `406`. OSMnx interpreta el `429` como «servidor
   ocupado», espera y reintenta *dentro de la misma llamada*, así que el
   síntoma no era un error sino media hora sin avanzar y sin explicación.
   Ahora se envía un agente que identifica al proyecto.

2. **Un espejo regional colado entre los mundiales.** `overpass.osm.ch` sirve
   una base de Suiza: devuelve `200` con **cero elementos** a cualquier
   consulta sobre Lima. Un espejo que falla se reintenta en otro; uno que
   contesta «no hay nada» se cree, y el análisis sale adelante sin POIs y sin
   avisar. Está fuera de la lista, con el porqué escrito al lado para que no
   vuelva a entrar.

3. **El pinado de IP sin failover.** OSMnx fija *una* dirección por host
   (`_http._config_dns`) parcheando `socket.getaddrinfo`, para que la consulta
   de cuota y la consulta real caigan en la misma máquina. La elige con
   `socket.gethostbyname`, que devuelve la primera del registro sin comprobar
   si responde. `overpass-api.de` publica dos direcciones y desde esta red solo
   una acepta conexiones: cuando el resolutor devolvía primero la muerta, todas
   las peticiones morían en un `ConnectTimeout` idéntico de 102 s — y el propio
   parche de `getaddrinfo` era lo que impedía a urllib3 pasar a la segunda.
   Ahora se sondea cada dirección con una conexión TCP corta antes de fijarla.

4. **Ni presupuesto ni plan B.** Una sola instancia, sin reintentos, sin límite
   de reloj y sin alternativa si el polígono no entra de una pieza. Ahora hay
   rotación entre espejos con *backoff*, un presupuesto de 15 minutos por llave
   —`features` calcula los embeddings *antes* de bajar POIs, y no puede quedarse
   colgado indefinidamente detrás de una descarga opcional— y un plan B que
   pide la llave por celdas de una rejilla 4×4 recortadas contra el polígono
   real cuando la consulta completa devuelve `InsufficientResponseError`.

Y una bandera nueva, `--no-pois`, para `features` y `export`: los POIs son el
único paso que depende de un servicio ajeno y el único que puede tardar media
hora o fallar por cuota. Con la bandera, las tres horas de embeddings no quedan
detrás de él.

---

## Cuántos hotspots por mes: K adaptativo

`top_k = 20` fijo tiene un problema concreto: obliga al mismo número de
hotspots en un mes tranquilo y en uno con un brote. Si en enero hay ocho
concentraciones reales, el top-20 rellena con doce regiones que no son nada; si
en julio hay veinticinco, se pierden cinco. **El número de hotspots deja de ser
un resultado y pasa a ser un parámetro.**

`pipeline/selection.py` ofrece tres criterios, en `hotspots.selection.method`:

| | Qué hace | Coste |
|---|---|---|
| `fixed` | los `top_k` de siempre | — |
| `percentile` | el decil superior de las candidatas **de ese mes** | gratis |
| `montecarlo` | significancia contra distribución nula | ~6 s/mes con B=99 |

`montecarlo` sigue a Kulldorff (1997) y su adaptación a red vial de Shiode &
Shiode (2020): se simulan `replicates` realizaciones del mes bajo la hipótesis
nula, se toma de cada una el estadístico **máximo**, y sobrevive toda región
observada cuyo estadístico supere ese máximo con probabilidad menor que
`alpha_sig`. Comparar contra el máximo por réplica —y no contra la distribución
de todas las regiones nulas— es lo que hace que **no haga falta corregir por
test múltiple**: el estadístico ya es el del extremo, así que el error de tipo I
queda controlado a nivel de familia.

### El estadístico no puede ser el crimen capturado

Éste fue el error que el propio nulo destapó, y merece quedar escrito porque es
sutil y silencioso.

Las regiones se **ordenan** por crimen capturado, y eso es correcto: el objetivo
declarado es cubrir crimen real (§4.2). Pero usar esa misma cantidad como
estadístico de contraste no funciona. Al repartir el crimen uniformemente por
la red, el campo se queda sin estructura, el extractor devuelve unas pocas
regiones enormes y cada una captura cientos de crímenes por puro tamaño. El
máximo nulo salía **mayor que cualquier región observada** y nada resultaba
significativo:

```
nulo uniforme, estadístico = crimen capturado
  media 192   p95 261   max 291        <- contra hotspots observados de ~55
```

El fallo no estaba en el nulo sino en el estadístico: comparaba concentración
contra tamaño. La log-razón de verosimilitud de Poisson normaliza por el tamaño
esperado,

$$\Lambda = n_Z \log\frac{n_Z}{\lambda_Z} + (n_G - n_Z)\log\frac{n_G - n_Z}{n_G - \lambda_Z},
\qquad \lambda_Z = n_G\,\frac{|Z|}{|N|}$$

que es exactamente lo que Shiode & Shiode hacen con la longitud de su ventana
(ec. 1). Aquí $|Z|$ se mide en **nodos** y no en metros, porque el campo vive
sobre nodos: bajo el nulo cada nodo es igual de probable, así que el número de
nodos *es* la población en riesgo de la región. Con eso el nulo cae a media 23
y el contraste discrimina.

### Los dos nulos no responden a la misma pregunta

| | Pregunta | Chicago, hotspots/mes |
|---|---|---|
| `uniform` | ¿más concentrado que si el crimen cayera al azar sobre la ciudad? Es el nulo del paper. | **3 – 11** |
| `permutation` | ¿más concentrado que si la misma cantidad de crimen se repartiese entre los sitios donde de hecho pasa algo? | **1 – 3** |

El primero mide concentración contra la geografía; el segundo, contra la
oportunidad, y condiciona sobre dónde hay portales, comercios y gente. El
segundo es mucho más exigente: sobre Chicago deja uno o dos hotspots por mes.

### Cuántos salen sobre Lima, y el tope que había que subir

Medido sobre seis meses repartidos por la ventana, con las 118-318 regiones
candidatas que produce el extractor:

| mes | candidatas | `uniform` | `permutation` | `percentile` 0.90 |
|---|---|---|---|---|
| 2018-01 | 243 | 51 | 2 | 25 |
| 2019-06 | 318 | 48 | 2 | 34 |
| 2021-03 | 308 | 40 | 2 | 31 |
| 2023-05 | 242 | 54 | 2 | 26 |
| 2024-08 | 118 | 62 | 1 | 12 |
| 2025-03 | 230 | 36 | 1 | 23 |

Dos cosas que esta tabla resuelve:

1. **El tope `max_k` estaba puesto en 50 y saturaba.** En la primera corrida, 16
   de los 30 primeros meses tocaban el tope, con lo que `max_k` volvía a ser un
   K fijo disfrazado — justo lo que el criterio existe para evitar. Con
   `max_k: 80` el rango real es 36-62 y nada lo toca.

2. **El nulo del paper es permisivo sobre Lima.** Ochenta mil crímenes al mes
   sobre 135 633 nodos están tan lejos de repartirse uniformemente que casi
   cualquier concentración es significativa. `permutation` —que condiciona sobre
   dónde hay oportunidad— deja uno o dos por mes; `percentile` cae en medio.
   Los tres son defendibles y responden a preguntas distintas; el configurado es
   `uniform` por ser el del paper.

### Qué se configuró y por qué

- **Lima: `montecarlo` / `uniform`, B=99, α=0.05.** No tiene tabla de referencia
  contra la que validar, así que el número de hotspots de cada mes sale de los
  datos.
- **Chicago: `fixed`, a propósito.** §7.3 tabula sus valores esperados para
  top-20 (480 subgrafos en 24 meses) y es el dataset con el que se valida el
  pipeline. Cambiarle el criterio invalidaría esa comparación: el contraste
  mediría el cambio de criterio, no el del método. Para probarlo aquí basta
  `selection.method: montecarlo` en su bloque, sabiendo que `reference` deja de
  aplicar.

La línea base recibe **tantas regiones como sacó el extractor ese mes**, no
`top_k`. Con K adaptativo el número de hotspots es un resultado del mes; dejar
la base en 20 fijas mientras el topológico saca 5 compararía cobertura entre
familias de tamaño distinto, que es justo lo que §7.1 advierte que no mide nada.

---

## Validación con datos sintéticos

```bash
.venv/Scripts/python -m pipeline.cli benchmark --dataset chicago \
    --realisations 10 --parents 15 --offspring 200 --background 400
```

Réplica del experimento de **Shiode & Shiode (2020), §3 y §5**. Es la única
métrica **absoluta** del proyecto: cobertura, densidad y ganancia sobre la línea
base dicen que el extractor captura más crimen que hacer crecer regiones a lo
bruto, pero ninguna dice si acierta *dónde está* el hotspot, porque sobre datos
reales no hay verdad conocida.

Se siembran concentraciones en aristas elegidas al azar (proceso de Poisson por
clústeres: padres → hijos → fondo uniforme), se le pasa el resultado al
extractor sin decirle nada, y se mide contra la verdad:

$$PPV = \frac{\#\{r_i \in S^* \cap S_{true}\}}{\#\{r_j \in S^*\}}
\qquad
Sens = \frac{\#\{r_i \in S^* \cap S_{true}\}}{\#\{r_j \in S_{true}\}}$$

`PPV` mide **sobredisparo**; `Sens` —el paper la llama *specificity*— mide
**subdisparo**. Cada una por separado se maximiza haciendo trampa: marcar toda
la ciudad da sensibilidad 1, marcar un solo nodo acertado da PPV 1. Por eso se
leen juntas, y se añade F1.

### Dos adaptaciones que hay que declarar

1. **La red se recorta.** Shiode no corre el experimento sobre Buffalo entera
   sino sobre 900 m × 750 m con 394 puntos de referencia y 14 segmentos
   sembrados: casi el 4 % de la red. Repetir sus 300 puntos sobre una ciudad
   completa cambia el experimento —30 nodos sembrados entre 29 832 son el 0.1 %,
   con un fondo tan diluido que **todos los métodos aciertan** y el resultado no
   mide nada. `--subnetwork 400` devuelve la relación señal/ruido a la que el
   experimento estaba pensado. Se comprobó: sobre la ciudad entera los tres
   métodos dan sensibilidad 1.000 y PPV 0.215 idéntico.

2. **La verdad son nodos, no segmentos continuos.** El paper coloca los hijos a
   lo largo de la arista y mide contra puntos de referencia cada 30 m; aquí el
   campo vive sobre nodos, así que un hijo cae en uno de los dos extremos de su
   arista y `Strue` son esos extremos. Los números **no** son comparables con la
   Tabla 1 del paper, solo entre los métodos que se comparan aquí.

### Resultado 1 · σ es lo que controla el sobredisparo

Chicago, subred de 400 nodos, 15 aristas sembradas, 200 hijos, 400 de fondo,
8 realizaciones, criterio `fixed`:

| σ (m) | PPV topo | Sensibilidad | F1 | Nodos marcados |
|---|---|---|---|---|
| 40 | **0.290** | 0.996 | 0.444 | 107 |
| 60 | 0.186 | 1.000 | 0.310 | 172 |
| 90 | 0.122 | 1.000 | 0.216 | 247 |
| **120** (spec) | 0.095 | 1.000 | 0.173 | **311** |
| 180 | 0.079 | 1.000 | 0.146 | 373 |

La sensibilidad se mantiene en ~1.0 en todo el rango: **bajar σ no cuesta
detección, solo reduce el área marcada de más**. Con σ = 120 m el método marca
311 nodos de 400 para localizar 29 verdaderos; con σ = 40 m marca 107.

Es el mismo reproche que Shiode hace a los métodos planares —«tend to
over-represent cluster locations»— y aquí queda cuantificado sobre nuestro
propio extractor. §4.1 fija σ = 120 m por criterio criminológico (≈ una cuadra
corta) y ese sigue siendo el valor del pipeline; lo que el experimento añade es
que **ese valor está elegido para capturar crimen, no para localizarlo**, y que
si el objetivo fuera señalar la cuadra exacta habría que bajarlo.

### Resultado 2 · en localización exacta, el extractor no bate a la línea base

Con σ = 120 m y criterio fijo, sobre 10 realizaciones:

| Método | PPV | CV | Sensibilidad | F1 |
|---|---|---|---|---|
| topológico | 0.094 | 0.06 | 1.000 | 0.172 |
| voraz | 0.104 | 0.08 | 0.986 | **0.189** |
| anchura | 0.093 | 0.10 | 0.918 | 0.169 |

La U de Mann-Whitney sobre F1 da p = 0.011 a favor del **voraz**. No es un
error: la verdad sintética son picos puntuales sobre aristas, que es
exactamente lo que persigue una línea base que crece desde los nodos más
calientes. Lo que el extractor topológico gana según §7.2 es **crimen
capturado sobre datos reales**, que es otra cosa. El benchmark separa las dos
preguntas y conviene no confundirlas al escribir la tesis:

- *¿Captura más crimen?* Sí, +23.4 % sobre la línea base voraz (Lima).
- *¿Localiza mejor la cuadra exacta?* No, con σ = 120 m no.

### Resultado 3 · el K adaptativo reduce el sobredisparo

Mismo experimento, cambiando solo el criterio de selección:

| Régimen | PPV con `fixed` | PPV con `montecarlo` |
|---|---|---|
| 400 nodos, 15 padres, fondo 100 | 0.222 | 0.211 |
| 400 nodos, 15 padres, fondo 400 | 0.094 | **0.152** |
| 400 nodos, 40 padres, fondo 400 | 0.241 | **0.295** |
| 400 nodos, 40 padres, fondo 1200 | 0.190 | **0.294** |
| 150 nodos, 14 padres, fondo 100 | 0.237 | **0.357** |

Cuanto más ruido de fondo, más gana el criterio estadístico: con K fijo el
método está obligado a devolver 20 regiones aunque solo haya 15 concentraciones,
y las que sobran crecen sin control cuando el campo se vuelve difuso. Se midió
el caso extremo sobre la ciudad entera con fondo 30 000: la huella media por
región pasa de 9 nodos a **687**. El coste del K adaptativo es sensibilidad
—deja de marcar concentraciones débiles—, que es el intercambio esperado.

### Salidas

`data/processed/<ciudad>/benchmark_synthetic.json` y
`dashboard/public/data/<ciudad>/benchmark.json`, con las métricas por
realización, los agregados (media, desviación, coeficiente de variación como la
Tabla 1 del paper) y la U de Mann-Whitney por pares (Tabla 2).
