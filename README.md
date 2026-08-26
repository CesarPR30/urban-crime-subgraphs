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

Cómo se construye, en la vista actual y con el filtro actual:

1. La carga por nodo (`computeNodeCounts`, la misma que alimenta la vista de
   nodos: las dos no pueden desincronizarse) se reparte en una rejilla.
2. Convolución gaussiana **separable** — una gaussiana 2D es el producto de dos
   1D, así que son dos pasadas de (2r+1) muestras en vez de una de (2r+1)². Con
   20 000 celdas y 25 muestras son ~0,5 M multiplicaciones, medio milisegundo.
3. Una celda `fill-extrusion` por casilla por encima del 4 % del máximo; por
   debajo no se dibuja, o el mapa sería una losa plana con bultos.

El lado de celda se mantiene en la banda **[σ/4, σ]**: por debajo de σ/4 la
gaussiana ya no varía dentro de la celda —más celdas no añaden información al
campo, solo polígonos—; por encima de σ el kernel no llega ni a tres muestras y
los cerros salen a cuadros. Un techo de 30 000 polígonos relaja el límite
superior cuando la vista es enorme. Resultado medido: 8 000–21 000 celdas de
9–17 px de lado a cualquier zoom, con el kernel entre 3 y 25 muestras.

Dos decisiones que no son obvias:

- **El ancho se mide a la escala del centro, no con el bbox.** Con la cámara
  inclinada el bbox se dispara hacia el horizonte, y usarlo hundiría la
  resolución justo cuando se está usando la vista en 3D. Por lo mismo el bbox se
  acota a 3× el ancho de la vista.
- **Un relieve visto a plomo es un mapa de calor con peor rampa.** Al activarlo,
  si la cámara está vertical se inclina sola una vez (55°); a partir de ahí manda
  el usuario. Se gira e inclina arrastrando con el **botón derecho**.

Y una diferencia que hay que decir: el pipeline mide distancias **geodésicas
sobre la red vial** y esta vista las mide en línea recta. Al otro lado de un río
o de una autopista sin cruce, el pipeline no propaga densidad y el relieve sí. Es
una aproximación para mirar, no el campo con el que se calculó nada. A escala de
ciudad, además, una campana de 120 m es más pequeña que una celda: lo que se ve
entonces es el agregado, y las campanas individuales aparecen al acercarse.

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

> El basemap usa teselas CARTO porque no requieren token. La especificación pide
> Mapbox GL JS; el cambio es directo cuando haya token, las dos APIs son
> compatibles.

---

## Estructura

```
.
├── PROJECT_SPEC.md
├── config.yaml                       # todos los parámetros (§10)
├── pyproject.toml
├── data/
│   ├── raw/                          # CSV crudo (gitignored)
│   ├── interim/                      # caché: red, snapping, hotspots
│   │   ├── crimes_canonical.csv      # el CSV ya proyectado a 7 columnas
│   │   ├── network.graphml           # red vial cacheada
│   │   ├── snapped.npz               # nodo asignado por crimen
│   │   └── hotspots.json             # los 480 subgrafos
│   └── processed/
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
│       ├── pois.py                   # §3.2 descarga y clasificación de POIs
│       ├── poi_taxonomy.yaml         # §3.2 mapeo OSM -> 8 categorías
│       └── network.py                # §3.3 descarga + caché + jerarquía vial
├── scripts/
│   ├── calibrate_fmin.py             # barrido de f_min contra §7.3
│   ├── sweep_params.py               # barrido autónomo σ × α × f_min × K
│   ├── build_map_data.py             # CSV -> artefacto binario del mapa
│   └── serve_map.py                  # servidor estático local
├── tests/                            # 126 tests
└── dashboard/
    └── public/
        ├── index.html                # mapa de verificación (MapLibre + D3)
        └── data/                     # artefactos generados (gitignored)
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

## El dataset

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
