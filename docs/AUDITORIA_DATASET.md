# Auditoría del dataset — Evaluación 03 (29 de octubre de 2026)

**Estado:** análisis de datos realizado sobre el ZIP exportado `Excavators.v2-raw-images_416by416.yolov8.zip`. **No** se ha entrenado ni evaluado todavía ningún modelo. Estas son estadísticas de **datos**, no métricas de YOLO.

## Identidad y licencia

- Fuente: Mohamed Sabek, **Excavators**, versión 2, Roboflow Universe.
- Enlace: https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2
- El archivo `data.yaml` del ZIP confirma `version: 2` y `license: CC BY 4.0`.
- `README.roboflow.txt` indica exportación **14/05/2024**, imágenes a **416×416 (Stretch)**, sin técnicas de aumento.
- Clases confirmadas: `0 = EXCAVATORS`, `1 = dump truck`, `2 = wheel loader`.
- Todos los archivos de imagen tienen un `.txt` asociado; las coordenadas revisadas están dentro de [0,1]. Hay **11 imágenes sin cajas** (10 Train y 1 Test), válidas como ejemplos negativos si corresponden a la realidad.

## Distribución original del ZIP descargado

| Split | Imágenes | Cajas EXCAVATORS | Cajas dump truck | Cajas wheel loader | Total cajas |
|---|---:|---:|---:|---:|---:|
| Train | 2 245 | 1 477 | 929 | 811 | 3 217 |
| Validation | 267 | 36 | 212 | 178 | 426 |
| Test | 144 | 19 | 128 | 91 | 238 |
| **Total** | **2 656** | **1 532** | **1 269** | **1 080** | **3 881** |

## Problema crítico encontrado: fuga visual

La separación **original** contenía fotogramas del **mismo video** en distintos splits, identificados por nombres como `construction--1-_mp4-...jpg`, `construction--2-_mp4-...` y `construction--3-_mp4-...`. Por ejemplo, los clips 1, 2 y 3 aparecen simultáneamente en Train, Validation y Test. La ausencia de augmentaciones no garantiza independencia entre fotogramas de una misma secuencia.

Además, no existían duplicados por SHA256 de archivos idénticos entre Train y Validation/Test, **pero sí había fotos visualmente casi idénticas** con nombres diferentes. Una comparación dHash y correlación de luminancia detectó **24 imágenes** en Validation/Test similares a imágenes de Train (10 en Validation, 14 en Test).

**Conclusión:** usar los splits originales sin corregir este problema podría inflar artificialmente las métricas. La pauta oficial prohíbe que fotogramas de la misma escena aparezcan simultáneamente en Train y Test.

## Corrección reproducible adoptada

Script: `scripts/preparar_dataset_sin_fugas.py`.

1. Mantener en **Train todos los fotogramas con nombre `*_mp4-*`**, agrupando así las secuencias de origen. Se trasladaron **161 imágenes** de Validation/Test a Train, conservando sus etiquetas.
2. Comparar las fotos restantes de Validation y Test con el contenido de Train usando **dHash de 64 bits (distancia de Hamming ≤6)** más **correlación de grises 64×64 ≥0,985**.
3. Eliminar del conjunto de trabajo las **24 copias visuales muy similares** detectadas (10 Validation, 14 Test; 27 anotaciones). Se conservaron el ZIP original y el registro `LIMPIEZA_DATOS.json` para explicar todos los cambios.
4. No se separaron aleatoriamente las copias aumentadas ni se usó Test para entrenar o elegir hiperparámetros. El reparto se corrigió **antes** de entrenar cualquier modelo.

### Distribución del dataset preparado

| Split | Imágenes | Cajas EXCAVATORS | Cajas dump truck | Cajas wheel loader | Total cajas |
|---|---:|---:|---:|---:|---:|
| Train | 2 406 | 1 477 | 1 067 | 973 | 3 517 |
| Validation | 144 | 35 | 120 | 55 | 210 |
| Test | 82 | 17 | 80 | 30 | 127 |
| **Total** | **2 632** | **1 529** | **1 267** | **1 058** | **3 854** |

**Limitaciones:** esta limpieza detecta copias muy próximas según un umbral específico y secuencias de video reconocibles por el nombre. No demuestra independencia perfecta de escenas, orígenes o sesiones de captura. Es recomendable revisar manualmente muestras de cada partición y comprobar que no existan más copias visualmente similares. Test contiene **solo 17 cajas de excavadoras**, por lo que el análisis de esta clase tendrá mayor incertidumbre. La corrección reduce deliberadamente el tamaño de Validation y Test.

## Cómo reproducir

```bash
pip install numpy Pillow
python scripts/preparar_dataset_sin_fugas.py \
  Excavators.v2-raw-images_416by416.yolov8.zip \
  Excavators.v2-UBO-sin-fugas-evidentes.yolov8.zip
```

Para entrenar con Colab: **subir el ZIP corregido**, que contiene `data.yaml`, `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels` y `LIMPIEZA_DATOS.json`.

El ZIP corregido **no se incluye en el repositorio** (el enunciado pide únicamente citar el dataset). Tampoco se publican aquí pesos ni resultados ficticios.

## Texto breve reutilizable en el informe

> Se utilizó el dataset *Excavators v2* de Roboflow Universe (licencia CC BY 4.0), con tres categorías y anotaciones YOLO. Aunque su versión no contiene aumentos artificiales, la auditoría reveló fotogramas de las mismas secuencias en diferentes particiones y 24 imágenes visualmente casi idénticas. Antes de entrenar se reagruparon todas las secuencias de video en Train y se excluyeron las copias visuales de Validation y Test. El conjunto de trabajo resultante consta de 2 406 imágenes Train, 144 Validation y 82 Test. Se mantienen registrados los cambios y se reconoce que la detección automática de copias no garantiza independencia absoluta.
