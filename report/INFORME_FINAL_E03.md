# Detección automática de maquinaria de construcción mediante YOLO26n

**Evaluación 03 — Machine Learning · Universidad Bernardo O’Higgins**  
**Integrantes:** Mathis Marsault · Julien Martinat · Louis Maillet  
**Docente:** Dr. Eliecer Peña Ancavil  
**Fecha de entrega:** 7 de noviembre de 2026  
**Grupo de trabajo:** Marsault · Martinat · Maillet (3 integrantes). **Módulo académico:** Machine Learning. *Identificador oficial del grupo: no comunicado.*

---

## 1. Problema y dataset

### Propósito y usuarios potenciales

En una obra es habitual encontrar distintas máquinas pesadas en un mismo campo visual. Este proyecto estudia si un detector de objetos basado en YOLO26n puede **localizar y clasificar** automáticamente tres tipos de maquinaria mediante cajas delimitadoras (*bounding boxes*): `EXCAVATORS` (excavadoras), `dump truck` (camiones de volteo) y `wheel loader` (cargadoras frontales). Un resultado de este tipo podría ayudar, de manera exploratoria, a documentar imágenes de obras, clasificar contenido audiovisual y apoyar inventarios visuales. No se plantea un sistema de seguridad ni se garantiza un funcionamiento fiable en situaciones críticas.

### Fuente, licencia y clases

Se utilizó **Excavators**, de Mohamed Sabek en Roboflow Universe, **versión 2**, exportada en formato **YOLOv8** y publicada con licencia **CC BY 4.0**. Sus imágenes originales se habían redimensionado a 416 × 416 píxeles. Elegimos exactamente las tres categorías ya presentes en el `data.yaml`; por tanto, no fue necesario etiquetar manualmente ni crear clases artificiales.

**Fuente verificable:** https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2

**Tabla 1. Tamaño del conjunto de trabajo después de la auditoría y antes de entrenar.**

| Conjunto | Imágenes | Excavadoras | Camiones | Cargadoras | Cajas totales |
|---|---:|---:|---:|---:|---:|
| Train | 2 406 | 1 477 | 1 067 | 973 | 3 517 |
| Validation | 144 | 35 | 120 | 55 | 210 |
| Test | 82 | 17 | 80 | 30 | 127 |
| **Total** | **2 632** | **1 529** | **1 267** | **1 058** | **3 854** |

La disponibilidad de fotografías y vídeos reales de maquinaria permite demostrar detecciones fuera del dataset. Para la posterior Evaluación 04 se reservaron dos clips externos de al menos diez segundos, con la procedencia indicada en las referencias.

## 2. Preparación de los datos

### Anotaciones y configuración

Cada imagen se asocia a un archivo `.txt` con una línea por caja: `clase x_centro y_centro ancho alto`. Las cuatro coordenadas son proporciones de la anchura y altura de la imagen. Por ejemplo, una anotación real de Train fue `0 0.5661057692 0.4507211538 0.7572115385 0.9014423077`, que significa: clase 0 (*EXCAVATORS*), centro horizontal en el 56,61 % de la anchura, centro vertical en el 45,07 % de la altura, caja de ancho 75,72 % y alto 90,14 % del tamaño de imagen.

La estructura final contiene `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images` y `test/labels`. El archivo `data.yaml` identifica las rutas y los nombres de clase; el código transforma `valid` en `val` sin mezclar las particiones. Se revisaron etiquetas mal formadas y límites de cajas, con resultados válidos. Algunos ejemplos sin caja se mantienen como fondos negativos.

### Control de fuga entre particiones

La auditoría detectó un problema no evidente: fotogramas de **las mismas secuencias de vídeo** aparecían inicialmente en Train, Validation y Test, aunque sus archivos no eran duplicados exactos por SHA256. Para evitar que Test contuviera escenas casi memorizadas del entrenamiento, se trasladaron **161 fotogramas** al conjunto Train, agrupando los clips identificables por sus nombres. Después, un filtro de similitud visual (dHash de 64 bits, distancia ≤ 6; correlación de grises 64 × 64 ≥ 0,985) excluyó **24 fotografías casi idénticas** de Validation/Test.

La exportación original tenía 2 656 imágenes y 3 881 cajas; el conjunto de trabajo quedó en 2 632 imágenes y 3 854 cajas. Las reglas, parámetros, imágenes excluidas y recuentos están registrados en `LIMPIEZA_DATOS.json` y documentados en `docs/AUDITORIA_DATASET.md`. La limpieza se aplicó **antes de entrenar** y no respondió a resultados de Test.

**Límite:** los umbrales automáticos y los nombres de los clips no garantizan independencia absoluta de escenas similares. Las 82 imágenes de Test y, especialmente, sus **17 cajas de excavadoras** hacen que los resultados por clase deban interpretarse con cautela.

## 3. Entrenamiento

El trabajo se ejecutó en **Google Colab gratuito** con GPU **NVIDIA Tesla T4 (14 913 MiB)** y Ultralytics **8.4.192**. Se fijó una semilla 42, `imgsz=416`, lote de 16 imágenes y utilización del 100 % de Train. El entrenamiento consta de dos etapas obligatorias.

**Tabla 2. Configuraciones efectivamente aplicadas y tiempo.**

| Parámetro | Transfer learning | Fine tuning |
|---|---|---|
| Modelo de partida | `yolo26n.pt` preentrenado | `best.pt` propio del transfer |
| Épocas | 8 | 5 |
| Resolución | 416 × 416 | 416 × 416 |
| Fracción Train | 1,0 | 1,0 |
| Lote (batch) | 16 | 16 |
| Capas congeladas | `freeze=10` | `freeze=0` |
| Optimizador real | AdamW, elegido por `auto` | AdamW |
| Learning rate inicial efectivo | **0,001429** | **0,0001** |
| GPU | Tesla T4 | Tesla T4 |
| Tiempo medido | **213,76 s** | **135,60 s** |
| Mejor época | 8 | 3 |

**Importante:** en la primera etapa se pasó `lr0=0.01` como argumento, pero Ultralytics avisó que `optimizer=auto` lo **ignoró** y fijó en realidad AdamW con **lr=0,001429**. El valor correcto que se presenta es el aplicado por el entrenamiento, no el solicitado en el código.

### Evolución de las pérdidas y del mAP

![Figura 1. Pérdidas y mAP50-95 por época del transfer learning](../assets/figures/transfer_curves.png)

*Figura 1. Transfer learning (8 épocas). Las pérdidas de entrenamiento se reducen y el mAP50-95 de Validation progresa, con mejor resultado al final de esta etapa.*

![Figura 2. Pérdidas y mAP50-95 por época del fine tuning](../assets/figures/fine_curves.png)

*Figura 2. Fine tuning (5 épocas). La mejor métrica de Validation aparece en la época 3 y se estabiliza o disminuye levemente después; más épocas no implicaron automáticamente mejor generalización.*

## 4. Resultados en Validation y elección del modelo

Se calcularon Precision, Recall, mAP50 y mAP50-95 con el **mismo conjunto Validation** y resolución de 416 píxeles. La selección estaba fijada **antes de mirar Test**: mayor mAP50-95, con desempate por mAP50 y posteriormente a favor del transfer learning.

**Tabla 3. Evaluación de ambos mejores pesos en Validation (144 imágenes, 210 cajas).**

| Modelo | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **Transfer learning** | **0,8863** | 0,8006 | 0,8708 | **0,7129** |
| Fine tuning | 0,8319 | **0,8350** | **0,8749** | 0,7076 |

Se eligió **transfer learning**, con una ventaja pequeña de **0,00535** en mAP50-95 (≈0,54 puntos porcentuales). El fine tuning detectó una proporción mayor de objetos (recall +0,0344) y obtuvo mAP50 ligeramente superior, pero perdió precisión y no mejoró la métrica principal de selección. Por ello no se argumenta que el fine tuning sea inútil: ambos modelos presentan un compromiso diferente.

Para el seleccionado se contaron **165 verdaderos positivos (TP)**, **35 falsas detecciones (FP)** y **45 objetos perdidos (FN)** en Validation, utilizando confianza mínima 0,25, IoU mínimo 0,50 y emparejamiento por la misma clase. Una falsa detección puede crear un equipo inexistente en un inventario visual; un objeto perdido puede hacer que no se contabilice maquinaria visible. Estos conteos ilustrativos **no son idénticos** al cálculo de Precision/Recall de la evaluación Ultralytics, que considera barridos de umbrales.

## 5. Evaluación en Test y análisis de errores

La evaluación final se realizó **una sola vez**, después de registrar la selección, con `modelo_elegido.pt` y `imgsz=416`, sobre **82 imágenes y 127 cajas**. No se utilizó Test para elegir el modelo, ajustar hiperparámetros ni seguir entrenando.

**Tabla 4. Métricas globales del conjunto Test.**

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| **0,9013** | **0,7705** | **0,8659** | **0,7166** |

**Tabla 5. Desglose por categoría en Test.**

| Clase | Cajas | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| EXCAVATORS | 17 | 0,848 | 0,765 | 0,864 | 0,699 |
| dump truck | 80 | 0,950 | 0,713 | 0,839 | 0,685 |
| wheel loader | 30 | 0,906 | 0,833 | 0,894 | 0,766 |

El mAP50-95 pasó de 0,7129 en Validation a **0,7166 en Test**, mientras que el recall pasó de 0,8006 a **0,7705**. No se observa una caída general de mAP50-95, pero eso **no demuestra** capacidad industrial robusta: el conjunto Test es pequeño y la categoría de excavadoras tiene muy pocas instancias.

### Evidencias cualitativas y errores representativos

![Figura 3. Cajas reales y predicciones sobre imágenes Test](../assets/figures/test_gallery.png)

*Figura 3. Comparación entre las cajas reales (columna izquierda) y las detecciones del modelo (derecha) sobre seis fotografías de Test. Las predicciones muestran aciertos claros, pero también cajas espurias o confusiones entre clases.*

**Error 1 — Confusión de clase (Test).** En una vista cenital de `wheel loader`, aparece una predicción adicional `EXCAVATORS` con confianza **0,44** sobre la maquinaria. La perspectiva aérea y la geometría parcialmente visible podrían explicar esta ambigüedad. La superposición no se interpreta como un segundo equipo real.

**Error 2 — Falsa detección (Test).** En otra `wheel loader`, parte de la máquina genera una caja `dump truck` con confianza **0,26**. El resultado está próximo al umbral de 0,25, por lo que evidencia sensibilidad a rasgos visuales compartidos y a la decisión de confianza.

**Error 3 — Objeto perdido (Test: evidencia cuantitativa, no visual).** El **Recall de Test = 0,7705**, inferior a 1, demuestra que el modelo no recuperó todos los objetos anotados. Estos fallos de omisión podrían afectar a un inventario visual. No se recuperó una pareja concreta de caja real y predicción de Test para ilustrar un FN individual; por rigor, **no se atribuye este fallo a una imagen específica**. Como evidencia independiente, el conteo a umbral fijo de **Validation** fue de **45 FN** (confianza 0,25; IoU 0,50), pero no se confunden ambos conjuntos. La falta de una captura inequívoca de FN en Test continúa siendo una limitación cualitativa del material disponible.

## 6. Uso con imágenes y vídeo externos

Se probaron **dos fotografías externas** al dataset, obtenidas de Pexels, con el `modelo_elegido.pt` (416 px, confianza mínima 0,25). La primera produjo una detección `wheel loader`; la segunda produjo `EXCAVATORS` y `dump truck`. En las imágenes anotadas recibidas se observaron respectivamente confianzas aproximadas de **0,42** para la cargadora y de **0,68 / 0,28** para las otras dos categorías. La detección de confianza 0,28 es débil y podría ser espuria; no se presenta como un acierto seguro. Las imágenes con las cajas predichas se generaron en `results/external_images` del respaldo ZIP del equipo; **el notebook entregado solo incorpora sus registros de inferencia, no estas capturas**, por lo que no se incluyen supuestas imágenes anotadas que no se hayan recuperado. La primera fotografía presenta una detección con confianza moderada y la segunda muestra una detección de baja confianza que requiere comprobación humana.

**Tabla 6. Material externo utilizado.**

| Recurso | Procedencia | Resultado verificado |
|---|---|---|
| Fotografía 1 | Pexels, Rohan Batra, imagen 40537857 | 1 `wheel loader` |
| Fotografía 2 | Pexels, celal keser, imagen 36506004 | 1 `EXCAVATORS` y 1 `dump truck` |
| Vídeo 1 | [Entrada 720p publicada por el equipo](https://github.com/MamatorHack/YOLO-Maquinaria-UBO/blob/main/assets/video_1_720p.mp4), original `16517436_3840_2160_30fps.mp4` | 245 fotogramas procesados |
| Vídeo 2 | [Entrada 720p publicada por el equipo](https://github.com/MamatorHack/YOLO-Maquinaria-UBO/blob/main/assets/video_2_720p.mp4), original `16499298_3840_2160_30fps.mp4` | 199 fotogramas procesados |

Ambos vídeos externos se transformaron de 4K a **720p / 10 fps** para limitar el uso de RAM de Colab, sin modificar el modelo. La inferencia se realizó con `stream=True` para no almacenar todas las predicciones en memoria. Se obtuvieron clips procesados de **24,5 y 19,9 segundos** aproximadamente. Los vídeos se introdujeron como material externo al entrenamiento; su independencia respecto del dataset se declara según el protocolo utilizado, sin equivaler a una comprobación exhaustiva de procedencia. Los dos clips anotados se guardaron en `results/video_1_streaming` y `results/video_2_streaming` en el archivo de evidencias, fuera del repositorio por su tamaño. **Los MP4 accesibles desde la tabla son únicamente las entradas sin cajas YOLO**: no deben presentarse como vídeos anotados. La URL original de sus autores no está documentada.

Estas pruebas muestran que el detector funciona en fuentes externas al dataset, pero las dos imágenes y dos vídeos **no constituyen una nueva prueba estadística independiente**. La iluminación, la distancia, el tamaño aparente de las máquinas y el punto de vista pueden modificar tanto las confianzas como la localización de las cajas. En particular, una predicción con confianza baja exige verificación humana.

## 7. Conclusiones y límites

El equipo construyó y entrenó un detector YOLO26n propio para tres tipos de maquinaria, comparó de forma controlada dos estrategias y seleccionó un modelo **sin consultar Test**. El modelo elegido alcanzó **mAP50=0,8659** y **mAP50-95=0,7166** en el conjunto Test depurado, con una **Precision=0,9013** y un **Recall=0,7705**. Este último valor también confirma que el detector pierde algunos objetos; los ejemplos visuales publicados no identifican todavía un caso concreto de FN en Test. La demostración en fotografías y vídeos externos confirma que puede producir cajas sobre material nuevo, aunque también aparecen falsas detecciones y clasificaciones confusas.

El fine tuning mejoró recall y mAP50 en Validation, pero no mejoró el mAP50-95 usado como criterio de selección. El resultado ilustra la importancia de **comparar modelos con una regla declarada**, y no elegir automáticamente la etapa más larga.

Las limitaciones más importantes son el tamaño y el reparto de Test, la fuerte desproporción de cajas por categoría, la similitud visual entre máquinas, la dependencia del punto de vista y la detección no exhaustiva de escenas duplicadas. Ninguna cifra permite afirmar seguridad operativa, recuentos infalibles ni precisión estable fuera de las condiciones evaluadas.

Con más tiempo de GPU se podría ampliar el dataset con obras, cámaras y condiciones meteorológicas variadas; auditar manualmente las etiquetas; incorporar escenas agrupadas por fuente; realizar entrenamientos de más épocas y varias semillas con selección exclusiva en Validation, y estudiar tamaños de imagen alternativos antes de una **nueva** evaluación final en un Test independiente. Son **mejoras propuestas**, no realizadas.

## Referencias

1. Universidad Bernardo O’Higgins. *Evaluación 03 — Proyecto YOLO: un detector de objetos entrenado por ustedes* (consigna de la asignatura, 2026) y *Formato de informe Evaluación 03*.
2. Mohamed Sabek. *Excavators — Computer Vision Model*, dataset v2, Roboflow Universe. CC BY 4.0. https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2
3. Ultralytics. *YOLO26 Documentation*. https://docs.ultralytics.com/models/yolo26/
4. Pexels. Rohan Batra. Imagen 40537857. https://www.pexels.com/photo/heavy-machinery-at-indian-construction-site-40537857/
5. Pexels. celal keser. Imagen 36506004. https://www.pexels.com/photo/dump-truck-and-excavator-at-construction-site-36506004/
6. Material audiovisual externo empleado en el notebook: `16517436_3840_2160_30fps.mp4` (24,5 s) y `16499298_3840_2160_30fps.mp4` (19,9 s). Copias de entrada reproducibles: https://github.com/MamatorHack/YOLO-Maquinaria-UBO/blob/main/assets/video_1_720p.mp4 y https://github.com/MamatorHack/YOLO-Maquinaria-UBO/blob/main/assets/video_2_720p.mp4. Los identificadores no permiten verificar **autoría, fuente editorial ni licencia original**; estas referencias bibliográficas siguen incompletas.

---

*Código, anotaciones de control, CSV de entrenamiento, gráficos de pérdidas, métricas, pesos y vídeos anotados documentados en el repositorio de la asignatura y en el archivo de resultados del equipo. Los modelos `.pt` se entregan por separado.*
