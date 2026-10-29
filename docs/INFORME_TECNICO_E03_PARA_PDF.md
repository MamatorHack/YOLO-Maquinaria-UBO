# Detección automática de maquinaria de construcción mediante YOLO26n

**Universidad Bernardo O'Higgins · Machine Learning · Evaluación 03**  
**Profesor:** Dr. Eliecer Peña Ancavil  
**Integrantes:** Mathis Marsault · Julien Martinat · Louis Maillet  
**Grupo:** [INDICAR NÚMERO] · **Módulo:** [INDICAR MÓDULO]  
**Fecha de entrega:** 7 de noviembre de 2026

> **Informe técnico en español, basado exclusivamente en los resultados reales compartidos por el equipo y en el material de la asignatura.** Versión preparada para maquetar como PDF (máximo 10 páginas sin portada ni anexos; las referencias cuentan). Mantener figuras, pies y referencias en el cuerpo. **Falta identificar con una captura inequívoca un falso negativo del conjunto Test antes de considerar definitivo el apartado de errores.**

## 1. Problema y dataset

En una obra pueden encontrarse diferentes tipos de maquinaria pesada, parcialmente ocultos, a distintas escalas y bajo condiciones cambiantes de iluminación. Este proyecto desarrolla un detector experimental de objetos mediante **YOLO26n** para identificar en fotografías y video tres categorías: **EXCAVATORS** (excavadoras), **dump truck** (camiones volquete) y **wheel loader** (cargadoras sobre ruedas). Su aplicación potencial es asistir a tareas de documentación visual de obras y revisión de material audiovisual. El modelo **no se ha validado** para decisiones críticas de seguridad, inventario oficial de equipos ni supervisión autónoma.

El conjunto de datos proviene de **Mohamed Sabek, *Excavators*, versión 2**, exportado de Roboflow Universe en formato YOLOv8 y disponible bajo **CC BY 4.0**: https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2. Las imágenes se exportaron a 416 × 416 píxeles; se conservaron las tres clases del archivo `data.yaml`. Antes de la auditoría, el ZIP contenía **2656 imágenes y 3881 cajas**. No se utilizaron renders ni capturas de videojuegos como material añadido por el equipo; debe reconocerse que no se auditó manualmente cada imagen del corpus original para descartar cualquier render residual.

**Tabla 1. Distribución final utilizada tras la auditoría.**

| Partición | Imágenes | EXCAVATORS | dump truck | wheel loader | Total de cajas |
|---|---:|---:|---:|---:|---:|
| Train | 2406 | 1477 | 1067 | 973 | 3517 |
| Validation | 144 | 35 | 120 | 55 | 210 |
| Test | 82 | 17 | 80 | 30 | 127 |
| **Total** | **2632** | **1529** | **1267** | **1058** | **3854** |

*Figura 1 (insertar): seis ejemplos de Train con etiquetas originales verdes y nombre de clase, `results/annotations_train.png`.* Las imágenes ilustran cómo varía la escala de las excavadoras y su entorno. Las etiquetas son **verdad de referencia (ground truth)** y todavía no corresponden a predicciones del modelo.

## 2. Preparación de los datos

Se trabajó con el formato de detección YOLO: las fotografías se almacenan en `images/` y cada imagen lleva un archivo `.txt` homónimo en `labels/`. Cada línea representa una caja: `clase x_centro y_centro ancho alto`, donde las cuatro coordenadas están normalizadas entre cero y uno. El archivo `data.yaml` declara las rutas `train/images`, `val/images`, `test/images` y los tres nombres de clase. Se comprobó la presencia de etiquetas, la validez de los identificadores de clase y que las coordenadas están en los rangos permitidos.

La inspección inicial reveló **fuga potencial de datos**: fotogramas procedentes de una misma secuencia de video estaban distribuidos entre Train, Validation y Test; también había fotografías casi idénticas con nombres distintos. Aunque la versión 2 no incorporaba aumentos artificiales declarados, eso **no aseguraba la independencia entre particiones**. Por ello se ejecutó una limpieza determinista **antes del entrenamiento y antes de observar las métricas**:

1. Se agruparon en Train **161 fotogramas de video** que estaban fuera de Train y pertenecían a las secuencias identificables por sus nombres.
2. Se eliminaron del conjunto de trabajo **24 fotografías casi duplicadas** presentes en Validation/Test, según una combinación de `dHash` de 64 bits (distancia ≤ 6) y correlación de escala de grises de 64 × 64 (≥ 0,985).
3. Se conservaron el ZIP original y un registro `LIMPIEZA_DATOS.json` con los cambios. La separación final se mantuvo durante el entrenamiento y la selección del modelo.

El método reduce la fuga evidente, pero **no garantiza que no existan otras tomas del mismo lugar, máquina o sesión**. Además, Test quedó con 82 imágenes y únicamente 17 objetos etiquetados como EXCAVATORS, lo que limita la precisión de conclusiones por clase. No se seleccionaron hiperparámetros en función de Test.

*Figura 2 (insertar): distribución de cajas por clase y partición, calculada a partir de la Tabla 1.*  
*Figura 3 (insertar): ejemplo de archivo de anotación con cinco campos y cajas reales superpuestas.*

## 3. Entrenamiento

Se realizaron **dos entrenamientos propios y sucesivos** en Google Colab gratuito con GPU **NVIDIA Tesla T4 (14 913 MiB)**. Se utilizó Ultralytics **8.4.192**, arquitectura **YOLO26n**, resolución **416 × 416 píxeles**, `batch=16`, fracción de Train **1,0**, semilla **42** y el dataset ya depurado. Las transformaciones estándar del entrenamiento que registra Ultralytics operan exclusivamente sobre Train.

**Primera etapa: transfer learning.** El modelo partió de `yolo26n.pt` preentrenado. Se congelaron las diez primeras capas (`freeze=10`) y se entrenó durante ocho épocas. Se había suministrado `lr0=0.01`, pero el registro de Ultralytics precisa que `optimizer=auto` **ignoró ese valor** y seleccionó **AdamW, `lr=0.001429`**. La duración registrada por la función fue **213,76 segundos**. El mAP50-95 en Validation creció aproximadamente de **0,603 en la época 1** a **0,714 en la época 8**; el archivo elegido fue el `best.pt` de esta fase.

**Segunda etapa: fine tuning.** Se inició desde el **`best.pt` entrenado por el equipo**, se descongelaron todas las capas (`freeze=0`) y se utilizó **AdamW, `lr0=0.0001`** durante cinco épocas. El tiempo registrado fue **135,60 segundos**. El mejor mAP50-95 de esta fase ocurrió alrededor de la **época 3 (≈0,708)**; las épocas 4 y 5 no lo mejoraron. Esto señala estabilización de la validación, mientras que algunas pérdidas de Train siguieron descendiendo, sin demostrar por sí solo sobreajuste significativo.

**Tabla 2. Comparación experimental.**

| Parámetro | Transfer learning | Fine tuning |
|---|---|---|
| Inicialización | yolo26n.pt | best.pt de transfer learning |
| Congelación | 10 capas | 0 capas |
| Épocas | 8 | 5 |
| Resolución / batch | 416 / 16 | 416 / 16 |
| Fracción de Train | 100 % | 100 % |
| Optimizador | AdamW por auto | AdamW |
| Tasa inicial efectiva | 0,001429 | 0,0001 |
| GPU | Tesla T4 | Tesla T4 |
| Duración | 213,76 s | 135,60 s |
| Mejor época | 8 | 3 |

*Figura 4: pérdidas de entrenamiento/validación y mAP50-95 por época del transfer learning (`results/transfer_curves.png`).*  
*Figura 5: mismas curvas para fine tuning (`results/fine_curves.png`).*

Las curvas indican progreso claro del transfer learning. El fine tuning mejora ciertas detecciones, pero no supera el primer modelo en la métrica más estricta definida para la selección.

## 4. Resultados en Validation y elección del modelo

Se evaluaron **ambos archivos `best.pt` sobre las mismas 144 imágenes de Validation (210 objetos)**, con tamaño de imagen 416. Las métricas utilizadas fueron precision, recall, mAP50 (IoU 0,50) y mAP50-95 (media de mAP para umbrales IoU de 0,50 a 0,95).

**Tabla 3. Métricas de Validation.**

| Modelo | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **Transfer learning** | **0,886345** | 0,800567 | 0,870787 | **0,712923** |
| Fine tuning | 0,831851 | **0,834979** | **0,874891** | 0,707573 |

**Regla de selección fijada antes de consultar Test:** elegir el mayor **mAP50-95 en Validation**; ante un empate, mAP50; ante un segundo empate, transfer learning. Por esta regla se seleccionó **transfer learning** y se copió a `modelo_elegido.pt`. El margen de mAP50-95 fue pequeño, **0,005350** (aprox. 0,54 puntos porcentuales). El fine tuning ofreció mayor recall, pero menor precision y mAP50-95. Ninguna mejora aislada se utilizó para cambiar la regla.

Con el modelo seleccionado se realizó un recuento explicativo en Validation usando umbral de confianza **0,25**, emparejamiento por clase y **IoU ≥0,50**: **165 aciertos (TP), 35 falsas detecciones (FP) y 45 objetos perdidos (FN)**. Esos valores revelan los tipos de error que puede encontrar una herramienta de documentación visual: elementos inexistentes que se reportan o máquinas que no se reconocen. **No deben compararse directamente** con precision/recall globales de Ultralytics porque la metodología de umbrales y agregación es distinta.

*Figura 6: barras comparativas de las cuatro métricas de Validation y recuento TP/FP/FN con umbrales explícitos.*

## 5. Evaluación en Test y análisis de errores

**Test se evaluó una sola vez**, después de guardar la selección en `results/selection.json`, y siempre a **416 píxeles**, sobre **82 imágenes con 127 cajas anotadas**.

**Tabla 4. Métricas finales del modelo elegido sobre Test.**

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| **0,901339** | **0,770506** | **0,865897** | **0,716560** |

**Tabla 5. Desglose por clase en Test (valores de Ultralytics redondeados).**

| Clase | Objetos | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| EXCAVATORS | 17 | 0,848 | 0,765 | 0,864 | 0,699 |
| dump truck | 80 | 0,950 | 0,713 | 0,839 | 0,685 |
| wheel loader | 30 | 0,906 | 0,833 | 0,894 | 0,766 |

El mAP50-95 de Test (**0,7166**) es muy próximo al de Validation (**0,7129**): diferencia de **+0,0036**. El recall baja aproximadamente **0,0301**, mientras la precision sube **0,0150**. En esta muestra no se observa una caída pronunciada del mAP global, pero **el Test es reducido**: no se afirma que la calidad se mantenga en cualquier obra, cámara, estación o país.

**Errores cualitativos observables en la galería de Test:**

- **Falsa detección:** en la imagen de una cargadora frontal de color amarillo fotografiada de tres cuartos, el modelo genera además una pequeña caja **`dump truck`** sobre una sección de la cargadora (**confianza 0,26**), aunque la anotación real solo indica `wheel loader`. Es un ejemplo de falsa alarma.
- **Confusión de clases:** en una toma aérea con `wheel loader` anotado, el modelo añade una predicción `EXCAVATORS` (**confianza 0,44**) sobre el mismo equipo, junto a la predicción `wheel loader` (**0,63**). La vista cenital y la silueta de maquinaria pesada pueden contribuir a este error.
- **Objeto perdido (FN):** **[PENDIENTE: elegir en Test una imagen concreta en la que una caja anotada no tenga predicción emparejada de la misma clase con IoU ≥0,50 y confianza ≥0,25; incluir nombre de archivo, captura GT/pred y explicación].** Las 45 omisiones cuantificadas anteriormente corresponden a Validation, no necesariamente a la galería de Test.

*Figura 7: galería de Test con **cajas reales** y **predicciones**, lado a lado (`results/test_gallery.png`). Insertar además el ejemplo del FN pendiente con su imagen identificada.*

Estos errores pueden relacionarse con similitudes geométricas entre máquinas, objetos pequeños, oclusiones, ángulos no habituales y confianza reducida, pero se trata de **hipótesis**, no de causas demostradas experimentalmente.

## 6. Uso con imágenes y video externos

Se aplicó el modelo elegido, **sin reentrenarlo**, a **dos fotografías externas a Roboflow**, aportadas desde Pexels, y a **dos videos externos**, cada uno procesado con inferencia en streaming. Se conservó `imgsz=416` y el umbral de confianza `conf=0.25`.

**Fotografía A, cargadora:** en `pexels-rohanbatra-40537857.jpg`, el modelo detectó una `wheel loader` con **confianza 0,42**. La localización es plausible, aunque la confianza refleja que este escenario externo puede ser más difícil que las imágenes de evaluación.

**Fotografía B, maquinaria mixta:** en `pexels-celalkeser-36506004.jpg`, el sistema generó una detección `EXCAVATORS` y una `dump truck`. La inspección de la fotografía revela que la caja de `EXCAVATORS` se sitúa sobre el camión principal, mientras que la caja `dump truck` se dirige a una máquina del fondo: ejemplo de **clasificación/localización insuficiente fuera del dataset**. No se reporta mAP externo porque estas imágenes no tienen anotaciones de referencia validadas por el equipo.

**Video 1:** `16517436_3840_2160_30fps.mp4` convertido a **720p/10 fps**, con **245 cuadros procesados**; resultado en `results/video_1_streaming`. La ejecución original en 4K saturaba RAM, por lo que se emplearon reducción de resolución y `stream=True` sin modificar los pesos del modelo. **Video 2:** `16499298_3840_2160_30fps.mp4` convertido también a **720p/10 fps**, con **199 cuadros procesados**, resultado en `results/video_2_streaming`.

El procesamiento completo confirma viabilidad técnica de inferencia en los dos videos, pero **no demuestra precisión fotograma a fotograma** sin anotaciones manuales. Las salidas deben revisarse visualmente y presentar al menos un fotograma correcto y otro difícil. Las fuentes exactas de ambos videos, autores y condiciones de licencia deben verificarse y citarse antes del PDF final; el nombre numérico del MP4 no permite atribuirlas con certeza.

*Figura 8: dos fotografías externas con resultados anotados, `results/external_images/`.*  
*Figura 9: un fotograma representativo de cada video procesado, extraído de `results/video_1_streaming` y `results/video_2_streaming`.*

## 7. Conclusiones y límites

El objetivo de **entrenar y evaluar un detector propio de tres clases** se cumplió en las dos fases requeridas. El modelo seleccionado por una regla definida previamente fue el **transfer learning**; alcanzó en Test **mAP50 de 0,8659**, **mAP50-95 de 0,7166**, **precision de 0,9013** y **recall de 0,7705**. Estas métricas, obtenidas en una muestra pequeña, muestran que puede identificar muchas máquinas en imágenes de características similares a las del dataset, pero también existen omisiones y falsas alarmas.

La comparación resulta instructiva: el fine tuning no siempre mejora la métrica usada para seleccionar un detector, aun cuando aumente el recall. La limpieza previa de datos también fue decisiva metodológicamente, porque el dataset original mezclaba secuencias y escenas visualmente similares. Sin esa revisión las cifras podrían ser demasiado optimistas.

Las principales limitaciones son: **82 imágenes Test**, **17 excavadoras anotadas**, posible fuga visual residual, desbalance de clases, cambios de dominio entre fotografías de dataset y escenas externas, e inferencia a 416 píxeles que puede perjudicar objetos pequeños. Las dos demostraciones en video son pruebas funcionales, no certificaciones de robustez. Con más disponibilidad GPU proponemos comparar más épocas mediante Validation, ampliar con nuevos contextos y anotaciones independientes, y ensayar otra resolución; estas mejoras **no se realizaron** durante esta evaluación. No se modifica el resultado Test en función de dichas propuestas.

## Referencias

1. Universidad Bernardo O'Higgins. *Instrucciones Evaluación 03: Proyecto YOLO* y *FormatoEntregable*, materiales docentes, octubre de 2026.
2. Sabek, Mohamed. *Excavators, versión 2*, Roboflow Universe, licencia CC BY 4.0. https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2
3. Ultralytics. *Documentación y biblioteca de YOLO*. https://docs.ultralytics.com/
4. Batra, Rohan. Fotografía externa `pexels-rohanbatra-40537857.jpg`. https://www.pexels.com/photo/heavy-machinery-at-indian-construction-site-40537857/
5. Keser, celal. Fotografía externa `pexels-celalkeser-36506004.jpg`. https://www.pexels.com/photo/dump-truck-and-excavator-at-construction-site-36506004/
6. **[PENDIENTE]** Autores, URL y licencia de los videos `16517436_3840_2160_30fps.mp4` y `16499298_3840_2160_30fps.mp4`, utilizados como pruebas externas.

---

**Archivos del experimento que deben entregarse aparte:** `E03_YOLO_Marsault_Martinat_Maillet.ipynb` ejecutado (con salidas); `transfer_learning.pt`; `fine_tuning.pt`; `modelo_elegido.pt`. El dataset no se entrega. El archivo `YOLO_E03_resultados.zip` generado antes de la segunda demostración debe actualizarse si se quiere incluir el video anotado 2.
