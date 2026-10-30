# Resultados experimentales reales — Evaluación 03

**Datos compartidos por el equipo el 29/10/2026 a partir de la ejecución de Google Colab**. Fuente: consola Ultralytics, tabla Validation, JSON de selección, recuento TP/FP/FN y registro Test; gráficos y `results.csv` de fine tuning. Este documento **no sustituye** el notebook ejecutado ni los pesos `.pt`.

## Dataset (antes de entrenar)

Se utilizó el ZIP corregido de Excavators v2 (licencia CC BY 4.0), con **2406 imágenes Train / 3517 cajas**, **144 imágenes Validation / 210 cajas** y **82 imágenes Test / 127 cajas**. Se documentaron 161 fotogramas reasignados a Train y 24 imágenes con coincidencias visuales excluidas, con las limitaciones descritas en `AUDITORIA_DATASET.md`.

## Hardware y configuraciones

| Parámetro | Transfer learning | Fine tuning |
|---|---|---|
| Arquitectura | YOLO26n | YOLO26n |
| Inicialización | `yolo26n.pt` preentrenado | `best.pt` de transfer learning |
| Épocas | 8 | 5 |
| `imgsz` | 416 | 416 |
| Batch | 16 | 16 |
| Fracción Train | 1,0 | 1,0 |
| `freeze` | 10 | 0 |
| Optimizador | Auto → AdamW | AdamW |
| Learning rate efectivo | **0,001429** elegido por `optimizer=auto` (el `lr0=0.01` enviado se ignoró) | **0,0001** |
| Semilla | 42 | 42 |
| GPU | Tesla T4, 14 913 MiB | Tesla T4, 14 913 MiB |
| Duración registrada por función | 213,76 s | 135,60 s |
| Mejor época según curvas/registro | 8 | 3 (mAP50-95 ≈ 0,7075) |

**Advertencia de transparencia:** no informar el learning rate 0,01 como real en la fase de transfer learning. Ultralytics indicó explícitamente que `optimizer=auto` reemplazó dicho valor por AdamW con `lr=0.001429`.

## Validation (métricas finales recalculadas sobre los pesos best.pt)

| Modelo | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Transfer learning | **0,886345** | 0,800567 | 0,870787 | **0,712923** |
| Fine tuning | 0,831851 | **0,834979** | **0,874891** | 0,707573 |

**Regla de selección preestablecida:** máximo mAP50-95 en Validation; desempate mAP50 y luego transfer learning.

**Modelo elegido:** **transfer learning**. Presentó mejor mAP50-95 por **0,005350** (≈0,54 puntos porcentuales) y mejor precision (+0,054494), mientras que fine tuning tuvo recall (+0,034411) y mAP50 (+0,004104) ligeramente superiores. Esta pequeña diferencia indica un compromiso entre errores de localización estrictos y tasa de detección; no garantiza que uno sea universalmente superior.

**Conteos sobre Validation, con `confidence=0.25`, `IoU=0.50` y matching educativo:**
- TP: **165**.
- FP: **35**.
- FN: **45**.

Los 165 aciertos corresponden a emparejamientos de predicciones con objetos anotados de la misma clase. Las 35 detecciones falsas y los 45 objetos perdidos justifican analizar escenas difíciles en el informe. Estos conteos usan un umbral específico y **no** equivalen exactamente al procedimiento de cálculo de precision/recall agregado por Ultralytics.

## Evaluación Test (una sola vez)

**Únicamente después de elegir en Validation**, se evaluó el modelo `modelo_elegido.pt`, con `imgsz=416`, en **82 imágenes / 127 objetos**.

| Métrica | Valor |
|---|---:|
| Precision | **0,901339** |
| Recall | **0,770506** |
| mAP50 | **0,865897** |
| mAP50-95 | **0,716560** |

**Por clase (salida Ultralytics Test):**

| Clase | Cajas | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| EXCAVATORS | 17 | 0,848 | 0,765 | 0,864 | 0,699 |
| dump truck | 80 | 0,950 | 0,713 | 0,839 | 0,685 |
| wheel loader | 30 | 0,906 | 0,833 | 0,894 | 0,766 |

Comparado con Validation: mAP50-95 +0,003637; mAP50 −0,004890; recall −0,030061; precision +0,014995. No hay evidencia de caída significativa de mAP50-95, pero el conjunto Test es **pequeño** y la clase excavadoras tiene solo **17 objetos anotados**; evitar conclusiones fuertes sobre generalización.

## Observaciones visuales en la galería Test

- Se aprecia detección correcta de `wheel loader` con confianza 0,93 y 0,92; y un `dump truck` con confianza 0,91.
- En una vista aérea de maquinaria, un mismo objeto `wheel loader` recibió también una predicción `EXCAVATORS` (confianza 0,44), posible falsa detección/confusión de clases.
- En otra imagen de una cargadora, aparece una predicción `dump truck` de confianza 0,26 sobre parte de la máquina, una posible falsa detección.
- **Pendiente:** seleccionar y justificar un objeto realmente perdido (FN) y documentar por separado una caja mal ubicada o confusión, incluyendo nombre de imagen y contexto. No declarar los tres casos completos hasta identificar la evidencia específica.

## Demostración externa documentada — Parte 12

**Fotografías externas (Pexels, capturas aportadas por el equipo):**
- `pexels-rohanbatra-40537857.jpg`: una cargadora `wheel loader` detectada con confianza **0,42**. Detección correcta pero con confianza relativamente baja.
- `pexels-celalkeser-36506004.jpg`: el modelo dibujó una caja `EXCAVATORS` con confianza **0,68** alrededor de buena parte del camión y otra `dump truck` con confianza **0,28** sobre la maquinaria del fondo. Este caso ilustra posible confusión, cajas mal localizadas y limitaciones al distinguir maquinaria solapada. Requiere inspección visual para describir con rigor cuál maquinaria está presente en cada caja.

**Video externo:**
- Original: `16517436_3840_2160_30fps.mp4` (4K, 30 fps, **735 fotogramas ≈24,5 s**).
- Para evitar saturación RAM en Colab se creó `video_1_720p.mp4` (720p, 10 fps); su inferencia con `stream=True` terminó correctamente **245 fotogramas**.
- Resultado indicado por Colab: `results/video_1_streaming`. No confundir el video de entrada optimizado con la copia de salida anotada por YOLO.
- Se observan detecciones `EXCAVATORS` y `dump truck` durante numerosos fotogramas, según los logs aportados; falta comprobar el video renderizado completo para evaluar falsos positivos y pérdidas. **No se reportan métricas de calidad cuantitativas de esta secuencia porque no tiene anotaciones ground-truth.**

**Respaldo:** se verificó la existencia de los tres pesos `transfer_learning.pt`, `fine_tuning.pt` y `modelo_elegido.pt` y se generó `YOLO_E03_resultados.zip` de **459,3 MB** en Colab. El ZIP se ha aportado a la conversación; el notebook ejecutado `.ipynb` debe descargarse también por separado.

## Faltantes obligatorios antes del informe PDF

- **Parte 12 completada en Colab:** imágenes externas y un video externo; reunir capturas seleccionadas y comprobar el video anotado. Para Evaluación 04 queda por procesar un segundo video externo de ≥10 s.
- Registrar los enlaces/autor/licencia de fotos y videos usados y comentar aciertos y errores externos.
- **ZIP de resultados y tres pesos confirmados en Colab.** Pendientes: notebook `.ipynb` **con salidas** descargado y PDF final en español.
- Incorporar gráficas legibles y tiempos/equipo a las secciones correspondientes del informe (máximo diez páginas, referencias incluidas).

**Nota técnica:** `results.csv` facilitado del fine tuning refleja las **cinco épocas** y su mAP50-95 máximo en época 3. El entrenamiento y la selección ya quedaron hechos; **no repetir Test** para obtener evidencias adicionales.


## Evidencia final de las demostraciones externas (ejecución 29/10/2026)

- **Foto 1:** `pexels-rohanbatra-40537857.jpg`, predicción de `wheel loader`; el material visual proporcionado mostró confianza aproximada 0,42.
- **Foto 2:** `pexels-celalkeser-36506004.jpg`, predicciones de `EXCAVATORS` y `dump truck`, con confianzas aproximadas 0,68 y 0,28. El segundo resultado es débil y debe evaluarse como posible falsa detección.
- **Vídeo 1:** `16517436_3840_2160_30fps.mp4`, salida `results/video_1_streaming`, **245 fotogramas** en 720p a 10 fps.
- **Vídeo 2:** `16499298_3840_2160_30fps.mp4`, salida `results/video_2_streaming`, **199 fotogramas** en 720p a 10 fps.
- **Respaldo final:** `YOLO_E03_resultados.zip` creado por Colab, **508,61 MiB** según la salida del notebook; pesos `transfer_learning.pt`, `fine_tuning.pt` y `modelo_elegido.pt` confirmados presentes. Este ZIP no se publica en GitHub; debe entregarse por plataforma o Drive.
- **CSV real del fine tuning:** [`assets/metrics/fine_tuning_results.csv`](../assets/metrics/fine_tuning_results.csv), con 5 épocas y máximo mAP50-95 en época 3.

El [informe PDF final](../report/Informe_Tecnico_E03_YOLO.pdf) contiene resultados, gráficos y análisis. El PDF fue validado automáticamente en 7 páginas totales (1 portada + 6 de informe). Los enlaces exactos de origen de los dos videos deben acreditarse antes de la entrega.
