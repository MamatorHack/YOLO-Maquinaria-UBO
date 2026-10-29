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

## Faltantes obligatorios antes del informe PDF

- Salidas **en imágenes y al menos un video externos** al dataset (parte 12). Para evaluación 04 se requieren **dos videos externos de ≥10 s**.
- Registrar los enlaces/autor/licencia de fotos y videos usados y comentar aciertos y errores externos.
- Descargar el ZIP de resultados con los **tres pesos entrenados**, el notebook `.ipynb` **con salidas** y el PDF final en español.
- Incorporar gráficas legibles y tiempos/equipo a las secciones correspondientes del informe (máximo diez páginas, referencias incluidas).

**Nota técnica:** `results.csv` facilitado del fine tuning refleja las **cinco épocas** y su mAP50-95 máximo en época 3. El entrenamiento y la selección ya quedaron hechos; **no repetir Test** para obtener evidencias adicionales.
