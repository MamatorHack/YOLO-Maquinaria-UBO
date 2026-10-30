<div align="center">

# 🏗️ YOLO Maquinaria UBO

### Detección y localización de maquinaria de construcción · YOLO26n

**Auditoría de datos → transfer learning → fine tuning → Validation → Test → demostraciones externas.**

**Universidad Bernardo O'Higgins · Machine Learning · Evaluación 03 · 2026**

**Mathis Marsault · Julien Martinat · Louis Maillet**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO26n](https://img.shields.io/badge/Ultralytics-YOLO26n-1D3B59)](https://docs.ultralytics.com/models/yolo26/)
[![GPU](https://img.shields.io/badge/Colab-Tesla%20T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-219E83)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://github.com/MamatorHack/YOLO-Maquinaria-UBO/actions/workflows/validate.yml/badge.svg)](https://github.com/MamatorHack/YOLO-Maquinaria-UBO/actions/workflows/validate.yml)

**Detector entrenado por el equipo y evaluado mediante un protocolo reproducible, sin utilizar Test para seleccionar el modelo.**

![mAP50-95 Test](https://img.shields.io/badge/mAP50--95%20Test-71.66%25-087F8C?style=for-the-badge) ![Precision Test](https://img.shields.io/badge/Precision%20Test-90.13%25-294C60?style=for-the-badge) ![Clases](https://img.shields.io/badge/Clases-3-3B8F6C?style=for-the-badge)

**Ir a:** [Resultados](#resultados) · [Dataset](#dataset) · [Entrenamiento](#entrenamiento) · [Vídeos](#videos) · [Reproducción y entrega](#reproduccion)

[📓 Notebook ejecutado](notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb) · [📑 Informe PDF](report/Informe_Tecnico_E03_YOLO.pdf) · [📊 Métricas completas](docs/RESULTADOS_REALES_E03.md) · [🔍 Auditoría de datos](docs/AUDITORIA_DATASET.md)

</div>

---

## 01 / Objetivo

Detectar y localizar tres categorías en **fotografías reales de obras** mediante cajas delimitadoras:

| ID | Clase del dataset | Interpretación |
|:--:|---|---|
| `0` | **EXCAVATORS** | Excavadoras |
| `1` | **dump truck** | Camiones de volteo / camiones-benne |
| `2` | **wheel loader** | Cargadoras frontales |

**Caso de uso exploratorio:** clasificación de material audiovisual de obras, documentación de maquinaria y apoyo a inventarios visuales. No es un detector de seguridad industrial ni un sistema certificado para aplicaciones críticas.

<a id="resultados"></a>
## 02 / Resultados reales

La comparación se realizó en Validation utilizando la misma resolución **416 × 416**. La regla declarada de selección fue **máximo mAP50-95**, con desempate por mAP50; Test se usó solamente una vez, después de elegir.

<div align="center">

| Métrica del modelo elegido | Validation | Test |
|:---|---:|---:|
| **Precision** | **88,63 %** | **90,13 %** |
| **Recall** | 80,06 % | 77,05 % |
| **mAP50** | 87,08 % | 86,59 % |
| **mAP50-95** | **71,29 %** | **71,66 %** |

</div>

**Modelo elegido:** `transfer_learning.pt` → `modelo_elegido.pt`. Su mAP50-95 en Validation fue **0,712923**, frente a **0,707573** del fine tuning. El fine tuning tuvo mayor recall (**0,834979** vs **0,800567**), pero no mejoró la métrica principal de selección.

**Test final:** 82 fotografías / 127 cajas; resultados obtenidos el **29 de octubre de 2026** en Google Colab con una NVIDIA Tesla T4.

### Curvas de aprendizaje — entrenamiento ejecutado

<table>
<tr><th>Transfer learning · 8 épocas</th><th>Fine tuning · 5 épocas</th></tr>
<tr>
<td><img src="assets/figures/transfer_curves.png" width="460" alt="Curvas reales del transfer learning"></td>
<td><img src="assets/figures/fine_curves.png" width="460" alt="Curvas reales del fine tuning"></td>
</tr>
</table>

**Observación:** el transfer learning mejora de forma progresiva hasta la época 8; en fine tuning, el mejor mAP50-95 apareció en la época 3. El learning rate real de transfer learning fue **0,001429**: Ultralytics lo eligió mediante `optimizer=auto`, ignorando el `lr0=0.01` propuesto.

### Ejemplos reales: Ground Truth vs predicción

![Ejemplos con cajas reales y detecciones del modelo sobre Test](assets/figures/test_gallery.png)

*Galería de seis imágenes de Test: izquierda, etiquetas reales; derecha, predicciones de `modelo_elegido.pt`. Incluye aciertos y falsas detecciones; los errores se interpretan en el informe.*

### Recuento TP / FP / FN (Validation)

| Umbral de confianza | IoU | TP | FP | FN |
|---:|---:|---:|---:|---:|
| 0,25 | 0,50 | **165** | 35 | 45 |

Este recuento se basa en emparejamiento educativo de clases y cajas; no sustituye las métricas Ultralytics mAP/Precision/Recall.

<a id="dataset"></a>
## 03 / Dataset y auditoría anti-fuga

**Fuente:** [Excavators, Mohamed Sabek, Roboflow Universe v2](https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2) · **Formato:** YOLOv8 · **Licencia de los datos:** CC BY 4.0.

El ZIP inicial contenía **2 656 imágenes y 3 881 cajas**. Identificamos escenas de los mismos vídeos repartidas entre Train, Validation y Test y 24 imágenes casi idénticas entre particiones. Antes de entrenar se agruparon **161 fotogramas** de origen común en Train y se excluyeron las **24 imágenes** cercanas detectadas mediante dHash y correlación. La corrección no equivale a una garantía absoluta de independencia visual.

| Split corregido | Imágenes | Cajas |
|---|---:|---:|
| Train | 2 406 | 3 517 |
| Validation | 144 | 210 |
| Test | 82 | 127 |
| **Total** | **2 632** | **3 854** |

**Transparencia y trazabilidad:** [metodología, límites y tablas por clase](docs/AUDITORIA_DATASET.md) · [script de limpieza](scripts/preparar_dataset_sin_fugas.py). El dataset completo **no se entrega** en GitHub, conforme a la consigna.

<a id="entrenamiento"></a>
## 04 / Arquitectura y entrenamiento

```mermaid
flowchart LR
    A["Roboflow v2<br/>imágenes + YOLO labels"] --> B["Auditoría y limpieza<br/>sin fugas evidentes"]
    B --> C["Train / Validation / Test"]
    C --> D["Transfer learning<br/>YOLO26n, freeze=10"]
    D --> E["Fine tuning<br/>desde best.pt, AdamW"]
    D --> F["Comparación<br/>en Validation"]
    E --> F
    F --> G["Elegir mayor mAP50-95"]
    G --> H["Test UNA sola vez"]
    H --> I["Imágenes / Videos externos"]
    I --> J["PDF + Notebook + pesos"]
```

| Configuración | Transfer learning | Fine tuning |
|---|---|---|
| Pesos de inicio | `yolo26n.pt` | `best.pt` propio |
| Épocas | 8 | 5 |
| `imgsz` / `batch` | 416 / 16 | 416 / 16 |
| Capas congeladas | 10 | 0 |
| Optimizador | AdamW seleccionado por `auto` | AdamW |
| LR inicial real | 0,001429 | 0,0001 |
| Train fraction | 1,0 | 1,0 |
| Semilla | 42 | 42 |
| GPU | Tesla T4 | Tesla T4 |
| Tiempo registrado | 213,76 s | 135,60 s |


<a id="videos"></a>
## 05 / Demostraciones con material externo

Se utilizaron dos fotografías de Pexels y dos vídeos externos al conjunto de entrenamiento. Las imágenes y vídeos fueron procesados con el **modelo elegido en Validation**, sin reajustar sus pesos. Las pruebas externas son demostrativas; al no tener etiquetas de referencia, no constituyen nuevas métricas de Test.

### Fotografías externas

| Fuente | Predicción observada | Interpretación |
|---|---|---|
| [Rohan Batra, Pexels 40537857](https://www.pexels.com/photo/heavy-machinery-at-indian-construction-site-40537857/) | `wheel loader` (conf. ≈ 0,42) | Detección con confianza moderada |
| [celal keser, Pexels 36506004](https://www.pexels.com/photo/dump-truck-and-excavator-at-construction-site-36506004/) | `EXCAVATORS` (≈ 0,68) y `dump truck` (≈ 0,28) | Posible confusión; la segunda predicción es débil |

### Vídeo 1 · Maquinaria de construcción (24,5 s)

<video controls preload="metadata" width="100%" src="assets/video_1_720p.mp4">
  Tu navegador no soporta la reproducción de vídeos. Puedes <a href="assets/video_1_720p.mp4">descargar el vídeo 1 aquí</a>.
</video>

### Vídeo 2 · Excavadora en obra (19,9 s)

<video controls preload="metadata" width="100%" src="assets/video_2_720p.mp4">
  Tu navegador no soporta la reproducción de vídeos. Puedes <a href="assets/video_2_720p.mp4">descargar el vídeo 2 aquí</a>.
</video>

> **Distinción esencial:** los dos MP4 publicados en `assets/` son los **vídeos de entrada, redimensionados a 720p y 10 fps, SIN cajas YOLO**. El notebook ejecutado documenta la inferencia de 245 y 199 fotogramas; las **salidas con detecciones** se generaron en `results/video_1_streaming/` y `results/video_2_streaming/`, guardadas en el ZIP de resultados del equipo, pero no están publicadas en este repositorio. No se deben confundir originales y predicciones.

La procedencia exacta y los derechos de reutilización de ambos vídeos deberán acreditarse en la entrega. Para la Evaluación 04 se usarán los **MP4 anotados**, no las entradas sin detecciones.

### Evidencia visual de Test

![Galería de comparación: anotaciones reales frente a predicciones YOLO](assets/figures/test_gallery.png)

*Comparación de seis imágenes de Test (izquierda: cajas reales; derecha: predicciones). Incluye aciertos, confusión de clase y falsa detección. El recuento de objetos perdidos está documentado por separado en Validation.*

<a id="reproduccion"></a>
## 06 / Reproducción y entrega (Google Colab T4)

1. Abrir [notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb](notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb) en [Google Colab](https://colab.research.google.com/).
2. Ir a **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**.
3. Ejecutar desde la primera celda, **en orden**, sin omitir la configuración. La primera celda clona el repositorio y carga `src/yolo_ubo.py`.
4. Cuando se solicite el dataset, **subir el ZIP original v2 de Roboflow en formato YOLOv8**. Se aplicará automáticamente la limpieza reproducible antes del entrenamiento; no se solicita el dataset por separado en la entrega final.
5. Ejecutar transfer learning (**8 épocas**) y luego fine tuning (**5 épocas**), comparar **ambos pesos en Validation** y registrar la selección *antes* de ejecutar Test.
6. **Ejecutar Test una única vez.** A continuación, subir dos fotografías y dos videos externos, con procedencia documentada.
7. Ejecutar la celda de exportación para descargar el ZIP de evidencias y tres pesos, y en Colab **Archivo → Descargar → .ipynb** para conservar todas las salidas.

**Aviso:** el notebook publicado se basa en la ejecución real y conserva sus métricas y gráficos principales. Para reducir su tamaño en GitHub, se omitieron grandes imágenes embebidas redundantes, que siguen disponibles en `assets/figures/` o se regeneran al ejecutarlo. **Volver a ejecutar desde cero genera nuevos resultados**; no reemplazar las cifras de este informe por otras sin documentar la nueva ejecución.

### VS Code: instalar y probar sin GPU

```bash
git clone https://github.com/MamatorHack/YOLO-Maquinaria-UBO.git
cd YOLO-Maquinaria-UBO
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Para reproducir solo el proceso de limpieza local:

```bash
python scripts/preparar_dataset_sin_fugas.py \
  Excavators.v2-raw-images_416by416.yolov8.zip \
  Excavators.v2-UBO-sin-fugas-evidentes.yolov8.zip
```

### Archivos para la entrega

- [Notebook ejecutado (.ipynb)](notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb)
- [Informe técnico en PDF](report/Informe_Tecnico_E03_YOLO.pdf)
- **Pesos propios:** `transfer_learning.pt`, `fine_tuning.pt` y `modelo_elegido.pt` (entrega por aula virtual o Drive, fuera de GitHub).
- **Dataset:** no se entrega el ZIP; la fuente, versión y licencia figuran arriba.

**Límite importante:** las predicciones visualizadas en la galería muestran una confusión de clase y un falso positivo. No se afirma haber documentado un falso negativo visual específico de **Test**: el conteo FN disponible corresponde a **Validation**. El informe explica esta limitación con transparencia.
