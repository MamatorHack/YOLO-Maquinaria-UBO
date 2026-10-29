# Revisión de materiales del profesor (29 octubre 2026)

## Archivos aportados

- **`ML Clase10 Caras YOLO.ipynb`**: plantilla oficial de Clase 10 del Dr. Eliecer Peña Ancavil, **44 celdas**. Tiene funciones reutilizables: `crear_dataset_de_trabajo`, `contar_dataset`, `mostrar_muestras`, `explicar_etiqueta`, `filtrar_clases`, `entrenar`, `graficar_curvas`, `evaluar`, `contar_aciertos`, `galeria_real_vs_predicho`, `predecir_imagenes`, `predecir_video`. La evaluación 03 explícitamente permite basarse en esta plantilla.
- **`ejemplos.zip`** y **`ejemplos(1).zip`**: mismo contenido, nueve imágenes `Imagen01..09`, cinco afiches de películas y `video_prueba.mp4`. **No son un dataset YOLO de maquinaria**: no contienen `data.yaml` ni etiquetas de bounding boxes.
- **`Monitos_evolutivos_estudiantes (1).zip`**: contiene un archivo `.html`, no un dataset YOLO.

## Cómo adaptar la plantilla de Clase 10 (ruta recomendada)

Se conserva el flujo **Partes 0–10** y las funciones docentes; la Parte 11 es solo instructiva:

1. Cambiar el título y la primera celda a *Evaluación 03: maquinaria de construcción* con **Mathis Marsault, Julien Martinat y Louis Maillet**.
2. En `PARAMETROS`, cambiar `dataset_zip` a `"maquinaria_yolo.zip"` (o la ruta real del ZIP descargado) y `clases` a `None` si hay exactamente tres clases en `data.yaml`; si hay más, especificar tres nombres **exactos**. `imgsz=416`, `modelo_base="yolo26n.pt"`, `capas_congeladas=10`, dos fases obligatorias.
3. Para este dataset se **exigen splits train, valid/val y test originales**; no se vuelve a repartir. Si alguno falta, detenerse y revisarlo antes de proceder.
4. En **Parte 3**, inspeccionar anotaciones y conteos de cajas por clase/split; auditar duplicados exactos y **también** imágenes visualmente similares, fotogramas y renders.
5. En **Parte 4**, hacer transfer learning desde pesos preentrenados con backbone congelado. En **Parte 6**, fine tuning desde el `best.pt` propio con `AdamW` y LR bajo; guardar duración, GPU, curvas y mejor época.
6. En **Partes 5 y 7**, evaluar los **dos modelos en Validation**. **Una vez elegido**, ejecutar `contar_aciertos` del modelo seleccionado en Validation (TP/FP/FN al umbral indicado). Guardar la decisión **antes de mirar Test**.
7. En **Parte 8**, evaluar Test **una sola vez**, al mismo tamaño de entrenamiento. La galería oficial `galeria_real_vs_predicho` superpone las cajas reales en rojo y las detecciones en verde.
8. En **Parte 9**, **reemplazar** `ejemplos.zip` y `video_prueba.mp4` por nuevas fotografías y un video real de **maquinaria** **fuera del dataset**. **No** usar el respaldo del profesor de generar un video artificial a partir de Test: no demuestra generalización externa. Para Evaluación 04 hacen falta dos videos externos ≥10 segundos.
9. En **Parte 10**, exportar `transfer_learning.pt`, `fine_tuning.pt`, `modelo_elegido.pt`, gráficos y tablas; descargar inmediatamente el ZIP de resultados de Colab.
10. En el PDF (máximo 10 páginas, referencias incluidas, excepto portada/anexos), justificar decisiones con **métricas reales** y comentar al menos tres errores: objeto perdido, detección falsa y localización/clase errónea.

## Estado real

**Código preparado, resultados pendientes.** No se ha recibido el ZIP YOLO de excavadoras. Las herramientas de Colab requieren ejecutarse con GPU para producir pesos y curvas reales. El archivo `notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb` que ya está en este repositorio es una implementación **propia** del mismo protocolo; **no es** una copia literal del notebook de Clase 10.

Existe también una variante adaptada directamente de la plantilla docente preparada como artefacto de trabajo en la conversación. Para incorporar esa variante al repositorio será necesario subir su archivo `.ipynb` correspondiente. En la entrega final elegir **un único notebook ejecutado** y asegurar consistencia con el PDF y los pesos.
