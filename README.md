# YOLO Maquinaria UBO — Evaluación 03

**Universidad Bernardo O'Higgins · Machine Learning · Evaluación 03 (25 %) · 7 noviembre 2026, 10:10**

**Equipo:** Mathis Marsault · Julien Martinat · Louis Maillet  
**Docente:** Dr. Eliecer Peña Ancavil

> **Estado:** código y protocolo de evaluación preparados. El entrenamiento, las métricas y el informe final **no** se consideran terminados hasta ejecutar el notebook con GPU y datos reales. No se inventan resultados.

## Objetivo

Entrenar **nuestro propio detector YOLO** para tres tipos de maquinaria de construcción: `EXCAVATORS`, `dump truck` y `wheel loader`. Comparar transfer learning y fine tuning, seleccionar el modelo únicamente con **Validation**, evaluarlo **una sola vez en Test** y demostrarlo con imágenes y videos **externos al dataset**.

- **Fuente propuesta:** Mohamed Sabek, *Excavators*, **versión 2**: https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2
- **Licencia declarada por el autor:** CC BY 4.0 (confirmar la versión al descargar).
- La fuente indica que v2 contiene imágenes redimensionadas a 416×416 **sin aumentos**. De todos modos, **conservamos los splits originales** y auditamos duplicados entre ellos.
- **Inscripción obligatoria:** comprobar disponibilidad del tema en el foro de su módulo; máximo tres grupos por tema.

## Inicio rápido en Google Colab (GPU T4)

1. Descargar el ZIP en formato **YOLO** desde Roboflow Universe (versión 2). No subirlo al repositorio.
2. Abrir `notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb` en Google Colab.
3. Elegir **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**.
4. Ejecutar las celdas en orden. Cuando se solicite, subir el ZIP del dataset.
5. Al terminar, descargar inmediatamente el ZIP de resultados generado y respaldarlo en Google Drive: **/content se pierde al terminar la sesión**.
6. Subir fotografías externas a `external/images/` y al menos un video externo a `external/videos/` para probar el detector. Para Evaluación 04 se necesitarán **dos videos de ≥10 segundos** y fuera del dataset.

**No se usa el modelo preentrenado como entrega**: las tres copias finales de pesos derivan de entrenamiento propio. Si el profesor proporciona `ML Clase10 Caras YOLO.ipynb`, contrastar nuestro notebook con su plantilla antes de entregar.

## Estructura

```text
notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb  # Flujo completo Colab
src/yolo_ubo.py                                     # Preparación, auditoría, entrenamiento, evaluación, exportación
tests/test_yolo_ubo.py                              # Pruebas con datos sintéticos
docs/INFORME_BORRADOR.md                            # Secciones del informe, sin métricas inventadas
docs/FORO_INSCRIPCION.md                            # Mensaje listo para adaptar al foro
requirements.txt                                     # Dependencias
models/                                             # .pt generados (no incluidos)
results/                                            # Evidencia generada (no incluida)
external/                                           # Imágenes/videos de demostración (no incluidos)
```

## Protocolo (evitar errores que invalidan puntos)

1. Mantener **Train / Validation / Test** separados; no volver a mezclar imágenes aumentadas ni fotogramas de una misma escena. Revisar alertas de duplicados.
2. Contar imágenes y **cajas** por split/clase, visualizar anotaciones reales y explicar una etiqueta normalizada.
3. **Transfer learning:** `yolo26n.pt`, `freeze=10`, configuración registrada.
4. **Fine tuning:** iniciar desde `best.pt` del transfer learning, `freeze=0`, optimizador `AdamW`, learning rate bajo; registrar GPU, tiempo, épocas y curvas.
5. Evaluar **ambos modelos en Validation**: precision, recall, mAP50, mAP50-95. Contabilizar TP/FP/FN del elegido a umbral de confianza explícito.
6. **Fijar la selección antes de Test** (máximo mAP50-95 en Validation; desempate por mAP50). El código registra una elección inmutable y bloquea repeticiones de Test.
7. Medir **Test una vez** con el mismo `imgsz` y hacer galería de detecciones/etiquetas. Comentar objeto perdido, falsa detección y caja mal colocada o confusión.
8. Probar imágenes y video externos. Documentar qué funciona, qué falla y limitaciones; no afirmar niveles de precisión no medidos.
9. Guardar `transfer_learning.pt`, `fine_tuning.pt`, `modelo_elegido.pt`; comprobar que cargan, y entregar notebook **ejecutado** y PDF **≤10 páginas**, sin portada/anexos (las referencias cuentan).

## Desarrollo local opcional (VS Code)

```bash
git clone https://github.com/MamatorHack/YOLO-Maquinaria-UBO.git
cd YOLO-Maquinaria-UBO
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

El entrenamiento se realiza en **Colab gratuito con GPU**, tal como exige la consigna; VS Code facilita edición, pruebas y colaboración.

## Entregables pendientes de producir con la GPU y las evidencias reales

- Notebook `.ipynb` **con todas sus salidas**.
- `models/transfer_learning.pt`, `models/fine_tuning.pt`, `models/modelo_elegido.pt`.
- PDF final en español siguiendo `FormatoEntregable.docx`, con cifras verificadas y ejemplos de errores.
- Fotos y video externos de procedencia documentada; dos videos ≥10s para Evaluación 04.

## Fuentes

- Consigna UBO: *Instrucciones Evaluacion 03.pdf* y *FormatoEntregable.docx* entregados en la asignatura.
- Dataset: https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2
- Documentación Ultralytics: https://docs.ultralytics.com/guides/finetuning-guide/
