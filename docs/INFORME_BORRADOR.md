# INFORME TÉCNICO — EVALUACIÓN 03 (BORRADOR, NO ENTREGAR COMO RESULTADO FINAL)

**Detección automática de maquinaria de construcción mediante YOLO**

**Integrantes:** Mathis Marsault, Julien Martinat y Louis Maillet  
**Grupo / módulo:** [COMPLETAR]  
**Asignatura:** Machine Learning  
**Docente:** Dr. Eliecer Peña Ancavil  
**Fecha:** 7 de noviembre de 2026

> Documento preparado siguiendo `FormatoEntregable.docx`. Los campos [PENDIENTE] requieren las salidas **reales** del notebook; no se inventan cifras. El PDF final deberá tener **máximo 10 páginas sin portada/anexos**, referencias incluidas. Se recomienda letra 11–12, figuras numeradas y pies interpretativos.

## 1. Problema y dataset

El proyecto aborda la detección automática de tres categorías de maquinaria pesada presentes en obras: excavadoras, camiones de volteo y cargadoras. Como aplicación potencial, un sistema experimental de clasificación visual de equipos puede ayudar a documentar escenas de construcción; **no** se han probado usos de seguridad operativa, control de acceso ni aplicaciones críticas.

**Fuente propuesta:** Roboflow Universe, *Excavators*, Mohamed Sabek, versión 2: https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2  
**Licencia:** CC BY 4.0 según la página de origen; [CONFIRMAR EN DESCARGA].  
**Categorías:** EXCAVATORS, dump truck, wheel loader [VERIFICAR CON data.yaml].  
**Video de aplicación:** imágenes de obras reales; demostrar con dos videos externos de al menos 10 segundos durante Evaluación 04. Fuentes y licencias en referencias.

**Tabla 1. Composición del dataset.** Completar a partir de `results`:

| Partición | Imágenes | Cajas excavator | Cajas dump truck | Cajas wheel loader | Total cajas |
|---|---:|---:|---:|---:|---:|
| Train | [PENDIENTE] | — | — | — | — |
| Validation | [PENDIENTE] | — | — | — | — |
| Test | [PENDIENTE] | — | — | — | — |

## 2. Preparación de los datos

Las imágenes se organizan en `images/` y sus anotaciones en `labels/`, con el formato YOLO de cinco campos: identificador de clase, centro x, centro y, ancho y alto, con valores normalizados. El archivo `data.yaml` enlaza las tres particiones y las categorías. Se mantienen las separaciones originales para evitar que fotos modificadas o cuadros de una misma escena pasen de Train a Test. Se comprueban duplicados exactos mediante SHA256 y se efectúa una inspección visual para detectar transformaciones que los hashes no identifiquen.

**Figura 1.** Ejemplos de imágenes y cajas reales de entrenamiento [INSERTAR results/annotations_train.png].  
**Ejemplo de etiqueta:** [COPIAR UNA LÍNEA REAL Y EXPLICAR SUS 5 VALORES].  
**Auditoría:** [INSERTAR NÚMERO DE DUPLICADOS Y RESULTADO DE INSPECCIÓN MANUAL].

## 3. Entrenamiento

Primero se realiza transfer learning sobre un detector YOLO preentrenado (`yolo26n.pt` o `yolo11n.pt`) congelando diez capas (`freeze=10`). A continuación se inicia fine tuning desde el archivo `best.pt` generado en la primera etapa, sin capas congeladas (`freeze=0`), con optimizador AdamW y tasa de aprendizaje reducida. Ambos modelos se entrenan exclusivamente con Train; Validation se utiliza para seleccionar y analizar.

**Tabla 2. Configuración y recursos.**

| Característica | Transfer learning | Fine tuning |
|---|---|---|
| Pesos iniciales | [PENDIENTE] | best.pt del transfer |
| Épocas | [PENDIENTE] | [PENDIENTE] |
| Tamaño de imagen | [PENDIENTE] | [PENDIENTE] |
| Fracción Train | [PENDIENTE] | [PENDIENTE] |
| Freeze | 10 | 0 |
| Optimizador | [PENDIENTE] | AdamW |
| LR inicial | [PENDIENTE] | [PENDIENTE] |
| GPU | [PENDIENTE] | [PENDIENTE] |
| Duración | [PENDIENTE] | [PENDIENTE] |
| Mejor época | [PENDIENTE] | [PENDIENTE] |

**Figuras 2 y 3.** Pérdidas y mAP por época, junto con la interpretación: [INSERTAR resultados/transfer_curves.png y fine_curves.png, comentar sobreajuste/convergencia sin inventarlo].

## 4. Resultados en Validation y elección del modelo

Se calcula precision, recall, mAP50 y mAP50-95 para ambos modelos en Validation, con el mismo tamaño de imagen. **Regla definida antes de mirar Test:** mayor mAP50-95 en Validation; si hay empate, mayor mAP50; si persiste, transfer learning.

**Tabla 3. Resultados Validation.**

| Modelo | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Transfer learning | [PENDIENTE] | — | — | — |
| Fine tuning | [PENDIENTE] | — | — | — |

**Modelo elegido:** [PENDIENTE].  
**TP / FP / FN en Validation:** [PENDIENTE] a confianza 0,25 e IoU 0,50; interpretar qué suponen para conteos de maquinaria. La asignación TP/FP/FN es por coincidencia de clase y solapamiento, y no reemplaza la evaluación estándar mAP.

## 5. Evaluación en Test y análisis de errores

La evaluación Test se ejecuta **una sola vez**, después de seleccionar el modelo y al mismo tamaño de imagen usado en entrenamiento.

**Tabla 4. Resultados Test:** precision [PENDIENTE], recall [PENDIENTE], mAP50 [PENDIENTE], mAP50-95 [PENDIENTE]. Comparar razonadamente con Validation.  
**Figura 4.** Cajas reales y detectadas sobre imágenes de Test [INSERTAR GALERÍA DEFINITIVA].

**Tres errores obligatorios con imagen, explicación y posible causa:**
1. **Objeto perdido:** [PENDIENTE: foto, clase, contexto, posible causa].
2. **Falsa detección:** [PENDIENTE: foto, clase, posible causa].
3. **Caja mal ubicada o clase confundida:** [PENDIENTE: foto, medidas y posible causa].

## 6. Uso con imágenes y video externos

**Figura 5.** Detecciones correctas y fallidas sobre fotografías exteriores al dataset [PENDIENTE].  
**Figura 6.** Fotograma representativo de un video externo [PENDIENTE].  
Interpretar iluminación, tamaño del objeto, oclusiones, perspectiva y posibles cambios de dominio. Identificar fuentes y condiciones de uso.

## 7. Conclusiones y límites

Con base en los resultados reales, responder: qué maquinaria detecta, con qué alcance, dónde falla y por qué. Señalar tamaño/variabilidad del dataset, similitud visual entre clases, objetos pequeños, condiciones de iluminación y las limitaciones de la GPU T4. Como mejoras propuestas **sin afirmar que ya se implementaron**, evaluar más ejemplos, nuevas condiciones de obra, más épocas controladas por Validation y revisión de etiquetas.

## Referencias y anexos

- Universidad Bernardo O'Higgins. *Instrucciones Evaluacion 03.pdf* y *FormatoEntregable.docx*, material docente.
- Mohamed Sabek. *Excavators*, dataset v2, Roboflow Universe. https://universe.roboflow.com/mohamed-sabek-6zmr6/excavators-cwlh0/dataset/2
- Ultralytics. Documentación del detector YOLO. https://docs.ultralytics.com/
- [COMPLETAR] Créditos de cada imagen y video externo realmente utilizados.

**Anexos opcionales:** resultados de diagnóstico, galerías adicionales, registro de entrenamiento. Toda evidencia necesaria para puntuar debe quedar en el cuerpo del informe.
