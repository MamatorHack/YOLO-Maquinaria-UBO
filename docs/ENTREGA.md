# Lista de entrega — Machine Learning, Evaluación 03

**Universidad Bernardo O'Higgins · Fecha límite: sábado 7 de noviembre de 2026, 10:10**  
**Equipo:** Mathis Marsault, Julien Martinat y Louis Maillet  
**Docente:** Dr. Eliecer Peña Ancavil  
**Estado a 29/10/2026:** entrenamientos y evaluación ejecutados; informe PDF terminado y validado automáticamente (**7 páginas totales, una portada + 6 de informe**).

## Archivos para la entrega (una entrega por equipo)

- [x] **Notebook ejecutado:** [`notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb`](../notebooks/E03_YOLO_Marsault_Martinat_Maillet.ipynb). Incluye participantes, código, etapas de entrenamiento, resultados numéricos y salidas. Se han reducido registros extensos e imágenes incrustadas redundantes para GitHub; conservar también el archivo Colab original descargado antes de esta edición, si la plataforma exige absolutamente todas las salidas.
- [x] **Informe final PDF:** [`report/Informe_Tecnico_E03_YOLO.pdf`](../report/Informe_Tecnico_E03_YOLO.pdf). Generado desde [`INFORME_FINAL_E03.md`](../report/INFORME_FINAL_E03.md) y comprobado en CI: **6 páginas de contenido, referencias incluidas, más portada**. Se respeta el límite de 10 páginas sin contar portada.
- [x] **Tres pesos entrenados** presentes y validados en Colab: `transfer_learning.pt`, `fine_tuning.pt`, `modelo_elegido.pt`. El ZIP completo `YOLO_E03_resultados.zip` se descargó por el equipo (≈508,61 MiB). **No se ha publicado en GitHub** debido al tamaño. Se debe adjuntar en el aula virtual o dar enlace Drive visible al profesor.
- [x] **Resultados de dos vídeos externos** guardados en `results/video_1_streaming/` y `results/video_2_streaming/`, con 245 y 199 fotogramas anotados. Se encuentran en el ZIP de resultados del equipo, **no reproducibles desde GitHub** mientras los videos anotados no se alojen externamente.
- [x] **Auditoría de particiones:** código y [documento técnico](../docs/AUDITORIA_DATASET.md). El dataset **no se entrega**, solo fuente, versión, clases y licencia.

## Dos datos administrativos que todavía deben completarse

- [ ] **Número de grupo y módulo** exactos: falta confirmación en el foro UBO. Sustituir el marcador de la portada y de `report/INFORME_FINAL_E03.md`, luego regenerar el PDF.
- [ ] **URL de origen, autor/licencia de los dos vídeos Pexels realmente descargados**: los nombres de archivos `16517436_3840_2160_30fps.mp4` y `16499298_3840_2160_30fps.mp4` no acreditan por sí mismos sus páginas exactas. Añadir los enlaces originales al apartado Referencias antes de entregar.

## Evidencia mejorable para 25/25

- [ ] **Un error FN visible en Test** con imagen y caja real: el informe registra 45 FN en Validation y una imagen identificada por nombre, pero **no presenta un falso negativo concreto de Test visualmente inspeccionado**. Para cumplir al máximo la exigencia de tres errores del Test, extraer de las imágenes Test *ya utilizadas* un caso con objeto perdido (sin recalcular las métricas Test ni cambiar los pesos) y describir su contexto. Puede aprovecharse el código de visualización del notebook, sin reentrenar ni reoptimizar.
- [ ] Verificar que los **dos vídeos anotados** y al menos dos imágenes externas quedan accesibles para la defensa/Evaluación 04.
- [ ] Abrir manualmente el PDF final, revisar encabezados y figuras, y comprobar que sus enlaces funcionan.
- [ ] Comprobar los permisos de lectura del enlace Drive si la plataforma rechaza archivos `.pt`.

## Resumen experimental que debe permanecer inalterado

**Modelo elegido:** transfer learning, por mAP50-95 en Validation **0,712923** (fine tuning **0,707573**).  
**Test realizado una sola vez:** Precision **0,901339**, Recall **0,770506**, mAP50 **0,865897**, mAP50-95 **0,716560**.  
**Validación TP/FP/FN:** 165 / 35 / 45 a confianza 0,25 e IoU 0,50.

## Producir el PDF en una nueva máquina

```bash
python -m pip install "WeasyPrint>=66,<69" "Markdown>=3.7" "Pillow>=10" "pypdf>=5"
python scripts/build_report.py
```

El script recorta de forma automática las filas ilustrativas de la galería de Test y genera `report/Informe_Tecnico_E03_YOLO.pdf`; se detiene si excede **10 páginas de contenido**. El workflow [`build-report.yml`](../.github/workflows/build-report.yml) hace lo mismo en GitHub Actions y publica el PDF en el repositorio.

**No eliminar ni regenerar Test.** Cualquier modificación posterior que implique elegir un modelo con la información de Test incumple las instrucciones originales.
