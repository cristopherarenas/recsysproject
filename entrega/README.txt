Proyecto RecSys
===============
Cristopher Arenas

Se incluyen los siguientes archivos:
- paper.pdf: incluye el reporte final del proyecto
- slides_handout.pdf: es la presentacion

La carpeta files se compone de:
- data.py contiene las funciones en python implementadas para leer datos y ejecutar el metodo del gradiente descendente. Genera un archivo donde se guardan valores reales y predichos para el conjunto de prueba.
- analisis.py: implementa las metricas utilizadas y graficos.
- GDM_1_100.txt, GDM_3_100.txt, GDM_10_100.txt: muestra el error al ejecutar GDM por 100 iteraciones usando 1, 3 y 10 factores latentes. Estos datos se graficaron en la Figura 1.
- predicted1_wtag.txt: muestra los valores reales y predichos para los ratings del conjunto de prueba sin uso de tags.
- predicted1_tag.txt: muestra los valores reales y predichos para los ratings del conjunto de prueba con uso de tags.
- r1.train: conjunto de entrenamiento usado.
- r1.test: conjunto de prueba usado.
