# Curso de Introducción a la Minería de Datos con WEKA

## Bibliografía
1. [Witten] Data Mining: Practical Machine Learning Tools and Techniques. (http://www.cs.waikato.ac.nz/~ml/weka/book.html)
2. [Larose] Discovering Knowledge in Data.

## Descargas
### WEKA
Estaremos trabajando con la última versión estable (3.8)
http://www.cs.waikato.ac.nz/ml/weka/downloading.html

### Notas de curso
En notas/notas.pdf

## Compilación y ejecución de código
Para compilar solo se modifica el `classpath`.
```
javac -cp .:../weka.jar Clustering.java
```
Para ejecutar se modifica el `classpath` y se agregan los parámetros.
El código está pensado para ser utilizado en clase.
```
java -cp .:../weka.jar Clustering iris.arff
```

## Datasets
Se pueden descargar datasets desde:
1. http://www.cs.waikato.ac.nz/ml/weka/datasets.html
2. http://archive.ics.uci.edu/ml/datasets.html
También en el directorio `data` dentro de WEKA.
Aunque WEKA es capaz de leer distintos formatos, trabajaremos con el formato propio de WEKA (.arff).